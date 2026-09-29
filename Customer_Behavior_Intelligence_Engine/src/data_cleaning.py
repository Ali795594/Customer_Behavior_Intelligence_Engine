"""
data_cleaning.py
-----------------
Cleans the raw customer transactions dataset.

Documented decisions:
- Exact duplicate rows are dropped.
- Negative unit_price values are data-entry errors -> absolute value.
- Rows still missing unit_price are dropped (price drives revenue and AOV
  features, so imputing it would distort customer-level metrics).
- Missing discount -> assume no discount (0).
- Missing payment_method / region -> labeled "Unknown" rather than dropped,
  since these are categorical context fields, not core numeric features.
- Anomalous quantity spikes (top 0.1%) are winsorized, not dropped, since a
  legitimate bulk purchase is plausible but shouldn't dominate aggregates.
"""

import numpy as np
import pandas as pd


def validate(df: pd.DataFrame) -> dict:
    report = {
        "n_rows": len(df),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "n_duplicate_transaction_ids": int(df["transaction_id"].duplicated().sum()),
        "missing_values": df.isna().sum().to_dict(),
        "negative_unit_price_count": int((df["unit_price"] < 0).sum()),
        "n_customers": df["customer_id"].nunique(),
    }
    return report


def print_report(report: dict):
    print("=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)
    for key, value in report.items():
        if key == "missing_values":
            print("\nMissing values by column:")
            for col, n in value.items():
                if n > 0:
                    print(f"  - {col}: {n}")
        else:
            print(f"{key}: {value}")
    print("=" * 60)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    n_dupes_removed = before - len(df)

    df["unit_price"] = df["unit_price"].abs()
    df = df.dropna(subset=["unit_price"])
    df = df[df["quantity"] > 0]

    df["discount"] = df["discount"].fillna(0.0).clip(0, 1)
    df["payment_method"] = df["payment_method"].fillna("Unknown")
    df["region"] = df["region"].fillna("Unknown")

    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    # Winsorize extreme quantity anomalies
    low, high = df["quantity"].quantile([0.0, 0.999])
    df["quantity_raw"] = df["quantity"]
    df["quantity"] = df["quantity"].clip(lower=low, upper=high)

    # Recompute revenue for consistency
    df["revenue"] = (df["quantity"] * df["unit_price"] * (1 - df["discount"])).round(2)

    df = df.reset_index(drop=True)

    print(f"Removed {n_dupes_removed} duplicate rows.")
    print(f"Rows after cleaning: {len(df):,} (from {before:,})")
    print(f"Unique customers after cleaning: {df['customer_id'].nunique():,}")

    return df


def build_sql_export(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe renamed/ordered to match sql/01_schema.sql."""
    sql_df = df.rename(columns={"transaction_date": "txn_date"})[
        [
            "transaction_id", "customer_id", "product_id", "category",
            "txn_date", "quantity", "unit_price", "discount",
            "payment_method", "region", "revenue",
        ]
    ]
    return sql_df


if __name__ == "__main__":
    from utils import RAW_DATA_PATH, CLEAN_DATA_PATH, BASE_DIR, ensure_dirs
    import os

    ensure_dirs()
    raw = pd.read_csv(RAW_DATA_PATH)
    print_report(validate(raw))
    cleaned = clean(raw)
    cleaned.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Saved cleaned dataset -> {CLEAN_DATA_PATH}")

    sql_export_path = os.path.join(BASE_DIR, "data", "processed", "customer_transactions_sql.csv")
    build_sql_export(cleaned).to_csv(sql_export_path, index=False)
    print(f"Saved SQL-ready export -> {sql_export_path}")
