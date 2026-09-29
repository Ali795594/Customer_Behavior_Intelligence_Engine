"""
customer_features.py
----------------------
Builds customer-level behavioral features (RFM and beyond) from cleaned
transaction-level data, as of a given snapshot date.

Features:
    recency               days since the customer's most recent purchase
                           (relative to snapshot_date)
    frequency             number of transactions
    monetary_value        total revenue
    average_order_value   monetary_value / frequency
    total_quantity        total units purchased
    category_diversity    number of distinct categories purchased from
    average_discount      mean discount received
    purchase_frequency    transactions per active month
    active_months         number of distinct calendar months with a purchase
    preferred_category    most frequently purchased category
    tenure_days           days between first and last purchase
    preferred_payment     most common payment method
"""

import pandas as pd


def build_customer_features(
    df: pd.DataFrame, snapshot_date: str, as_of: str | None = None
) -> pd.DataFrame:
    """
    Build RFM+ features using only transactions on/before `as_of` (defaults
    to snapshot_date). Passing a separate `as_of` lets propensity_model.py
    build features as of an earlier cutoff, while `snapshot_date` (usually
    "now") is used to compute recency.
    """
    if as_of is None:
        as_of = snapshot_date

    df = df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    snapshot = pd.Timestamp(snapshot_date)
    cutoff = pd.Timestamp(as_of)

    df = df[df["transaction_date"] <= cutoff]
    if df.empty:
        return pd.DataFrame()

    df["year_month"] = df["transaction_date"].dt.to_period("M")

    grouped = df.groupby("customer_id")

    features = grouped.agg(
        last_purchase_date=("transaction_date", "max"),
        first_purchase_date=("transaction_date", "min"),
        frequency=("transaction_id", "count"),
        monetary_value=("revenue", "sum"),
        total_quantity=("quantity", "sum"),
        category_diversity=("category", pd.Series.nunique),
        average_discount=("discount", "mean"),
        active_months=("year_month", pd.Series.nunique),
    ).reset_index()

    features["recency"] = (snapshot - features["last_purchase_date"]).dt.days
    features["tenure_days"] = (
        features["last_purchase_date"] - features["first_purchase_date"]
    ).dt.days
    features["average_order_value"] = (
        features["monetary_value"] / features["frequency"]
    ).round(2)
    features["purchase_frequency"] = (
        features["frequency"] / features["active_months"].clip(lower=1)
    ).round(3)

    preferred_category = (
        df.groupby(["customer_id", "category"])
        .size()
        .reset_index(name="n")
        .sort_values(["customer_id", "n"], ascending=[True, False])
        .drop_duplicates("customer_id")[["customer_id", "category"]]
        .rename(columns={"category": "preferred_category"})
    )
    preferred_payment = (
        df.groupby(["customer_id", "payment_method"])
        .size()
        .reset_index(name="n")
        .sort_values(["customer_id", "n"], ascending=[True, False])
        .drop_duplicates("customer_id")[["customer_id", "payment_method"]]
        .rename(columns={"payment_method": "preferred_payment"})
    )

    features = features.merge(preferred_category, on="customer_id", how="left")
    features = features.merge(preferred_payment, on="customer_id", how="left")

    features["monetary_value"] = features["monetary_value"].round(2)
    features["average_discount"] = features["average_discount"].round(4)

    cols = [
        "customer_id", "recency", "frequency", "monetary_value",
        "average_order_value", "total_quantity", "category_diversity",
        "average_discount", "purchase_frequency", "active_months",
        "tenure_days", "preferred_category", "preferred_payment",
        "first_purchase_date", "last_purchase_date",
    ]
    return features[cols]


if __name__ == "__main__":
    from utils import CLEAN_DATA_PATH, CUSTOMER_FEATURES_PATH, SNAPSHOT_DATE

    df = pd.read_csv(CLEAN_DATA_PATH)
    features = build_customer_features(df, snapshot_date=SNAPSHOT_DATE)
    features.to_csv(CUSTOMER_FEATURES_PATH, index=False)
    print(f"Built features for {len(features):,} customers -> {CUSTOMER_FEATURES_PATH}")
    print(features.describe(include="all").T)
