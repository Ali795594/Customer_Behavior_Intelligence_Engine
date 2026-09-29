"""
propensity_model.py
----------------------
Predicts purchase propensity: will an existing customer make at least one
purchase in the 30 days AFTER a fixed cutoff date?

LEAKAGE PREVENTION (important):
- Features (recency, frequency, monetary_value, etc.) are computed using
  ONLY transactions on or before FEATURE_CUTOFF_DATE.
- The label is computed using ONLY transactions strictly AFTER
  FEATURE_CUTOFF_DATE and within the next LABEL_WINDOW_DAYS days.
- These two windows never overlap, so no information from the label
  window leaks into the features. This is a temporal, not random,
  train/test methodology; the model is evaluated on data that comes
  chronologically after the features it was trained on.
- The customer population is restricted to customers who had already
  transacted at least once as of the cutoff (a propensity model for
  "existing" customers); brand-new customers who first appear after the
  cutoff would have no pre-cutoff features to score them on.
"""

import numpy as np
import pandas as pd

from customer_features import build_customer_features


def build_propensity_dataset(
    transactions: pd.DataFrame,
    feature_cutoff: str,
    label_window_days: int,
) -> pd.DataFrame:
    transactions = transactions.copy()
    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])

    cutoff = pd.Timestamp(feature_cutoff)
    label_end = cutoff + pd.Timedelta(days=label_window_days)

    # Features: snapshot_date = cutoff itself, so recency is "as of cutoff"
    features = build_customer_features(
        transactions, snapshot_date=feature_cutoff, as_of=feature_cutoff
    )

    # Label: did the customer purchase again in (cutoff, label_end]?
    future_txns = transactions[
        (transactions["transaction_date"] > cutoff)
        & (transactions["transaction_date"] <= label_end)
    ]
    purchasers = set(future_txns["customer_id"].unique())
    features["purchased_next_30_days"] = features["customer_id"].isin(purchasers).astype(int)

    return features


def build_model_matrix(df: pd.DataFrame):
    feature_cols = [
        "recency", "frequency", "monetary_value", "average_order_value",
        "category_diversity", "average_discount", "purchase_frequency",
        "active_months", "tenure_days",
    ]
    X = df[feature_cols].copy()
    if "cluster" in df.columns:
        X["cluster"] = df["cluster"]
        X = pd.get_dummies(X, columns=["cluster"], prefix="cluster", drop_first=True)
    y = df["purchased_next_30_days"].copy()
    return X, y, feature_cols


if __name__ == "__main__":
    from utils import CLEAN_DATA_PATH, FEATURE_CUTOFF_DATE, LABEL_WINDOW_DAYS, BASE_DIR
    import os

    transactions = pd.read_csv(CLEAN_DATA_PATH)
    propensity_df = build_propensity_dataset(
        transactions, FEATURE_CUTOFF_DATE, LABEL_WINDOW_DAYS
    )

    print(f"Customers with pre-cutoff history: {len(propensity_df):,}")
    print(f"Positive rate (purchased in next {LABEL_WINDOW_DAYS} days): "
          f"{propensity_df['purchased_next_30_days'].mean():.3f}")

    out_path = os.path.join(BASE_DIR, "data", "processed", "propensity_dataset.csv")
    propensity_df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
