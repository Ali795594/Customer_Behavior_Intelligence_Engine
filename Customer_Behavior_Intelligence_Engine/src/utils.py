"""Shared path/IO helpers for the customer behavior intelligence project."""

import os

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "customer_transactions.csv")
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "customer_transactions_clean.csv")
CUSTOMER_FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "customer_features.csv")
FINAL_CUSTOMER_PATH = os.path.join(BASE_DIR, "data", "processed", "final_customer_dataset.csv")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")

# The date we treat as "today" for recency calculations - one day after the
# last date in the synthetic dataset's generation window.
SNAPSHOT_DATE = "2024-01-01"

# Propensity-modeling cutoff: features are computed using only transactions
# on/before this date; the label looks at the FOLLOWING 30 days. This keeps
# the two windows non-overlapping and avoids leakage.
FEATURE_CUTOFF_DATE = "2023-11-01"
LABEL_WINDOW_DAYS = 30


def ensure_dirs():
    os.makedirs(os.path.dirname(CLEAN_DATA_PATH), exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)
