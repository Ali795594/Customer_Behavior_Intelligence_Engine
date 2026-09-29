"""
build_final_dataset.py
------------------------
Combines segmentation + a FULL-HISTORY propensity score (scored using all
available data up to the true end of the dataset, unlike the leakage-safe
train/test split used for model evaluation) into the single final
customer-level output table requested by the business:

    customer_id, segment, recency, frequency, monetary_value, AOV,
    purchase_probability
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from customer_features import build_customer_features
from propensity_model import build_propensity_dataset, build_model_matrix
from segmentation import fit_kmeans, profile_clusters, name_segments


def build_final_dataset(
    transactions: pd.DataFrame,
    snapshot_date: str,
    feature_cutoff: str,
    label_window_days: int,
    best_k: int,
):
    # 1. Segment ALL customers using full-history features (as of snapshot_date)
    all_features = build_customer_features(transactions, snapshot_date=snapshot_date)
    clustered, _, _, _ = fit_kmeans(all_features, best_k)
    profile = profile_clusters(clustered)
    names = name_segments(profile)
    clustered["segment"] = clustered["cluster"].map(names)

    # 2. Train the propensity model on the leakage-safe historical window,
    #    then SCORE current customers using their up-to-date features.
    train_df = build_propensity_dataset(transactions, feature_cutoff, label_window_days)
    X_train, y_train, feature_cols = build_model_matrix(train_df)
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(scaler.transform(X_train), y_train)

    score_X = all_features[feature_cols]
    all_features["purchase_probability"] = model.predict_proba(scaler.transform(score_X))[:, 1]

    final = clustered.merge(
        all_features[["customer_id", "purchase_probability"]], on="customer_id", how="left"
    )

    final = final.rename(columns={"average_order_value": "AOV"})
    return final[
        ["customer_id", "segment", "recency", "frequency", "monetary_value",
         "AOV", "purchase_probability"]
    ]


if __name__ == "__main__":
    from utils import (
        CLEAN_DATA_PATH, FINAL_CUSTOMER_PATH, SNAPSHOT_DATE,
        FEATURE_CUTOFF_DATE, LABEL_WINDOW_DAYS,
    )
    from segmentation import prepare_features, choose_k
    from sklearn.preprocessing import StandardScaler as _StandardScaler

    transactions = pd.read_csv(CLEAN_DATA_PATH)

    # Pick K the same data-driven way as segmentation.py (highest silhouette)
    _all_features = build_customer_features(transactions, snapshot_date=SNAPSHOT_DATE)
    _X_scaled = _StandardScaler().fit_transform(prepare_features(_all_features))
    _k_table = choose_k(_X_scaled)
    best_k = int(_k_table.loc[_k_table["silhouette_score"].idxmax(), "k"])
    print(f"Selected K={best_k} based on silhouette score.")

    final_df = build_final_dataset(
        transactions,
        snapshot_date=SNAPSHOT_DATE,
        feature_cutoff=FEATURE_CUTOFF_DATE,
        label_window_days=LABEL_WINDOW_DAYS,
        best_k=best_k,
    )
    final_df.to_csv(FINAL_CUSTOMER_PATH, index=False)
    print(f"Saved final customer dataset ({len(final_df):,} rows) -> {FINAL_CUSTOMER_PATH}")
    print(final_df.head(10))
