"""
segmentation.py
-----------------
Customer segmentation using K-Means clustering on RFM+ features.

Methodology:
1. Select features that describe purchase behavior (recency, frequency,
   monetary value, AOV, category diversity, average discount).
2. Log-transform monetary_value and average_order_value (heavily
   right-skewed) before scaling.
3. Standardize features (K-Means is distance-based and sensitive to scale).
4. Try multiple candidate K values (2-8) and compare silhouette scores
   instead of arbitrarily picking one.
5. Fit the chosen K, then profile each cluster using the ORIGINAL
   (unscaled) feature values so segment descriptions are interpretable.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

CLUSTER_FEATURES = [
    "recency", "frequency", "monetary_value", "average_order_value",
    "category_diversity", "average_discount",
]
RANDOM_STATE = 42


def prepare_features(features: pd.DataFrame) -> pd.DataFrame:
    X = features[CLUSTER_FEATURES].copy()
    # Log-transform skewed monetary columns (add 1 to handle zeros safely)
    X["monetary_value"] = np.log1p(X["monetary_value"])
    X["average_order_value"] = np.log1p(X["average_order_value"])
    return X


def choose_k(X_scaled: np.ndarray, k_range=range(2, 9)) -> pd.DataFrame:
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette_score": score})
    return pd.DataFrame(rows)


def fit_kmeans(features: pd.DataFrame, k: int):
    X = prepare_features(features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X_scaled)

    result = features.copy()
    result["cluster"] = labels
    return result, model, scaler, X_scaled


def profile_clusters(clustered_features: pd.DataFrame) -> pd.DataFrame:
    """Describe each cluster using ORIGINAL (unscaled) feature values."""
    profile = clustered_features.groupby("cluster")[CLUSTER_FEATURES].mean().round(2)
    profile["n_customers"] = clustered_features.groupby("cluster").size()
    profile["pct_of_customers"] = (
        profile["n_customers"] / profile["n_customers"].sum() * 100
    ).round(1)
    return profile.sort_values("monetary_value", ascending=False)


def name_segments(profile: pd.DataFrame) -> dict:
    """
    Derive a human-readable segment label from actual cluster statistics
    (not invented) using simple relative thresholds against the overall
    median of each metric.
    """
    names = {}
    monetary_median = profile["monetary_value"].median()
    frequency_median = profile["frequency"].median()
    recency_median = profile["recency"].median()

    for cluster_id, row in profile.iterrows():
        high_value = row["monetary_value"] >= monetary_median
        high_freq = row["frequency"] >= frequency_median
        recent = row["recency"] <= recency_median

        if high_value and high_freq and recent:
            label = "Champions (high value, frequent, recent)"
        elif high_value and not recent:
            label = "At-Risk High Value (used to spend a lot, going quiet)"
        elif high_freq and not high_value:
            label = "Frequent Low-Spend Shoppers"
        elif not high_freq and not high_value and recent:
            label = "New / Low-Engagement Customers"
        elif not recent and not high_freq:
            label = "Dormant / Churned"
        else:
            label = "Regular Customers"
        names[cluster_id] = label
    return names


if __name__ == "__main__":
    from utils import CUSTOMER_FEATURES_PATH, BASE_DIR
    import os

    features = pd.read_csv(CUSTOMER_FEATURES_PATH)
    X = prepare_features(features)
    X_scaled = StandardScaler().fit_transform(X)

    k_table = choose_k(X_scaled)
    print("Candidate K values and silhouette scores:")
    print(k_table.to_string(index=False))

    best_k = int(k_table.loc[k_table["silhouette_score"].idxmax(), "k"])
    print(f"\nSelected K = {best_k} (highest silhouette score)")

    clustered, model, scaler, _ = fit_kmeans(features, best_k)
    profile = profile_clusters(clustered)
    names = name_segments(profile)
    profile["segment_name"] = profile.index.map(names)

    print("\nCluster profile:")
    print(profile.to_string())

    clustered["segment_name"] = clustered["cluster"].map(names)

    out_path = os.path.join(BASE_DIR, "data", "processed", "customer_segments.csv")
    clustered.to_csv(out_path, index=False)
    print(f"\nSaved segmented customer dataset -> {out_path}")
