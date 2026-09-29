"""
modeling.py
------------
Trains a Logistic Regression model (scikit-learn) to predict purchase
propensity (see propensity_model.py for how the leakage-safe target is
built), using a standard random train/test split since each row here is
an independent customer snapshot at a single cutoff date (no temporal
ordering issue at the row level, unlike Project 1's per-transaction data).

Evaluation uses confusion matrix, precision, recall, F1, and ROC-AUC -
NOT accuracy alone, since propensity targets are often imbalanced and the
business cares differently about false positives vs false negatives
(see README for the cost discussion).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report,
)

from propensity_model import build_model_matrix

RANDOM_STATE = 42


def train_and_evaluate(df: pd.DataFrame):
    X, y, feature_cols = build_model_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    coef_table = (
        pd.Series(model.coef_[0], index=X.columns)
        .sort_values(key=abs, ascending=False)
        .rename("coefficient")
        .to_frame()
    )

    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(cm)
    print(f"\nROC-AUC: {metrics['roc_auc']:.3f}")

    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "confusion_matrix": cm,
        "roc_curve": (fpr, tpr),
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "coefficients": coef_table,
        "feature_columns": list(X.columns),
    }


if __name__ == "__main__":
    from utils import BASE_DIR
    import os

    df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "propensity_dataset.csv"))
    results = train_and_evaluate(df)

    print("\nTop 10 standardized coefficients by magnitude:")
    print(results["coefficients"].head(10))
