"""
statistical_analysis.py
-------------------------
Uses statsmodels Logit to quantify the STATISTICAL ASSOCIATION between
customer behavioral features and purchase propensity, reporting
coefficients, standard errors, p-values, and confidence intervals.

IMPORTANT: These coefficients describe association within this
observational dataset, not a proven causal effect (e.g. we cannot claim
that artificially increasing a customer's "purchase_frequency" would
cause the modeled increase in propensity - frequency may itself be a
symptom of unobserved factors like product fit or satisfaction).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_logit(df: pd.DataFrame, feature_cols: list, target_col: str = "purchased_next_30_days"):
    X = df[feature_cols].copy()
    # Standardize for comparable coefficient magnitudes
    X = (X - X.mean()) / X.std(ddof=0)
    X = sm.add_constant(X)
    y = df[target_col]

    model = sm.Logit(y, X)
    result = model.fit(disp=False)
    return result


def summarize(result) -> pd.DataFrame:
    conf_int = result.conf_int()
    conf_int.columns = ["ci_lower", "ci_upper"]
    odds_ratio = np.exp(result.params)

    summary = pd.DataFrame(
        {
            "coefficient": result.params,
            "std_err": result.bse,
            "z_value": result.tvalues,
            "p_value": result.pvalues,
            "odds_ratio": odds_ratio,
        }
    ).join(conf_int)
    return summary


if __name__ == "__main__":
    from utils import BASE_DIR
    import os

    df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "propensity_dataset.csv"))

    feature_cols = [
        "recency", "frequency", "monetary_value", "average_order_value",
        "category_diversity", "average_discount", "purchase_frequency",
        "active_months", "tenure_days",
    ]

    result = fit_logit(df, feature_cols)
    summary = summarize(result)

    print(f"Pseudo R-squared: {result.prsquared:.4f}")
    print(f"Log-Likelihood: {result.llf:.2f}    LL-Null: {result.llnull:.2f}")
    print(f"LLR p-value: {result.llr_pvalue:.6g}\n")
    print(summary.round(4))
