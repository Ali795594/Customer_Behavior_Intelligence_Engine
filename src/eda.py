"""
eda.py
-------
Generates core EDA charts for the customer behavior project and saves
them to visualizations/.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_theme(style="whitegrid")


def plot_monetary_distribution(features, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(np.log1p(features["monetary_value"]), bins=40, ax=ax, color="#4C72B0")
    ax.set_title("Distribution of log(1 + Customer Monetary Value)")
    ax.set_xlabel("log(1 + total revenue)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "monetary_value_distribution.png"), dpi=120)
    plt.close(fig)


def plot_frequency_distribution(features, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(features["frequency"], bins=40, ax=ax, color="#DD8452")
    ax.set_title("Distribution of Purchase Frequency")
    ax.set_xlabel("Number of transactions")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "frequency_distribution.png"), dpi=120)
    plt.close(fig)


def plot_recency_distribution(features, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(features["recency"], bins=40, ax=ax, color="#55A868")
    ax.set_title("Distribution of Recency (days since last purchase)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "recency_distribution.png"), dpi=120)
    plt.close(fig)


def plot_aov_vs_frequency(features, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    sample = features.sample(min(3000, len(features)), random_state=42)
    sns.scatterplot(
        data=sample, x="frequency", y="average_order_value",
        alpha=0.4, s=18, ax=ax, hue="preferred_category"
    )
    ax.set_title("Average Order Value vs Purchase Frequency")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "aov_vs_frequency.png"), dpi=120)
    plt.close(fig)


def plot_customer_concentration(features, out_dir):
    """Pareto-style chart: cumulative % of revenue by cumulative % of customers."""
    sorted_rev = features["monetary_value"].sort_values(ascending=False).reset_index(drop=True)
    cum_rev_pct = sorted_rev.cumsum() / sorted_rev.sum() * 100
    cum_cust_pct = (sorted_rev.index + 1) / len(sorted_rev) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(cum_cust_pct, cum_rev_pct, color="#C44E52")
    ax.axhline(80, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Cumulative % of Customers")
    ax.set_ylabel("Cumulative % of Revenue")
    ax.set_title("Customer Revenue Concentration (Pareto Curve)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "customer_concentration_pareto.png"), dpi=120)
    plt.close(fig)


def plot_category_preference(features, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = features["preferred_category"].value_counts()
    sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#8172B2")
    ax.set_xlabel("Number of Customers")
    ax.set_title("Customers by Preferred Category")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "category_preference.png"), dpi=120)
    plt.close(fig)


def plot_regional_behavior(transactions, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    rev = transactions.groupby("region")["revenue"].sum().sort_values(ascending=False)
    sns.barplot(x=rev.index, y=rev.values, ax=ax, color="#937860")
    ax.set_ylabel("Total Revenue")
    ax.set_title("Total Revenue by Region")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "regional_revenue.png"), dpi=120)
    plt.close(fig)


def run_eda(features: pd.DataFrame, transactions: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    plot_monetary_distribution(features, out_dir)
    plot_frequency_distribution(features, out_dir)
    plot_recency_distribution(features, out_dir)
    plot_aov_vs_frequency(features, out_dir)
    plot_customer_concentration(features, out_dir)
    plot_category_preference(features, out_dir)
    plot_regional_behavior(transactions, out_dir)
    print(f"Saved 7 EDA charts -> {out_dir}")


if __name__ == "__main__":
    from utils import CUSTOMER_FEATURES_PATH, CLEAN_DATA_PATH, VIZ_DIR

    features = pd.read_csv(CUSTOMER_FEATURES_PATH)
    transactions = pd.read_csv(CLEAN_DATA_PATH)
    run_eda(features, transactions, VIZ_DIR)
