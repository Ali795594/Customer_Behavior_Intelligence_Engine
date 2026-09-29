"""
generate_data.py
-----------------
Generates a realistic SYNTHETIC customer transactions dataset for the
Customer Behavior Intelligence Engine.

Run:
    python src/generate_data.py

Output:
    data/raw/customer_transactions.csv

Deterministic (fixed random seed). Deliberately contains missing values,
duplicate rows, anomalies, and highly varying purchase frequency per
customer (some one-time buyers, some frequent repeat buyers) so downstream
RFM / segmentation / propensity modeling has something realistic to work
with.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RANDOM_SEED = 7
N_CUSTOMERS = 5_600
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2023, 12, 31)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "customer_transactions.csv")

CATEGORIES = [
    "Electronics", "Home & Kitchen", "Apparel",
    "Beauty & Personal Care", "Sports & Outdoors", "Grocery",
]
REGIONS = ["North", "South", "East", "West", "Central"]
PAYMENT_METHODS = ["Credit Card", "Debit Card", "PayPal", "Gift Card", "Cash on Delivery"]

# Category base price ranges (kept consistent with the pricing project for realism)
CATEGORY_PRICE_RANGE = {
    "Electronics": (60, 700),
    "Home & Kitchen": (12, 180),
    "Apparel": (10, 130),
    "Beauty & Personal Care": (5, 80),
    "Sports & Outdoors": (12, 250),
    "Grocery": (2, 35),
}


def build_customers(rng):
    """
    Assign each customer a latent 'purchase propensity profile' that drives
    how often they buy, how much they spend, and how loyal they are to a
    category. This creates realistic heterogeneity for RFM/segmentation.
    """
    customer_ids = [f"C{100000+i}" for i in range(N_CUSTOMERS)]

    # Customer archetypes with different base purchase-rate and spend levels
    archetypes = rng.choice(
        ["one_time", "occasional", "regular", "frequent", "vip"],
        size=N_CUSTOMERS,
        p=[0.22, 0.33, 0.25, 0.14, 0.06],
    )
    archetype_lambda = {  # avg number of transactions over the 2-year window
        "one_time": 1.0,
        "occasional": 4.0,
        "regular": 11.0,
        "frequent": 22.0,
        "vip": 38.0,
    }
    archetype_spend_mult = {
        "one_time": 0.9,
        "occasional": 1.0,
        "regular": 1.15,
        "frequent": 1.3,
        "vip": 1.8,
    }

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "archetype": archetypes,
            "n_transactions_lambda": [archetype_lambda[a] for a in archetypes],
            "spend_multiplier": [archetype_spend_mult[a] for a in archetypes],
            "preferred_category": rng.choice(CATEGORIES, size=N_CUSTOMERS),
            "home_region": rng.choice(REGIONS, size=N_CUSTOMERS),
            "signup_offset_days": rng.integers(0, 500, size=N_CUSTOMERS),
        }
    )
    return customers


def generate(seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    customers = build_customers(rng)
    date_span = (END_DATE - START_DATE).days

    rows = []
    txn_counter = 1

    for _, cust in customers.iterrows():
        n_txns = max(0, rng.poisson(lam=cust["n_transactions_lambda"]))
        if n_txns == 0:
            continue

        # A customer's transactions are spread across the period AFTER signup
        earliest_day = min(cust["signup_offset_days"], date_span)
        txn_days = rng.integers(earliest_day, date_span + 1, size=n_txns)
        txn_days.sort()

        for day in txn_days:
            date = START_DATE + timedelta(days=int(day))

            # 70% chance of buying from preferred category, else random
            if rng.random() < 0.7:
                category = cust["preferred_category"]
            else:
                category = rng.choice(CATEGORIES)

            price_low, price_high = CATEGORY_PRICE_RANGE[category]
            unit_price = round(rng.uniform(price_low, price_high) * cust["spend_multiplier"], 2)

            quantity = int(max(1, rng.poisson(lam=1.6)))
            discount = float(np.clip(rng.beta(2, 10), 0, 0.6))
            payment_method = rng.choice(PAYMENT_METHODS, p=[0.42, 0.28, 0.16, 0.06, 0.08])

            # Region is usually home region, occasionally travel/gift purchase
            region = cust["home_region"] if rng.random() < 0.85 else rng.choice(REGIONS)

            revenue = round(quantity * unit_price * (1 - discount), 2)

            rows.append(
                {
                    "transaction_id": f"T{txn_counter:07d}",
                    "customer_id": cust["customer_id"],
                    "product_id": f"P{rng.integers(1, 400):04d}",
                    "category": category,
                    "transaction_date": date.strftime("%Y-%m-%d"),
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "discount": discount,
                    "payment_method": payment_method,
                    "region": region,
                    "revenue": revenue,
                }
            )
            txn_counter += 1

    df = pd.DataFrame(rows)

    # ---- Inject realistic data-quality issues ----

    # 1) Missing values
    for col, frac in [
        ("discount", 0.02),
        ("payment_method", 0.015),
        ("unit_price", 0.004),
        ("region", 0.01),
    ]:
        mask = rng.random(len(df)) < frac
        df.loc[mask, col] = np.nan

    # 2) Duplicate transactions (data pipeline re-ingestion issue)
    dup_sample = df.sample(frac=0.006, random_state=seed)
    df = pd.concat([df, dup_sample], ignore_index=True)

    # 3) Anomalies: implausibly large quantity/revenue (data entry or fraud-like)
    anomaly_mask = rng.random(len(df)) < 0.003
    df.loc[anomaly_mask, "quantity"] = df.loc[anomaly_mask, "quantity"] * rng.integers(20, 60, size=anomaly_mask.sum())
    df.loc[anomaly_mask, "revenue"] = (
        df.loc[anomaly_mask, "quantity"] * df.loc[anomaly_mask, "unit_price"].fillna(df["unit_price"].median())
        * (1 - df.loc[anomaly_mask, "discount"].fillna(0))
    ).round(2)

    # 4) A few negative prices (entry errors)
    err_mask = rng.random(len(df)) < 0.0015
    df.loc[err_mask, "unit_price"] = -df.loc[err_mask, "unit_price"].abs()

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df, customers


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df, customers = generate()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df):,} transactions for {customers['customer_id'].nunique():,} "
          f"candidate customers -> {OUTPUT_PATH}")
    print(f"Customers with >=1 transaction: {df['customer_id'].nunique():,}")
    print(df.head())
    print("\nMissing values:\n", df.isna().sum())


if __name__ == "__main__":
    main()
