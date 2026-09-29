"""
generate_dashboard.py
-----------------------
Builds a SINGLE self-contained HTML file with four tabbed dashboards
(Customer Overview, Segmentation, Purchase Propensity, Behavior) - just a web browser needed.

Run:
    python src/generate_dashboard.py

Output:
    dashboard/customer_dashboard.html
"""

import os
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def kpi_card(label, value):
    return f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>"""


def img_block(title, b64):
    return f"""
    <div class="chart-block">
        <h3>{title}</h3>
        <img src="data:image/png;base64,{b64}" />
    </div>"""


# ---------------------------------------------------------------- charts --

def chart_revenue_by_category(transactions):
    rev = transactions.groupby("category")["revenue"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.barplot(x=rev.values, y=rev.index, ax=ax, color="#4C72B0")
    ax.set_xlabel("Revenue")
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_monthly_active_customers(transactions):
    t = transactions.copy()
    t["transaction_date"] = pd.to_datetime(t["transaction_date"])
    monthly = t.set_index("transaction_date").resample("ME")["customer_id"].nunique()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    monthly.plot(ax=ax, marker="o", color="#55A868")
    ax.set_ylabel("Active Customers")
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_segment_scatter(final_df):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.scatterplot(data=final_df, x="recency", y="monetary_value", hue="segment", alpha=0.5, s=18, ax=ax)
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_segment_counts(final_df):
    counts = final_df["segment"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#8172B2")
    ax.set_xlabel("Number of Customers")
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_propensity_distribution(final_df):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.histplot(final_df["purchase_probability"], bins=30, ax=ax, color="#C44E52")
    ax.set_xlabel("Purchase Probability (next 30 days)")
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_propensity_by_segment(final_df):
    avg = final_df.groupby("segment")["purchase_probability"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.barplot(x=avg.values, y=avg.index, ax=ax, color="#DD8452")
    ax.set_xlabel("Avg. Purchase Probability")
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_preferred_category(features):
    counts = features["preferred_category"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#937860")
    ax.set_xlabel("Number of Customers")
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_pareto(features):
    sorted_rev = features["monetary_value"].sort_values(ascending=False).reset_index(drop=True)
    cum_rev_pct = sorted_rev.cumsum() / sorted_rev.sum() * 100
    cum_cust_pct = (sorted_rev.index + 1) / len(sorted_rev) * 100
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(cum_cust_pct, cum_rev_pct, color="#4C72B0")
    ax.axhline(80, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Cumulative % of Customers")
    ax.set_ylabel("Cumulative % of Revenue")
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_discount_by_segment(transactions, final_df):
    merged = transactions.merge(final_df[["customer_id", "segment"]], on="customer_id", how="left")
    avg = merged.groupby("segment")["discount"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.barplot(x=avg.values, y=avg.index, ax=ax, color="#55A868")
    ax.set_xlabel("Average Discount")
    fig.tight_layout()
    return fig_to_base64(fig)


# ------------------------------------------------------------------ HTML --

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Customer Behavior Intelligence Dashboard</title>
<style>
  :root {{ --accent:#4C72B0; --bg:#f4f6f9; --card:#ffffff; --text:#222; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; background:var(--bg); margin:0; color:var(--text); }}
  header {{ background:linear-gradient(135deg,#2c3e63,#4C72B0); color:white; padding:28px 32px; }}
  header h1 {{ margin:0; font-size:1.6rem; }}
  header p {{ margin:6px 0 0; opacity:0.85; font-size:0.95rem; }}
  .tabs {{ display:flex; gap:4px; padding:0 24px; background:#e7eaf0; border-bottom:1px solid #d3d8e2; flex-wrap: wrap;}}
  .tab-btn {{ padding:12px 20px; cursor:pointer; border:none; background:transparent; font-size:0.95rem; font-weight:600; color:#555; border-bottom:3px solid transparent; }}
  .tab-btn.active {{ color:var(--accent); border-bottom:3px solid var(--accent); }}
  .tab-content {{ display:none; padding:24px 32px 48px; max-width:1400px; margin:0 auto; }}
  .tab-content.active {{ display:block; }}
  .kpi-row {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:24px; }}
  .kpi-card {{ background:var(--card); border-radius:10px; padding:18px 24px; box-shadow:0 1px 4px rgba(0,0,0,0.08); min-width:180px; flex:1; }}
  .kpi-value {{ font-size:1.6rem; font-weight:700; color:var(--accent); }}
  .kpi-label {{ font-size:0.85rem; color:#777; margin-top:4px; }}
  .chart-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
  .chart-block {{ background:var(--card); border-radius:10px; padding:16px 20px; box-shadow:0 1px 4px rgba(0,0,0,0.08); }}
  .chart-block.full {{ grid-column:1 / -1; }}
  .chart-block h3 {{ margin:0 0 10px; font-size:1rem; color:#444; }}
  .chart-block img {{ width:100%; height:auto; border-radius:6px; }}
  table.data-table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
  table.data-table th, table.data-table td {{ padding:6px 10px; border-bottom:1px solid #eee; text-align:right; }}
  table.data-table th:first-child, table.data-table td:first-child {{ text-align:left; }}
  footer {{ text-align:center; padding:20px; color:#999; font-size:0.8rem; }}
  @media (max-width:900px) {{ .chart-grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>Customer Behavior Intelligence Dashboard</h1>
  <p>Generated locally from the cleaned transactions, customer features and final customer dataset. Open this file directly in any browser.</p>
</header>
<div class="tabs">
  <button class="tab-btn active" onclick="showTab(0)">Customer Overview</button>
  <button class="tab-btn" onclick="showTab(1)">Customer Segmentation</button>
  <button class="tab-btn" onclick="showTab(2)">Purchase Propensity</button>
  <button class="tab-btn" onclick="showTab(3)">Customer Behavior</button>
</div>

<div class="tab-content active" id="tab-0">
  <div class="kpi-row">{kpi_row_1}</div>
  <div class="chart-grid">
    {chart_rev_cat}
    {chart_active_customers}
  </div>
</div>

<div class="tab-content" id="tab-1">
  <div class="kpi-row">{kpi_row_2}</div>
  <div class="chart-grid">
    {chart_seg_scatter}
    {chart_seg_counts}
    <div class="chart-block full">{segment_table}</div>
  </div>
</div>

<div class="tab-content" id="tab-2">
  <div class="chart-grid">
    {chart_propensity_dist}
    {chart_propensity_segment}
    <div class="chart-block full">{high_propensity_table}</div>
  </div>
</div>

<div class="tab-content" id="tab-3">
  <div class="chart-grid">
    {chart_preferred_cat}
    {chart_pareto}
    <div class="chart-block full">{chart_discount_segment}</div>
  </div>
</div>

<footer>Customer Behavior Intelligence Engine - dashboard generated by src/generate_dashboard.py</footer>

<script>
function showTab(i) {{
  document.querySelectorAll('.tab-content').forEach((el, idx) => el.classList.toggle('active', idx === i));
  document.querySelectorAll('.tab-btn').forEach((el, idx) => el.classList.toggle('active', idx === i));
}}
</script>
</body>
</html>
"""


def df_to_html_table(df, float_cols=None):
    float_cols = float_cols or []
    df = df.copy()
    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].map(lambda x: f"{x:,.2f}")
    return df.to_html(classes="data-table", index=False, border=0)


def build_dashboard(transactions, features, final_df, out_path):
    total_customers = final_df["customer_id"].nunique()
    total_revenue = transactions["revenue"].sum()
    aov = transactions["revenue"].sum() / transactions["transaction_id"].nunique()
    repeat_rate = (features["frequency"] > 1).mean()

    kpi_row_1 = (
        kpi_card("Total Customers", f"{total_customers:,}")
        + kpi_card("Total Revenue", f"${total_revenue:,.0f}")
        + kpi_card("Avg. Order Value", f"${aov:,.2f}")
        + kpi_card("Repeat Purchase Rate", f"{repeat_rate*100:.1f}%")
    )

    seg_profile = (
        final_df.groupby("segment")
        .agg(
            n_customers=("customer_id", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary_value", "mean"),
            avg_purchase_probability=("purchase_probability", "mean"),
        )
        .reset_index()
        .sort_values("avg_monetary", ascending=False)
    )
    kpi_row_2 = "".join(
        kpi_card(row["segment"], f"{row['n_customers']:,} customers")
        for _, row in seg_profile.iterrows()
    )

    high_propensity = (
        final_df[final_df["purchase_probability"] >= 0.7]
        .sort_values("monetary_value", ascending=False)
        .head(15)[["customer_id", "segment", "monetary_value", "AOV", "purchase_probability"]]
    )

    html = PAGE_TEMPLATE.format(
        kpi_row_1=kpi_row_1,
        kpi_row_2=kpi_row_2,
        chart_rev_cat=img_block("Revenue by Category", chart_revenue_by_category(transactions)),
        chart_active_customers=img_block("Monthly Active Customers", chart_monthly_active_customers(transactions)),
        chart_seg_scatter=img_block("Segments: Recency vs Monetary Value (log scale)", chart_segment_scatter(final_df)),
        chart_seg_counts=img_block("Customers per Segment", chart_segment_counts(final_df)),
        segment_table=f"<h3>Segment Profile</h3>{df_to_html_table(seg_profile, ['avg_recency','avg_frequency','avg_monetary','avg_purchase_probability'])}",
        chart_propensity_dist=img_block("Purchase Probability Distribution", chart_propensity_distribution(final_df)),
        chart_propensity_segment=img_block("Avg. Purchase Probability by Segment", chart_propensity_by_segment(final_df)),
        high_propensity_table=f"<h3>Top High-Propensity, High-Value Customers</h3>{df_to_html_table(high_propensity, ['monetary_value','AOV','purchase_probability'])}",
        chart_preferred_cat=img_block("Customers by Preferred Category", chart_preferred_category(features)),
        chart_pareto=img_block("Customer Revenue Concentration (Pareto)", chart_pareto(features)),
        chart_discount_segment=img_block("Average Discount by Segment", chart_discount_by_segment(transactions, final_df)),
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Saved self-contained HTML dashboard -> {out_path}")
    print("Open this file directly in any web browser.")


if __name__ == "__main__":
    from utils import CLEAN_DATA_PATH, CUSTOMER_FEATURES_PATH, FINAL_CUSTOMER_PATH, BASE_DIR

    for p in [CLEAN_DATA_PATH, CUSTOMER_FEATURES_PATH, FINAL_CUSTOMER_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {p}. Run the full pipeline first (see README).")

    transactions = pd.read_csv(CLEAN_DATA_PATH)
    features = pd.read_csv(CUSTOMER_FEATURES_PATH)
    final_df = pd.read_csv(FINAL_CUSTOMER_PATH)

    out_path = os.path.join(BASE_DIR, "dashboard", "customer_dashboard.html")
    build_dashboard(transactions, features, final_df, out_path)
