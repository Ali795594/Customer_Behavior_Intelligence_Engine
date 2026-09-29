-- =====================================================================
-- 03_analysis_queries.sql
-- Analytical queries for the Customer Behavior Intelligence Engine.
-- =====================================================================

-- 1. Top customers by lifetime revenue
SELECT
    customer_id,
    ROUND(SUM(revenue), 2) AS lifetime_revenue,
    COUNT(*)               AS n_transactions
FROM customer_transactions
GROUP BY customer_id
ORDER BY lifetime_revenue DESC
LIMIT 20;

-- 2. Revenue by category
SELECT
    category,
    ROUND(SUM(revenue), 2) AS total_revenue,
    COUNT(*)               AS n_transactions,
    COUNT(DISTINCT customer_id) AS n_customers
FROM customer_transactions
GROUP BY category
ORDER BY total_revenue DESC;

-- 3. Revenue by region
SELECT
    region,
    ROUND(SUM(revenue), 2) AS total_revenue,
    COUNT(DISTINCT customer_id) AS n_customers
FROM customer_transactions
GROUP BY region
ORDER BY total_revenue DESC;

-- 4. Monthly active customers and revenue
SELECT
    DATE_TRUNC('month', txn_date)::DATE AS month,
    COUNT(DISTINCT customer_id)         AS active_customers,
    ROUND(SUM(revenue), 2)              AS total_revenue
FROM customer_transactions
GROUP BY 1
ORDER BY 1;

-- 5. Payment method usage
SELECT
    payment_method,
    COUNT(*)                AS n_transactions,
    ROUND(SUM(revenue), 2)  AS total_revenue
FROM customer_transactions
GROUP BY payment_method
ORDER BY n_transactions DESC;

-- 6. Customer purchase frequency distribution (transactions per customer)
SELECT
    n_transactions,
    COUNT(*) AS n_customers
FROM (
    SELECT customer_id, COUNT(*) AS n_transactions
    FROM customer_transactions
    GROUP BY customer_id
) t
GROUP BY n_transactions
ORDER BY n_transactions;

-- 7. Segment performance (requires customer_analytics to be loaded)
SELECT
    segment,
    COUNT(*)                       AS n_customers,
    ROUND(AVG(monetary_value), 2)  AS avg_monetary_value,
    ROUND(AVG(frequency), 2)       AS avg_frequency,
    ROUND(AVG(recency), 1)         AS avg_recency,
    ROUND(AVG(purchase_probability), 3) AS avg_purchase_probability
FROM customer_analytics
GROUP BY segment
ORDER BY avg_monetary_value DESC;

-- 8. High-propensity, high-value customers worth prioritizing for outreach
SELECT
    customer_id, segment, monetary_value, purchase_probability
FROM customer_analytics
WHERE purchase_probability >= 0.7
ORDER BY monetary_value DESC
LIMIT 50;

-- 9. Category diversity: customers who buy from many different categories
SELECT
    customer_id,
    COUNT(DISTINCT category) AS categories_purchased,
    ROUND(SUM(revenue), 2)   AS total_revenue
FROM customer_transactions
GROUP BY customer_id
HAVING COUNT(DISTINCT category) >= 4
ORDER BY total_revenue DESC
LIMIT 20;

-- 10. Average discount received by segment (requires customer_analytics + a join)
SELECT
    ca.segment,
    ROUND(AVG(ct.discount), 4) AS avg_discount,
    ROUND(SUM(ct.revenue), 2)  AS total_revenue
FROM customer_transactions ct
JOIN customer_analytics ca ON ct.customer_id = ca.customer_id
GROUP BY ca.segment
ORDER BY total_revenue DESC;
