-- =====================================================================
-- 02_load_data.sql
-- Loads the CLEANED CSVs produced by the Python pipeline into Postgres.
--
-- IMPORTANT: run the pipeline (see README "How to Run") FIRST so that
-- data/processed/customer_transactions_sql.csv and
-- data/processed/final_customer_dataset.csv exist.
--
-- Run from the sql/ folder so the relative paths below resolve:
--   psql -U your_user -d your_database -f 01_schema.sql
--   psql -U your_user -d your_database -f 02_load_data.sql
-- =====================================================================

\copy customer_transactions (transaction_id, customer_id, product_id, category, txn_date, quantity, unit_price, discount, payment_method, region, revenue) FROM '../data/processed/customer_transactions_sql.csv' WITH (FORMAT csv, HEADER true, NULL '');

\copy customer_analytics (customer_id, segment, recency, frequency, monetary_value, aov, purchase_probability) FROM '../data/processed/final_customer_dataset.csv' WITH (FORMAT csv, HEADER true, NULL '');

SELECT COUNT(*) AS transaction_row_count FROM customer_transactions;
SELECT COUNT(*) AS customer_row_count FROM customer_analytics;
