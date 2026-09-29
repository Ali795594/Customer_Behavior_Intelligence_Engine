-- =====================================================================
-- 01_schema.sql
-- Customer Behavior Intelligence Engine
-- =====================================================================

DROP TABLE IF EXISTS customer_transactions;

CREATE TABLE customer_transactions (
    transaction_id   VARCHAR(20)     PRIMARY KEY,
    customer_id      VARCHAR(20)     NOT NULL,
    product_id       VARCHAR(10)     NOT NULL,
    category         VARCHAR(50)     NOT NULL,
    txn_date         DATE            NOT NULL,
    quantity         NUMERIC(10, 2)  NOT NULL,
    unit_price       NUMERIC(10, 2),
    discount         NUMERIC(5, 4),
    payment_method   VARCHAR(30),
    region           VARCHAR(30),
    revenue          NUMERIC(12, 2)  NOT NULL
);

CREATE INDEX idx_cust_txn_customer_id  ON customer_transactions (customer_id);
CREATE INDEX idx_cust_txn_date         ON customer_transactions (txn_date);
CREATE INDEX idx_cust_txn_category     ON customer_transactions (category);
CREATE INDEX idx_cust_txn_region       ON customer_transactions (region);


-- Optional: table for the final customer-level analytics output
-- (populated from data/processed/final_customer_dataset.csv)
DROP TABLE IF EXISTS customer_analytics;

CREATE TABLE customer_analytics (
    customer_id            VARCHAR(20)     PRIMARY KEY,
    segment                VARCHAR(60),
    recency                INTEGER,
    frequency              INTEGER,
    monetary_value         NUMERIC(12, 2),
    aov                     NUMERIC(12, 2),
    purchase_probability   NUMERIC(6, 4)
);
