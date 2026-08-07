-- ============================================
-- Bank Customer Churn — Data Quality Check
-- Dataset: 10,000 bank customer records
-- ============================================

-- Check for NULL values across all key columns
SELECT
    SUM(CASE WHEN CustomerId IS NULL THEN 1 ELSE 0 END) AS null_customer_id,
    SUM(CASE WHEN CreditScore IS NULL THEN 1 ELSE 0 END) AS null_credit_score,
    SUM(CASE WHEN Geography IS NULL THEN 1 ELSE 0 END) AS null_geography,
    SUM(CASE WHEN Gender IS NULL THEN 1 ELSE 0 END) AS null_gender,
    SUM(CASE WHEN Age IS NULL THEN 1 ELSE 0 END) AS null_age,
    SUM(CASE WHEN Tenure IS NULL THEN 1 ELSE 0 END) AS null_tenure,
    SUM(CASE WHEN Balance IS NULL THEN 1 ELSE 0 END) AS null_balance,
    SUM(CASE WHEN NumOfProducts IS NULL THEN 1 ELSE 0 END) AS null_num_products,
    SUM(CASE WHEN HasCrCard IS NULL THEN 1 ELSE 0 END) AS null_has_cr_card,
    SUM(CASE WHEN IsActiveMember IS NULL THEN 1 ELSE 0 END) AS null_is_active,
    SUM(CASE WHEN EstimatedSalary IS NULL THEN 1 ELSE 0 END) AS null_salary,
    SUM(CASE WHEN Exited IS NULL THEN 1 ELSE 0 END) AS null_exited,
    COUNT(*) AS total_rows
FROM bank_customers;

-- Result: 0 nulls across all columns — dataset confirmed clean
