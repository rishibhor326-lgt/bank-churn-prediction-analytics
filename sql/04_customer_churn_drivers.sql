-- =====================================================
-- Bank Customer Churn — Customer Churn Driver Analysis
-- =====================================================


-- 1. Churn by age group
SELECT
    CASE
        WHEN Age < 30 THEN 'Under 30'
        WHEN Age BETWEEN 30 AND 39 THEN '30-39'
        WHEN Age BETWEEN 40 AND 49 THEN '40-49'
        WHEN Age BETWEEN 50 AND 59 THEN '50-59'
        ELSE '60+'
    END AS age_group,
    COUNT(*) AS total_customers,
    SUM(Exited) AS churned_customers,
    ROUND(AVG(Exited) * 100, 2) AS churn_rate_percentage
FROM bank_churn.churn_modelling
GROUP BY age_group
ORDER BY MIN(Age);


-- 2. Average balance by age group
SELECT
    CASE
        WHEN Age < 30 THEN 'Under 30'
        WHEN Age BETWEEN 30 AND 39 THEN '30-39'
        WHEN Age BETWEEN 40 AND 49 THEN '40-49'
        WHEN Age BETWEEN 50 AND 59 THEN '50-59'
        ELSE '60+'
    END AS age_group,
    ROUND(AVG(Balance), 2) AS average_balance,
    ROUND(
        AVG(CASE WHEN Exited = 1 THEN Balance END),
        2
    ) AS average_balance_of_churned_customers
FROM bank_churn.churn_modelling
GROUP BY age_group
ORDER BY MIN(Age);


-- 3. Total balance associated with churn by age group
SELECT
    CASE
        WHEN Age < 30 THEN 'Under 30'
        WHEN Age BETWEEN 30 AND 39 THEN '30-39'
        WHEN Age BETWEEN 40 AND 49 THEN '40-49'
        WHEN Age BETWEEN 50 AND 59 THEN '50-59'
        ELSE '60+'
    END AS age_group,
    ROUND(SUM(Balance), 2) AS total_balance,
    ROUND(
        SUM(CASE WHEN Exited = 1 THEN Balance ELSE 0 END),
        2
    ) AS churned_customer_balance,
    ROUND(
        SUM(CASE WHEN Exited = 1 THEN Balance ELSE 0 END)
        / NULLIF(SUM(Balance), 0) * 100,
        2
    ) AS balance_associated_with_churn_percentage
FROM bank_churn.churn_modelling
GROUP BY age_group
ORDER BY MIN(Age);


-- 4. Churn by activity status
SELECT
    CASE
        WHEN IsActiveMember = 1 THEN 'Active'
        ELSE 'Inactive'
    END AS activity_status,
    COUNT(*) AS total_customers,
    SUM(Exited) AS churned_customers,
    ROUND(AVG(Exited) * 100, 2) AS churn_rate_percentage
FROM bank_churn.churn_modelling
GROUP BY IsActiveMember
ORDER BY churn_rate_percentage DESC;


-- 5. Churn by number of products
SELECT
    NumOfProducts AS number_of_products,
    COUNT(*) AS total_customers,
    SUM(Exited) AS churned_customers,
    ROUND(AVG(Exited) * 100, 2) AS churn_rate_percentage
FROM bank_churn.churn_modelling
GROUP BY NumOfProducts
ORDER BY NumOfProducts;