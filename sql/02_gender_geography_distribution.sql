-- ============================================
-- Bank Customer Churn — Gender & Geography Distribution
-- ============================================

-- Churn rate by Gender
SELECT
    Gender,
    COUNT(*) AS total_customers,
    SUM(Exited) AS churned_customers,
    ROUND(SUM(Exited) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM 
GROUP BY Gender
ORDER BY churn_rate_pct DESC;

-- Churn rate by Geography
SELECT
    Geography,
    COUNT(*) AS total_customers,
    SUM(Exited) AS churned_customers,
    ROUND(SUM(Exited) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM 
GROUP BY Geography
ORDER BY churn_rate_pct DESC;
