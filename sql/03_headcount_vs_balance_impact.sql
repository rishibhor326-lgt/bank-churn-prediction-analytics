-- ============================================
-- Bank Customer Churn — Headcount vs. Balance Impact
-- Key finding: churned customers are a disproportionate
-- share of total balance relative to their headcount share
-- ============================================

SELECT
    Exited,
    COUNT(*) AS headcount,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM bank_churn.churn_modelling), 2) AS headcount_pct,
    ROUND(SUM(Balance), 2) AS total_balance,
    ROUND(SUM(Balance) * 100.0 / (SELECT SUM(Balance) FROM bank_churn.churn_modelling), 2) AS balance_pct
FROM bank_churn.churn_modelling
GROUP BY Exited;

-- Result:
-- Churned customers (Exited = 1) make up ~20% of headcount
-- but hold ~24% of total balance — a disproportionate revenue risk
