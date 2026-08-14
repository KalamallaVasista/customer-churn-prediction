-- ============================================================
-- CUSTOMER CHURN ANALYSIS
-- PostgreSQL
-- Database: customer_churn_db
-- Table: customers
-- ============================================================


-- ============================================================
-- 1. OVERALL CUSTOMER CHURN
-- ============================================================

SELECT
    COUNT(*) AS total_customers,

    SUM(
        CASE
            WHEN churn = 'Yes' THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    SUM(
        CASE
            WHEN churn = 'No' THEN 1
            ELSE 0
        END
    ) AS non_churned_customers,

    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN churn = 'Yes' THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS churn_rate

FROM customers;


-- ============================================================
-- 2. CHURN BY CONTRACT
-- ============================================================

SELECT
    contract,

    COUNT(*) AS total_customers,

    SUM(
        CASE
            WHEN churn = 'Yes' THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN churn = 'Yes' THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS churn_rate

FROM customers

GROUP BY contract

ORDER BY churn_rate DESC;


-- ============================================================
-- 3. CHURN BY PAYMENT METHOD
-- ============================================================

SELECT
    paymentmethod,

    COUNT(*) AS total_customers,

    SUM(
        CASE
            WHEN churn = 'Yes' THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN churn = 'Yes' THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS churn_rate

FROM customers

GROUP BY paymentmethod

ORDER BY churn_rate DESC;


-- ============================================================
-- 4. CHURN BY INTERNET SERVICE
-- ============================================================

SELECT
    internetservice,

    COUNT(*) AS total_customers,

    SUM(
        CASE
            WHEN churn = 'Yes' THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN churn = 'Yes' THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS churn_rate

FROM customers

GROUP BY internetservice

ORDER BY churn_rate DESC;


-- ============================================================
-- 5. CHURN BY TENURE GROUP
-- ============================================================

SELECT
    CASE
        WHEN tenure <= 12 THEN '0-12 months'
        WHEN tenure <= 24 THEN '13-24 months'
        WHEN tenure <= 48 THEN '25-48 months'
        ELSE '49-72 months'
    END AS tenure_group,

    COUNT(*) AS total_customers,

    SUM(
        CASE
            WHEN churn = 'Yes' THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN churn = 'Yes' THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS churn_rate

FROM customers

GROUP BY
    CASE
        WHEN tenure <= 12 THEN '0-12 months'
        WHEN tenure <= 24 THEN '13-24 months'
        WHEN tenure <= 48 THEN '25-48 months'
        ELSE '49-72 months'
    END

ORDER BY churn_rate DESC;


-- ============================================================
-- 6. CHURN BY TECH SUPPORT
-- ============================================================

SELECT
    techsupport,

    COUNT(*) AS total_customers,

    SUM(
        CASE
            WHEN churn = 'Yes' THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN churn = 'Yes' THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS churn_rate

FROM customers

GROUP BY techsupport

ORDER BY churn_rate DESC;


-- ============================================================
-- 7. HIGH-RISK CUSTOMER SEGMENTS
-- ============================================================

SELECT
    contract,
    paymentmethod,
    internetservice,

    COUNT(*) AS total_customers,

    SUM(
        CASE
            WHEN churn = 'Yes' THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN churn = 'Yes' THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS churn_rate

FROM customers

GROUP BY
    contract,
    paymentmethod,
    internetservice

HAVING COUNT(*) >= 50

ORDER BY churn_rate DESC;


-- ============================================================
-- 8. HIGH MONTHLY CHARGE CUSTOMERS
-- ============================================================

SELECT
    customerid,
    tenure,
    monthlycharges,
    contract,
    internetservice,
    churn

FROM customers

WHERE monthlycharges >= 100

ORDER BY monthlycharges DESC;


-- ============================================================
-- 9. CUSTOMERS WITH SHORT TENURE AND CHURN
-- ============================================================

SELECT
    customerid,
    tenure,
    monthlycharges,
    contract,
    paymentmethod,
    churn

FROM customers

WHERE tenure <= 12
  AND churn = 'Yes'

ORDER BY tenure ASC;


-- ============================================================
-- 10. CHURNED CUSTOMER REVENUE
-- ============================================================

SELECT
    COUNT(*) AS churned_customers,

    ROUND(
        SUM(monthlycharges),
        2
    ) AS total_monthly_revenue_lost,

    ROUND(
        AVG(monthlycharges),
        2
    ) AS average_monthly_charge

FROM customers

WHERE churn = 'Yes';


-- ============================================================
-- END OF CUSTOMER CHURN ANALYSIS
-- ============================================================