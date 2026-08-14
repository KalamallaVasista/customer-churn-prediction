from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import pandas as pd


# ==========================================
# POSTGRESQL CONFIGURATION
# ==========================================

DB_USER = "postgres"
DB_PASSWORD = "Reddy@9441"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "customer_churn_db"


# ==========================================
# CREATE DATABASE CONNECTION
# ==========================================

encoded_password = quote_plus(DB_PASSWORD)

DATABASE_URL = (
    f"postgresql+psycopg2://"
    f"{DB_USER}:{encoded_password}@"
    f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DATABASE_URL)


print("=" * 60)
print("POSTGRESQL CUSTOMER CHURN ANALYSIS")
print("=" * 60)


try:

    with engine.connect() as connection:

        # ==========================================
        # 1. OVERALL CHURN
        # ==========================================

        query = text("""
            SELECT
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
            FROM customers;
        """)

        result = connection.execute(query).fetchone()

        print("\n" + "=" * 60)
        print("OVERALL CHURN")
        print("=" * 60)

        print("\nTotal customers:")
        print(result.total_customers)

        print("\nChurned customers:")
        print(result.churned_customers)

        print("\nOverall churn rate:")
        print(result.churn_rate)


        # ==========================================
        # 2. CHURN BY CONTRACT
        # ==========================================

        query = text("""
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
        """)

        contract_df = pd.read_sql(
            query,
            connection
        )

        print("\n" + "=" * 60)
        print("CHURN BY CONTRACT")
        print("=" * 60)

        print(contract_df.to_string(index=False))


        # ==========================================
        # 3. CHURN BY PAYMENT METHOD
        # ==========================================

        query = text("""
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
        """)

        payment_df = pd.read_sql(
            query,
            connection
        )

        print("\n" + "=" * 60)
        print("CHURN BY PAYMENT METHOD")
        print("=" * 60)

        print(payment_df.to_string(index=False))


        # ==========================================
        # 4. CHURN BY INTERNET SERVICE
        # ==========================================

        query = text("""
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
        """)

        internet_df = pd.read_sql(
            query,
            connection
        )

        print("\n" + "=" * 60)
        print("CHURN BY INTERNET SERVICE")
        print("=" * 60)

        print(internet_df.to_string(index=False))


        # ==========================================
        # 5. CHURN BY TENURE GROUP
        # ==========================================

        query = text("""
            SELECT
                CASE
                    WHEN tenure <= 12
                        THEN '0-12 months'
                    WHEN tenure <= 24
                        THEN '13-24 months'
                    WHEN tenure <= 48
                        THEN '25-48 months'
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
                    WHEN tenure <= 12
                        THEN '0-12 months'
                    WHEN tenure <= 24
                        THEN '13-24 months'
                    WHEN tenure <= 48
                        THEN '25-48 months'
                    ELSE '49-72 months'
                END

            ORDER BY churn_rate DESC;
        """)

        tenure_df = pd.read_sql(
            query,
            connection
        )

        print("\n" + "=" * 60)
        print("CHURN BY TENURE GROUP")
        print("=" * 60)

        print(tenure_df.to_string(index=False))


        # ==========================================
        # 6. CHURN BY TECH SUPPORT
        # ==========================================

        query = text("""
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
        """)

        support_df = pd.read_sql(
            query,
            connection
        )

        print("\n" + "=" * 60)
        print("CHURN BY TECH SUPPORT")
        print("=" * 60)

        print(support_df.to_string(index=False))


except Exception as e:

    print("\nPostgreSQL analysis failed!")

    print("\nError:")
    print(e)