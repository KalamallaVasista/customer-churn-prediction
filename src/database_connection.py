from sqlalchemy import create_engine, text
from urllib.parse import quote_plus


# ==========================================
# POSTGRESQL DATABASE CONFIGURATION
# ==========================================

DB_USER = "postgres"
DB_PASSWORD = "Reddy@9441"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "customer_churn_db"


# ==========================================
# ENCODE PASSWORD
# ==========================================

encoded_password = quote_plus(DB_PASSWORD)


# ==========================================
# CREATE DATABASE CONNECTION
# ==========================================

DATABASE_URL = (
    f"postgresql+psycopg2://"
    f"{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DATABASE_URL)


# ==========================================
# TEST DATABASE CONNECTION
# ==========================================

print("=" * 60)
print("POSTGRESQL + SQLALCHEMY CONNECTION TEST")
print("=" * 60)

try:

    with engine.connect() as connection:

        # --------------------------------------
        # Test connection
        # --------------------------------------

        result = connection.execute(
            text("SELECT 1")
        )

        print("\nDatabase connection successful!")

        print("\nTest query result:")
        print(result.scalar())


        # --------------------------------------
        # Count customers
        # --------------------------------------

        customer_count = connection.execute(
            text("SELECT COUNT(*) FROM customers")
        ).scalar()

        print("\nTotal customers in PostgreSQL:")
        print(customer_count)


        # --------------------------------------
        # Get churn count
        # --------------------------------------

        churn_count = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM customers
                WHERE churn = 'Yes'
            """)
        ).scalar()

        print("\nTotal churned customers:")
        print(churn_count)


        # --------------------------------------
        # Get non-churn count
        # --------------------------------------

        non_churn_count = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM customers
                WHERE churn = 'No'
            """)
        ).scalar()

        print("\nTotal non-churned customers:")
        print(non_churn_count)


except Exception as e:

    print("\nDatabase connection failed!")

    print("\nError:")
    print(e)