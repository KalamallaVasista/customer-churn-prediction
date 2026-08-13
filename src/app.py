import streamlit as st
import joblib
import pandas as pd

# ==========================================
# CUSTOMER CHURN PREDICTION APP
# ==========================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# ==========================================
# APP HEADER
# ==========================================

st.title("📊 Customer Churn Prediction")

st.markdown(
    """
    **Predict customer churn using Machine Learning**

    Enter customer information below to estimate:

    - Churn probability
    - Customer risk level
    - Recommended business action
    """
)

# ==========================================
# LOAD MODEL
# ==========================================

model_path = "Data/Processed/balanced_random_forest.pkl"

model = joblib.load(model_path)

# ==========================================
# LOAD PREPROCESSOR AND SCALER
# ==========================================

preprocessor = joblib.load(
    "Data/Processed/preprocessor.pkl"
)

scaler = joblib.load(
    "Data/Processed/scaler.pkl"
)

st.success(
    "Model and preprocessing loaded successfully!"
)

# ==========================================
# MODEL INFORMATION
# ==========================================

st.subheader("Model Information")

st.write(
    "Model: Balanced Random Forest"
)

st.write(
    "Purpose: Customer Churn Prediction"
)

# ==========================================
# STEP 78 — MODEL PERFORMANCE
# ==========================================

st.subheader("Model Performance")

model_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Balanced Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Balanced Random Forest"
    ],
    "Accuracy": [
        0.8055,
        0.7381,
        0.7984,
        0.8062,
        0.7559
    ],
    "Precision": [
        0.6572,
        0.5043,
        0.6347,
        0.6797,
        0.5265
    ],
    "Recall": [
        0.5588,
        0.7834,
        0.5668,
        0.5107,
        0.7968
    ],
    "F1 Score": [
        0.6040,
        0.6136,
        0.5989,
        0.5832,
        0.6340
    ]
})

# Convert values to percentages

performance_display = model_results.copy()

performance_display[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
] = (
    performance_display[
        ["Accuracy", "Precision", "Recall", "F1 Score"]
    ] * 100
).round(2)

# Add percentage symbol

for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]:
    performance_display[column] = (
        performance_display[column].astype(str) + "%"
    )

st.dataframe(
    performance_display,
    use_container_width=True,
    hide_index=True
)

st.info(
    "Balanced Random Forest was selected because it provides "
    "the highest recall and F1 score among the evaluated models. "
    "This is useful for identifying customers who are more likely "
    "to churn."
)

# ==========================================
# CUSTOMER INPUT FORM
# ==========================================

st.subheader("Customer Information")

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

senior_citizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.selectbox(
    "Partner",
    ["No", "Yes"]
)

dependents = st.selectbox(
    "Dependents",
    ["No", "Yes"]
)

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=12
)

phone_service = st.selectbox(
    "Phone Service",
    ["No", "Yes"]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["No", "No phone service", "Yes"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["No", "No internet service", "Yes"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["No", "No internet service", "Yes"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["No", "No internet service", "Yes"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["No", "No internet service", "Yes"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["No", "No internet service", "Yes"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["No", "No internet service", "Yes"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=18.25,
    max_value=118.75,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)

# ==========================================
# CREATE CUSTOMER DATAFRAME
# ==========================================

customer_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

st.subheader("Customer Data Preview")

st.dataframe(
    customer_data,
    use_container_width=True
)

# ==========================================
# TRANSFORM CUSTOMER DATA
# ==========================================

encoded_customer = preprocessor.transform(
    customer_data
)

scaled_customer = encoded_customer.copy()

scaled_customer[:, -4:] = scaler.transform(
    encoded_customer[:, -4:]
)

# ==========================================
# PREDICT CHURN
# ==========================================

if st.button("🔮 Predict Churn"):

    prediction = model.predict(
        scaled_customer
    )

    probability = model.predict_proba(
        scaled_customer
    )

    churn_probability = probability[0][1]

    no_churn_probability = probability[0][0]

    # ==========================================
    # PREDICTION RESULT
    # ==========================================

    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.error(
            "⚠️ CUSTOMER WILL CHURN"
        )

    else:

        st.success(
            "✅ CUSTOMER WILL NOT CHURN"
        )

    st.write(
        f"**Churn Probability:** "
        f"{churn_probability * 100:.2f}%"
    )

    st.write(
        f"**No Churn Probability:** "
        f"{no_churn_probability * 100:.2f}%"
    )

    # ==========================================
    # RISK LEVEL
    # ==========================================

    if churn_probability < 0.30:

        risk_level = "LOW RISK"

    elif churn_probability < 0.60:

        risk_level = "MEDIUM RISK"

    else:

        risk_level = "HIGH RISK"

    st.subheader(
        "Customer Risk Level"
    )

    if risk_level == "LOW RISK":

        st.success(
            "🟢 LOW RISK"
        )

    elif risk_level == "MEDIUM RISK":

        st.warning(
            "🟡 MEDIUM RISK"
        )

    else:

        st.error(
            "🔴 HIGH RISK"
        )

    # ==========================================
    # BUSINESS RECOMMENDATION
    # ==========================================

    st.subheader(
        "Recommended Action"
    )

    if risk_level == "LOW RISK":

        st.info(
            "Customer has a low churn risk. "
            "Continue regular engagement and maintain "
            "good service."
        )

    elif risk_level == "MEDIUM RISK":

        st.warning(
            "Customer has a medium churn risk. "
            "Monitor the customer and consider offering "
            "a suitable discount or service upgrade."
        )

    else:

        st.error(
            "Customer has a high churn risk. "
            "Contact the customer proactively and consider "
            "a retention offer, contract incentive, or "
            "service improvement."
        )

    # ==========================================
    # CHURN PROBABILITY VISUALIZATION
    # ==========================================

    st.subheader(
        "Churn Probability Analysis"
    )

    st.write(
        f"**Churn Probability:** "
        f"{churn_probability * 100:.2f}%"
    )

    st.progress(
        churn_probability
    )

    st.write(
        f"**No Churn Probability:** "
        f"{no_churn_probability * 100:.2f}%"
    )

    st.progress(
        no_churn_probability
    )