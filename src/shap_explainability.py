import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

print("=" * 60)
print("SHAP EXPLAINABILITY")
print("=" * 60)

# ============================================================
# STEP 1 — LOAD SAVED MODEL
# ============================================================

model_path = "Data/Processed/balanced_random_forest.pkl"

model = joblib.load(model_path)

print("\nModel loaded successfully!")
print("Model: Balanced Random Forest")

# ============================================================
# STEP 2 — LOAD TEST DATA
# ============================================================

X_test_path = "Data/Processed/X_test.csv"

X_test = pd.read_csv(X_test_path)

print("\nTest data loaded successfully!")
print("Test data shape:")
print(X_test.shape)

# ============================================================
# STEP 3 — LOAD ENCODED FEATURE NAMES
# ============================================================

preprocessor = joblib.load(
    "Data/Processed/preprocessor.pkl"
)

encoded_feature_names = preprocessor.get_feature_names_out()

print("\nEncoded feature names loaded successfully!")

print("Number of feature names:")
print(len(encoded_feature_names))

# ============================================================
# STEP 4 — CREATE DATAFRAME WITH FEATURE NAMES
# ============================================================

X_test.columns = encoded_feature_names

print("\nFeature names assigned successfully!")

print("First 10 feature names:")
print(X_test.columns[:10].tolist())

# ============================================================
# STEP 5 — CREATE SHAP EXPLAINER
# ============================================================

print("\nCreating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(model)

print("SHAP explainer created successfully!")

# ============================================================
# STEP 6 — CALCULATE SHAP VALUES
# ============================================================

print("\nCalculating SHAP values...")

# Use a smaller sample for faster processing
X_shap = X_test.sample(
    n=min(500, len(X_test)),
    random_state=42
)

shap_values = explainer.shap_values(X_shap)

print("SHAP values calculated successfully!")

# ============================================================
# STEP 7 — HANDLE SHAP OUTPUT
# ============================================================

if isinstance(shap_values, list):

    # Older SHAP versions
    shap_values_churn = shap_values[1]

elif len(shap_values.shape) == 3:

    # Newer SHAP versions
    # Shape = (samples, features, classes)
    shap_values_churn = shap_values[:, :, 1]

else:

    shap_values_churn = shap_values

# ============================================================
# STEP 8 — GLOBAL FEATURE IMPORTANCE
# ============================================================

mean_shap_importance = np.abs(
    shap_values_churn
).mean(axis=0)

feature_importance = pd.DataFrame({
    "Feature": encoded_feature_names,
    "Mean_SHAP_Importance": mean_shap_importance
})

feature_importance = feature_importance.sort_values(
    by="Mean_SHAP_Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("TOP 15 SHAP FEATURES")
print("=" * 60)

print(
    feature_importance.head(15).to_string(index=False)
)

# ============================================================
# STEP 9 — SAVE SHAP FEATURE IMPORTANCE
# ============================================================

processed_path = "Data/Processed"

os.makedirs(
    processed_path,
    exist_ok=True
)

shap_importance_path = (
    f"{processed_path}/shap_feature_importance.csv"
)

feature_importance.to_csv(
    shap_importance_path,
    index=False
)

print("\nSHAP feature importance saved:")
print(shap_importance_path)

# ============================================================
# STEP 10 — SHAP SUMMARY PLOT
# ============================================================

print("\nCreating SHAP summary plot...")

plt.figure()

shap.summary_plot(
    shap_values_churn,
    X_shap,
    show=False
)

plt.tight_layout()

summary_plot_path = (
    f"{processed_path}/shap_summary.png"
)

plt.savefig(
    summary_plot_path,
    bbox_inches="tight"
)

plt.close()

print("SHAP summary plot saved:")
print(summary_plot_path)

# ============================================================
# STEP 11 — SHAP BAR PLOT
# ============================================================

print("\nCreating SHAP feature importance bar plot...")

plt.figure()

shap.summary_plot(
    shap_values_churn,
    X_shap,
    plot_type="bar",
    show=False
)

plt.tight_layout()

bar_plot_path = (
    f"{processed_path}/shap_feature_importance.png"
)

plt.savefig(
    bar_plot_path,
    bbox_inches="tight"
)

plt.close()

print("SHAP feature importance bar plot saved:")
print(bar_plot_path)

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("SHAP EXPLAINABILITY COMPLETED SUCCESSFULLY!")
print("=" * 60)