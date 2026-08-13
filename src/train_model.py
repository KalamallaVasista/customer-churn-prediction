import pandas as pd


# ==========================================
# STEP 36 — LOAD PROCESSED DATA
# ==========================================

processed_path = "Data/Processed"

X_train = pd.read_csv(
    f"{processed_path}/X_train.csv"
)

X_test = pd.read_csv(
    f"{processed_path}/X_test.csv"
)

y_train = pd.read_csv(
    f"{processed_path}/y_train.csv"
).squeeze()

y_test = pd.read_csv(
    f"{processed_path}/y_test.csv"
).squeeze()


print("=" * 60)
print("MACHINE LEARNING MODEL TRAINING")
print("=" * 60)

print("\nTraining features shape:")
print(X_train.shape)

print("\nTesting features shape:")
print(X_test.shape)

print("\nTraining target shape:")
print(y_train.shape)

print("\nTesting target shape:")
print(y_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

# ==========================================
# STEP 37 — TRAIN LOGISTIC REGRESSION
# ==========================================

from sklearn.linear_model import LogisticRegression


# Create the model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# Train the model
model.fit(X_train, y_train)


print("\n" + "=" * 60)
print("LOGISTIC REGRESSION TRAINING")
print("=" * 60)

print("\nModel training completed successfully!")

# ==========================================
# STEP 38 — MAKE PREDICTIONS
# ==========================================

# Predict churn for test customers
y_pred = model.predict(X_test)

print("\n" + "=" * 60)
print("MODEL PREDICTIONS")
print("=" * 60)

print("\nNumber of predictions:")
print(len(y_pred))

print("\nFirst 20 predictions:")
print(y_pred[:20])

print("\nPrediction distribution:")

# ==========================================
# STEP 39 — MODEL EVALUATION
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print("\nAccuracy:")
print(accuracy)

print("\nPrecision:")
print(precision)

print("\nRecall:")
print(recall)

print("\nF1 Score:")
print(f1)

print("\nConfusion Matrix:")
print(cm)

# ==========================================
# STEP 40 — CLASSIFICATION REPORT
# ==========================================

from sklearn.metrics import classification_report

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"]
    )
)

# ==========================================
# STEP 41 — BALANCED LOGISTIC REGRESSION
# ==========================================

balanced_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

# Train balanced model
balanced_model.fit(X_train, y_train)

# Predictions
y_pred_balanced = balanced_model.predict(X_test)

# Evaluation
balanced_accuracy = accuracy_score(
    y_test,
    y_pred_balanced
)

balanced_precision = precision_score(
    y_test,
    y_pred_balanced
)

balanced_recall = recall_score(
    y_test,
    y_pred_balanced
)

balanced_f1 = f1_score(
    y_test,
    y_pred_balanced
)

print("\n" + "=" * 60)
print("BALANCED LOGISTIC REGRESSION")
print("=" * 60)

print("\nAccuracy:")
print(balanced_accuracy)

print("\nPrecision:")
print(balanced_precision)

print("\nRecall:")
print(balanced_recall)

print("\nF1 Score:")
print(balanced_f1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_balanced))

# ==========================================
# STEP 42 — MODEL COMPARISON
# ==========================================

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Balanced Logistic Regression"
    ],
    "Accuracy": [
        accuracy,
        balanced_accuracy
    ],
    "Precision": [
        precision,
        balanced_precision
    ],
    "Recall": [
        recall,
        balanced_recall
    ],
    "F1 Score": [
        f1,
        balanced_f1
    ]
})

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    comparison.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format
        }
    )
)

# ==========================================
# STEP 43 — DECISION TREE
# ==========================================

from sklearn.tree import DecisionTreeClassifier

# Create Decision Tree model
tree_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

# Train the model
tree_model.fit(X_train, y_train)

print("\n" + "=" * 60)
print("DECISION TREE TRAINING")
print("=" * 60)

print("\nDecision Tree training completed successfully!")
# ==========================================
# STEP 44 — DECISION TREE PREDICTIONS
# ==========================================

# Make predictions
y_pred_tree = tree_model.predict(X_test)

print("\n" + "=" * 60)
print("DECISION TREE PREDICTIONS")
print("=" * 60)

print("\nNumber of predictions:")
print(len(y_pred_tree))

print("\nFirst 20 predictions:")
print(y_pred_tree[:20])

print("\nPrediction distribution:")
print(pd.Series(y_pred_tree).value_counts())

# ==========================================
# STEP 45 — DECISION TREE EVALUATION
# ==========================================

tree_accuracy = accuracy_score(
    y_test,
    y_pred_tree
)

tree_precision = precision_score(
    y_test,
    y_pred_tree
)

tree_recall = recall_score(
    y_test,
    y_pred_tree
)

tree_f1 = f1_score(
    y_test,
    y_pred_tree
)

tree_cm = confusion_matrix(
    y_test,
    y_pred_tree
)

print("\n" + "=" * 60)
print("DECISION TREE EVALUATION")
print("=" * 60)

print("\nAccuracy:")
print(tree_accuracy)

print("\nPrecision:")
print(tree_precision)

print("\nRecall:")
print(tree_recall)

print("\nF1 Score:")
print(tree_f1)

print("\nConfusion Matrix:")
print(tree_cm)

# ==========================================
# STEP 46 — FINAL MODEL COMPARISON
# ==========================================

final_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Balanced Logistic Regression",
        "Decision Tree"
    ],
    "Accuracy": [
        accuracy,
        balanced_accuracy,
        tree_accuracy
    ],
    "Precision": [
        precision,
        balanced_precision,
        tree_precision
    ],
    "Recall": [
        recall,
        balanced_recall,
        tree_recall
    ],
    "F1 Score": [
        f1,
        balanced_f1,
        tree_f1
    ]
})

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(
    final_comparison.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format
        }
    )
)

# ==========================================
# STEP 47 — RANDOM FOREST
# ==========================================

from sklearn.ensemble import RandomForestClassifier

# Create Random Forest model
forest_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

# Train the model
forest_model.fit(X_train, y_train)

print("\n" + "=" * 60)
print("RANDOM FOREST TRAINING")
print("=" * 60)

print("\nRandom Forest training completed successfully!")

# ==========================================
# STEP 48 — RANDOM FOREST PREDICTIONS
# ==========================================

# Make predictions
y_pred_forest = forest_model.predict(X_test)

print("\n" + "=" * 60)
print("RANDOM FOREST PREDICTIONS")
print("=" * 60)

print("\nNumber of predictions:")
print(len(y_pred_forest))

print("\nFirst 20 predictions:")
print(y_pred_forest[:20])

print("\nPrediction distribution:")
print(pd.Series(y_pred_forest).value_counts())

# ==========================================
# STEP 49 — RANDOM FOREST EVALUATION
# ==========================================

forest_accuracy = accuracy_score(
    y_test,
    y_pred_forest
)

forest_precision = precision_score(
    y_test,
    y_pred_forest
)

forest_recall = recall_score(
    y_test,
    y_pred_forest
)

forest_f1 = f1_score(
    y_test,
    y_pred_forest
)

forest_cm = confusion_matrix(
    y_test,
    y_pred_forest
)

print("\n" + "=" * 60)
print("RANDOM FOREST EVALUATION")
print("=" * 60)

print("\nAccuracy:")
print(forest_accuracy)

print("\nPrecision:")
print(forest_precision)

print("\nRecall:")
print(forest_recall)

print("\nF1 Score:")
print(forest_f1)

print("\nConfusion Matrix:")
print(forest_cm)

# ==========================================
# STEP 50 — BALANCED RANDOM FOREST
# ==========================================

balanced_forest_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Train balanced Random Forest
balanced_forest_model.fit(X_train, y_train)

# Make predictions
y_pred_balanced_forest = balanced_forest_model.predict(X_test)

# Evaluation
balanced_forest_accuracy = accuracy_score(
    y_test,
    y_pred_balanced_forest
)

balanced_forest_precision = precision_score(
    y_test,
    y_pred_balanced_forest
)

balanced_forest_recall = recall_score(
    y_test,
    y_pred_balanced_forest
)

balanced_forest_f1 = f1_score(
    y_test,
    y_pred_balanced_forest
)

balanced_forest_cm = confusion_matrix(
    y_test,
    y_pred_balanced_forest
)

print("\n" + "=" * 60)
print("BALANCED RANDOM FOREST")
print("=" * 60)

print("\nAccuracy:")
print(balanced_forest_accuracy)

print("\nPrecision:")
print(balanced_forest_precision)

print("\nRecall:")
print(balanced_forest_recall)

print("\nF1 Score:")
print(balanced_forest_f1)

print("\nConfusion Matrix:")
print(balanced_forest_cm)

# ==========================================
# STEP 51 — FEATURE IMPORTANCE
# ==========================================

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": balanced_forest_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("TOP 15 FEATURE IMPORTANCE")
print("=" * 60)

print(
    feature_importance.head(15).to_string(
        index=False
    )
)

# ==========================================
# STEP 52 — ACTUAL FEATURE NAMES
# ==========================================

print("\n" + "=" * 60)
print("FEATURE NAME CHECK")
print("=" * 60)

print("\nNumber of encoded features:")
print(len(X_train.columns) if hasattr(X_train, "columns") else X_train.shape[1])

# ==========================================
# STEP 54 — FEATURE IMPORTANCE WITH NAMES
# ==========================================

encoded_feature_names = [
    "gender_Female",
    "gender_Male",
    "Partner_No",
    "Partner_Yes",
    "Dependents_No",
    "Dependents_Yes",
    "PhoneService_No",
    "PhoneService_Yes",
    "MultipleLines_No",
    "MultipleLines_No phone service",
    "MultipleLines_Yes",
    "InternetService_DSL",
    "InternetService_Fiber optic",
    "InternetService_No",
    "OnlineSecurity_No",
    "OnlineSecurity_No internet service",
    "OnlineSecurity_Yes",
    "OnlineBackup_No",
    "OnlineBackup_No internet service",
    "OnlineBackup_Yes",
    "DeviceProtection_No",
    "DeviceProtection_No internet service",
    "DeviceProtection_Yes",
    "TechSupport_No",
    "TechSupport_No internet service",
    "TechSupport_Yes",
    "StreamingTV_No",
    "StreamingTV_No internet service",
    "StreamingTV_Yes",
    "StreamingMovies_No",
    "StreamingMovies_No internet service",
    "StreamingMovies_Yes",
    "Contract_Month-to-month",
    "Contract_One year",
    "Contract_Two year",
    "PaperlessBilling_No",
    "PaperlessBilling_Yes",
    "PaymentMethod_Bank transfer (automatic)",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

feature_importance = pd.DataFrame({
    "Feature": encoded_feature_names,
    "Importance": balanced_forest_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("TOP 15 FEATURE IMPORTANCE")
print("=" * 60)

print(
    feature_importance.head(15).to_string(
        index=False
    )
)

# ==========================================
# STEP 55 — SAVE FEATURE IMPORTANCE
# ==========================================

feature_importance_path = "Data/Processed/feature_importance.csv"

feature_importance.to_csv(
    feature_importance_path,
    index=False
)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE SAVING")
print("=" * 60)

print("\nFeature importance saved successfully!")
print("File:")
print(feature_importance_path)

# ==========================================
# STEP 56 — SAVE BEST MODEL
# ==========================================

import joblib

model_path = "Data/Processed/balanced_random_forest.pkl"

joblib.dump(
    balanced_forest_model,
    model_path
)

print("\n" + "=" * 60)
print("BEST MODEL SAVING")
print("=" * 60)

print("\nBest model:")
print("Balanced Random Forest")

print("\nModel saved successfully!")
print("File:")
print(model_path)

# ==========================================
# STEP 57 — LOAD SAVED MODEL
# ==========================================

loaded_model = joblib.load(model_path)

# Make predictions using the loaded model
loaded_predictions = loaded_model.predict(X_test)

print("\n" + "=" * 60)
print("SAVED MODEL VALIDATION")
print("=" * 60)

print("\nSaved model loaded successfully!")

print("\nNumber of predictions:")
print(len(loaded_predictions))

print("\nPredictions match original model:")
print(
    (loaded_predictions == y_pred_balanced_forest).all()
)

# ==========================================
# STEP 58 — SAVE MODEL COMPARISON
# ==========================================

model_comparison_path = "Data/Processed/model_comparison.csv"

final_model_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Balanced Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Balanced Random Forest"
    ],
    "Accuracy": [
        accuracy,
        balanced_accuracy,
        tree_accuracy,
        forest_accuracy,
        balanced_forest_accuracy
    ],
    "Precision": [
        precision,
        balanced_precision,
        tree_precision,
        forest_precision,
        balanced_forest_precision
    ],
    "Recall": [
        recall,
        balanced_recall,
        tree_recall,
        forest_recall,
        balanced_forest_recall
    ],
    "F1 Score": [
        f1,
        balanced_f1,
        tree_f1,
        forest_f1,
        balanced_forest_f1
    ]
})

final_model_comparison.to_csv(
    model_comparison_path,
    index=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON SAVING")
print("=" * 60)

print("\nModel comparison saved successfully!")
print("File:")
print(model_comparison_path)

# ==========================================
# STEP 50A — XGBOOST
# ==========================================

from xgboost import XGBClassifier

print("\n" + "=" * 60)
print("XGBOOST TRAINING")
print("=" * 60)

# Create XGBoost model
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

# Train XGBoost
xgb_model.fit(
    X_train,
    y_train
)

print("\nXGBoost training completed successfully!")

# ==========================================
# XGBOOST PREDICTIONS
# ==========================================

y_pred_xgb = xgb_model.predict(X_test)

print("\n" + "=" * 60)
print("XGBOOST PREDICTIONS")
print("=" * 60)

print("\nNumber of predictions:")
print(len(y_pred_xgb))

print("\nPrediction distribution:")
print(pd.Series(y_pred_xgb).value_counts())

# ==========================================
# XGBOOST EVALUATION
# ==========================================

xgb_accuracy = accuracy_score(
    y_test,
    y_pred_xgb
)

xgb_precision = precision_score(
    y_test,
    y_pred_xgb
)

xgb_recall = recall_score(
    y_test,
    y_pred_xgb
)

xgb_f1 = f1_score(
    y_test,
    y_pred_xgb
)

xgb_cm = confusion_matrix(
    y_test,
    y_pred_xgb
)

print("\n" + "=" * 60)
print("XGBOOST EVALUATION")
print("=" * 60)

print("\nAccuracy:")
print(xgb_accuracy)

print("\nPrecision:")
print(xgb_precision)

print("\nRecall:")
print(xgb_recall)

print("\nF1 Score:")
print(xgb_f1)

print("\nConfusion Matrix:")
print(xgb_cm)

# ==========================================
# ADD XGBOOST TO MODEL COMPARISON
# ==========================================

final_model_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Balanced Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Balanced Random Forest",
        "XGBoost"
    ],
    "Accuracy": [
        accuracy,
        balanced_accuracy,
        tree_accuracy,
        forest_accuracy,
        balanced_forest_accuracy,
        xgb_accuracy
    ],
    "Precision": [
        precision,
        balanced_precision,
        tree_precision,
        forest_precision,
        balanced_forest_precision,
        xgb_precision
    ],
    "Recall": [
        recall,
        balanced_recall,
        tree_recall,
        forest_recall,
        balanced_forest_recall,
        xgb_recall
    ],
    "F1 Score": [
        f1,
        balanced_f1,
        tree_f1,
        forest_f1,
        balanced_forest_f1,
        xgb_f1
    ]
})

print("\n" + "=" * 60)
print("UPDATED MODEL COMPARISON")
print("=" * 60)

print(
    final_model_comparison.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format
        }
    )
)

# ==========================================
# BEST MODEL BASED ON F1 SCORE
# ==========================================

best_model_name = final_model_comparison.loc[
    final_model_comparison["F1 Score"].idxmax(),
    "Model"
]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("\nBest model based on F1 Score:")
print(best_model_name)

# ==========================================
# STEP 59 — SAVE UPDATED MODEL COMPARISON
# ==========================================

final_model_comparison.to_csv(
    "Data/Processed/model_comparison.csv",
    index=False
)

print("\n" + "=" * 60)
print("UPDATED MODEL COMPARISON SAVING")
print("=" * 60)

print("\nUpdated model comparison saved successfully!")
print("File:")
print("Data/Processed/model_comparison.csv")