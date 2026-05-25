# =========================================================
# PHASE 1: TODDLER AUTISM SCREENING MODEL
# Logistic Regression + Feature Importance + Saved Outputs
# =========================================================

# ---------------------------------------------------------
# STEP 1: IMPORTS
# ---------------------------------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ---------------------------------------------------------
# PROJECT DIRECTORIES
# ---------------------------------------------------------
# ---------------------------------------------------------
# PROJECT DIRECTORIES
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

models_dir = os.path.join(BASE_DIR, "models")

graphs_dir = os.path.join(BASE_DIR, "reports")

results_dir = os.path.join(BASE_DIR, "reports")

os.makedirs(models_dir, exist_ok=True)

os.makedirs(graphs_dir, exist_ok=True)

os.makedirs(results_dir, exist_ok=True)

# ---------------------------------------------------------
# STEP 2: LOAD DATASET
# ---------------------------------------------------------
df = pd.read_csv("data/Toddler.csv")

df.columns = df.columns.str.strip()

print("Dataset Preview:")
print(df.head())

# ---------------------------------------------------------
# STEP 3: REMOVE LEAKAGE COLUMNS
# ---------------------------------------------------------
drop_cols = [
    'Case_No',
    'Who completed the test',
    'Qchat-10-Score'
]

for col in drop_cols:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# ---------------------------------------------------------
# STEP 4: CLEAN + ENCODE
# ---------------------------------------------------------
df.dropna(inplace=True)

df = pd.get_dummies(df, drop_first=True)

# ---------------------------------------------------------
# STEP 5: FEATURES + TARGET
# ---------------------------------------------------------
# ---------------------------------------------------------
# STEP 5: FEATURES + TARGET
# UI ALIGNED
# ---------------------------------------------------------

target_column = (
    "Class/ASD Traits_Yes"
)

selected_cols = [

    "A1",
    "A2",
    "A3",
    "A4",
    "A5",

    "A6",
    "A7",
    "A8",
    "A9",
    "A10",

    "Age_Mons"
]

X_phase1 = df[
    selected_cols
]

print(
    "Training Columns:"
)

print(
    X_phase1.columns.tolist()
)

y_phase1 = df[
    target_column
]

# ---------------------------------------------------------
# STEP 6: SCALE FEATURES
# ---------------------------------------------------------
phase1_scaler = StandardScaler()

X_phase1_scaled = phase1_scaler.fit_transform(X_phase1)

# ---------------------------------------------------------
# STEP 7: TRAIN-TEST SPLIT
# ---------------------------------------------------------
X_train_phase1, X_test_phase1, y_train_phase1, y_test_phase1 = train_test_split(
    X_phase1_scaled,
    y_phase1,
    test_size=0.2,
    random_state=42,
    stratify=y_phase1
)

# ---------------------------------------------------------
# STEP 8: TRAIN LOGISTIC REGRESSION
# ---------------------------------------------------------
phase1_model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    penalty='l2',
    solver='liblinear'
)

phase1_model.fit(X_train_phase1, y_train_phase1)

# ---------------------------------------------------------
# STEP 9: EVALUATION
# ---------------------------------------------------------
phase1_y_pred = phase1_model.predict(X_test_phase1)

phase1_accuracy = accuracy_score(
    y_test_phase1,
    phase1_y_pred
) * 100

phase1_probability = np.mean(
    phase1_model.predict_proba(X_test_phase1)[:,1]
) * 100

phase1_report = classification_report(
    y_test_phase1,
    phase1_y_pred
)

phase1_cm = confusion_matrix(
    y_test_phase1,
    phase1_y_pred
)

print("\n========================================")
print("PHASE 1 RESULTS")
print("========================================")

print(f"Accuracy: {phase1_accuracy:.2f}%")

print("\nClassification Report:")
print(phase1_report)

print("\nConfusion Matrix:")
print(phase1_cm)

# ---------------------------------------------------------
# STEP 10: FEATURE IMPORTANCE
# ---------------------------------------------------------
coefficients = pd.DataFrame({
    'Feature': X_phase1.columns,
    'Coefficient': phase1_model.coef_[0]
})

coefficients['Absolute Importance'] = coefficients[
    'Coefficient'
].abs()

coefficients = coefficients.sort_values(
    by='Absolute Importance',
    ascending=False
)

# ---------------------------------------------------------
# STEP 11: FEATURE IMPORTANCE GRAPH
# ---------------------------------------------------------
plt.figure(figsize=(18,9))

colors = [
    'royalblue' if coef > 0 else 'crimson'
    for coef in coefficients['Coefficient']
]

plt.bar(
    coefficients['Feature'],
    coefficients['Coefficient'],
    color=colors,
    edgecolor='black'
)

plt.axhline(0, color='black')

plt.title(
    "Phase 1 Feature Importance Analysis",
    fontsize=18,
    fontweight='bold'
)

plt.xticks(rotation=75)

plt.tight_layout()

# SAVE GRAPH
plt.savefig(
    os.path.join(
        graphs_dir,
        "phase1_feature_importance.png"
    )
)

plt.show()

# ---------------------------------------------------------
# STEP 12: SAVE MODELS
# ---------------------------------------------------------
joblib.dump(
    phase1_model,
    os.path.join(
        models_dir,
        "phase1_logistic_model.pkl"
    )
)

joblib.dump(
    phase1_scaler,
    os.path.join(
        models_dir,
        "phase1_scaler.pkl"
    )
)

# ---------------------------------------------------------
# SAVE FEATURE COLUMNS
# ---------------------------------------------------------

joblib.dump(
    X_phase1.columns.tolist(),
    os.path.join(
        models_dir,
        "phase1_feature_columns.pkl"
    )
)

# ---------------------------------------------------------
# STEP 13: SAVE RESULTS
# ---------------------------------------------------------
with open(
    os.path.join(
        results_dir,
        "phase1_results.txt"
    ),
    "w"
) as f:

    f.write("PHASE 1: TODDLER AUTISM SCREENING MODEL\n")

    f.write("=====================================\n\n")

    f.write(f"Accuracy: {phase1_accuracy:.2f}%\n")

    f.write(f"Average Screening Risk: {phase1_probability:.2f}%\n\n")

    f.write("Classification Report:\n")

    f.write(phase1_report)

    f.write("\nConfusion Matrix:\n")

    f.write(str(phase1_cm))

    f.write("\n\nFeature Importance Ranking:\n")

    f.write(coefficients.to_string())

# ---------------------------------------------------------
# COMPLETION MESSAGE
# ---------------------------------------------------------
print("\nPhase 1 completed successfully.")

print("Model, scaler, graph, and results saved.")