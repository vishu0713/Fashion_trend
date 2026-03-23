import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1. Load dataset
# -----------------------------
INPUT_FILE = "data/processed/fashion_trend_features.csv"

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded")
print("Shape before cleaning:", df.shape)

# -----------------------------
# 2. Encode keyword column
# -----------------------------
le = LabelEncoder()
df["keyword_encoded"] = le.fit_transform(df["keyword"])

# -----------------------------
# 3. Replace inf values
# -----------------------------
df = df.replace([np.inf, -np.inf], np.nan)

# Drop rows with missing values in model columns
model_columns = [
    "keyword_encoded",
    "trend",
    "lag1",
    "lag2",
    "lag3",
    "ma3",
    "ma6",
    "std3",
    "slope",
    "pct_change",
    "target"
]

df = df[model_columns].dropna()

print("Shape after cleaning:", df.shape)

# -----------------------------
# 4. Define features and target
# -----------------------------
X = df[
    [
        "keyword_encoded",
        "trend",
        "lag1",
        "lag2",
        "lag3",
        "ma3",
        "ma6",
        "std3",
        "slope",
        "pct_change",
    ]
]

y = df["target"]

# -----------------------------
# 5. Train / Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# -----------------------------
# 6. Scale data for Logistic Regression
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 7. Train Logistic Regression
# -----------------------------
log_model = LogisticRegression(max_iter=2000)
log_model.fit(X_train_scaled, y_train)

log_pred = log_model.predict(X_test_scaled)
log_acc = accuracy_score(y_test, log_pred)

print("\nLogistic Regression Accuracy:", round(log_acc, 4))
print(classification_report(y_test, log_pred))

# -----------------------------
# 8. Train Random Forest
# -----------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:", round(rf_acc, 4))
print(classification_report(y_test, rf_pred))

# -----------------------------
# 9. Choose best model
# -----------------------------
if rf_acc > log_acc:
    best_model = rf_model
    best_model_name = "RandomForestClassifier"
else:
    best_model = log_model
    best_model_name = "LogisticRegression"

print("\nBest model selected:", best_model_name)

# -----------------------------
# 10. Save model files
# -----------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/trend_classifier.pkl")
joblib.dump(le, "models/keyword_encoder.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel saved successfully")
print("models/trend_classifier.pkl")
print("models/keyword_encoder.pkl")
print("models/scaler.pkl")