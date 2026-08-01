import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# ---------------------------
# Task 1: Data Understanding and Preprocessing
# ---------------------------

df = pd.read_csv("heart.csv")

print("First five records:")
print(df.head())

target_col = "target"
numerical_features = [col for col in df.columns if col != target_col]

print("\nNumerical features:", numerical_features)
print("Target variable:", target_col)

print("\nMissing values per column:")
print(df.isnull().sum())

X = df[numerical_features]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# ---------------------------
# Task 2: Model Development
# ---------------------------

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

joblib.dump(model, "model.pkl")
joblib.dump(numerical_features, "feature_names.pkl")
print("\nModel saved as model.pkl")
