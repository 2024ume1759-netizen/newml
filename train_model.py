# train_model.py — train a Logistic Regression classifier on Student Productivity data
#
# dataset  : Final_200_Productivity_Clean.csv (200 student survey responses)
# task     : multi-class classification → Low / Medium / High productivity
# model    : Logistic Regression (best Macro F1 = 0.69 among 6 models tested)
# exports  : preprocessor.pkl, model.pkl, metadata.pkl
#
# run once before launching the app:
#   pip install pandas scikit-learn imbalanced-learn
#   python train_model.py

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# ── 1. load & clean data ──────────────────────────────────────────────────────

df = pd.read_csv("Final_200_Productivity_Clean.csv")
df = df.drop(columns=["Timestamp"])
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.lower().str.strip()

# fix categorical columns
class_map  = {"4-5": "4-5 hrs", "5-6": "5-6 hrs", "6-7": "6-7 hrs", "7+": "7+ hrs"}
sleep_map  = {"<4": "<4 hrs", "4-5": "4-5 hrs", "5-6": "5-6 hrs",
              "6-7": "6-7 hrs", "7-8": "7-8 hrs", "8+": "8+ hrs"}
screen_map = {"<4": "<4 hrs", "4-5": "4-5 hrs", "5-6": "5-6 hrs", "6+": "6+ hrs"}
weekend_map = {
    "mostly watching tv / social media":        "mostly watching tv / social media",
    "mixed leisure and some study":              "mixed leisure and some study",
    "mostly studying / completing assignments":  "mostly studying / completing assignments",
    "mostly relaxation":                         "mostly relaxing / doing nothing",
    "mostly relaxing / doing nothing":           "mostly relaxing / doing nothing",
    "balanced (study + relaxation)":             "balanced (study + relaxation)",
    "mostly studying":                           "mostly studying / completing assignments",
}
study_map = {"0-2": 1, "3-5": 4, "6-8": 7, "9-12": 10}

df["Class (hrs/day)"]    = df["Class (hrs/day)"].map(class_map)
df["Sleep (hrs/night)"]  = df["Sleep (hrs/night)"].map(sleep_map)
df["Screen Time (hrs/day)"] = df["Screen Time (hrs/day)"].map(screen_map)
df["Weekend"]            = df["Weekend"].map(weekend_map)
df["Weekly Study (hrs)"] = df["Weekly Study (hrs)"].map(study_map)
df = df.fillna(df.mode().iloc[0])

# ── 2. encode target ──────────────────────────────────────────────────────────

productivity_order = ["Low", "Medium", "High"]
df["Overall Productivity"] = pd.Categorical(
    df["Overall Productivity"],
    categories=[x.lower() for x in productivity_order],
    ordered=True
).codes

print("Class distribution:")
print(df["Overall Productivity"].value_counts().rename({0: "Low", 1: "Medium", 2: "High"}))

# ── 3. features & preprocessing ───────────────────────────────────────────────

X = df.drop("Overall Productivity", axis=1)
y = df["Overall Productivity"]

ordinal_cols = {
    "Year of study":              ["1st year", "2nd year"],
    "Class (hrs/day)":            ["4-5 hrs", "5-6 hrs", "6-7 hrs", "7+ hrs"],
    "Sleep (hrs/night)":          ["<4 hrs", "4-5 hrs", "5-6 hrs", "6-7 hrs", "7-8 hrs", "8+ hrs"],
    "Screen Time (hrs/day)":      ["<4 hrs", "4-5 hrs", "5-6 hrs", "6+ hrs"],
    "Academic work completion":   ["none", "very little", "moderate", "high", "very high"],
    "Weekly Attendance":          ["less than 50%", "about half", "most classes", "nearly all", "all classes"],
    "Understanding of lectures":  ["very poor", "poor", "average", "good", "excellent"],
    "Study time management":      ["very poor", "poor", "average", "good", "excellent"],
    "Class participation":        ["never", "rarely", "sometimes", "often", "always"],
}
onehot_cols = ["Branch", "Weekend"]

for col, order in ordinal_cols.items():
    mode_val = X[col].mode()[0]
    X[col] = X[col].apply(lambda x: x if x in order else mode_val)

preprocessor = ColumnTransformer(transformers=[
    ("ordinal", OrdinalEncoder(
        categories=list(ordinal_cols.values()),
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ), list(ordinal_cols.keys())),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ), onehot_cols),
], remainder="passthrough")

# ── 4. train / test split + SMOTE ─────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

smote = SMOTE(random_state=42, k_neighbors=4)
X_train_sm, y_train_sm = smote.fit_resample(X_train_proc, y_train)

# ── 5. train all models ───────────────────────────────────────────────────────

models_to_train = {
    "KNN":               KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Random Forest":     RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM":               SVC(kernel="rbf", C=5, class_weight="balanced", random_state=42, probability=True),
    "ANN":               MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
}

results = {}
print("\nTraining all models...")
print("-" * 40)

for name, clf in models_to_train.items():
    clf.fit(X_train_sm, y_train_sm)
    y_pred = clf.predict(X_test_proc)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")
    results[name] = {"model": clf, "accuracy": round(acc, 4), "f1_macro": round(f1, 4), "y_pred": y_pred}
    print(f"{name:<22} | Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")

# pick best model by macro F1
best_name = max(results, key=lambda n: results[n]["f1_macro"])
best_model = results[best_name]["model"]
print(f"\nBest model: {best_name} (F1={results[best_name]['f1_macro']:.4f})")

# ── 6. detailed report for best model ─────────────────────────────────────────

y_best_pred = results[best_name]["y_pred"]
print(f"\nClassification Report — {best_name}:")
print(classification_report(y_test, y_best_pred, target_names=productivity_order))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_best_pred))

# ── 7. feature importance (permutation-style proxy via RF) ────────────────────

rf_model = results["Random Forest"]["model"]
onehot_feature_names = preprocessor.named_transformers_["onehot"].get_feature_names_out(onehot_cols).tolist()
feature_names_out = list(ordinal_cols.keys()) + onehot_feature_names + ["Weekly Study (hrs)"]
rf_importances = dict(zip(feature_names_out, rf_model.feature_importances_))

# ── 8. export artifacts ───────────────────────────────────────────────────────

model_comparison = {
    name: {"accuracy": v["accuracy"], "f1_macro": v["f1_macro"]}
    for name, v in results.items()
}

metadata = {
    "best_model_name":    best_name,
    "best_accuracy":      results[best_name]["accuracy"],
    "best_f1_macro":      results[best_name]["f1_macro"],
    "class_names":        productivity_order,
    "model_comparison":   model_comparison,
    "feature_names_out":  feature_names_out,
    "rf_importances":     rf_importances,
    "ordinal_cols":       ordinal_cols,
    "onehot_cols":        onehot_cols,
    # slider / selectbox options for the UI
    "branch_options":     sorted(X["Branch"].dropna().unique().tolist()),
    "weekend_options":    sorted(X["Weekend"].dropna().unique().tolist()),
    "weekly_study_options": [1, 4, 7, 10],
    "weekly_study_labels":  {"1": "0-2 hrs", "4": "3-5 hrs", "7": "6-8 hrs", "10": "9-12 hrs"},
    # class distribution for charts
    "class_distribution": df["Overall Productivity"].value_counts().rename({0: "Low", 1: "Medium", 2: "High"}).to_dict(),
    "branch_distribution": df["Branch"].value_counts().to_dict(),
    "confusion_matrix":   confusion_matrix(y_test, y_best_pred).tolist(),
}

artifacts = {
    "preprocessor.pkl": preprocessor,
    "model.pkl":        best_model,
    "metadata.pkl":     metadata,
}

for filename, obj in artifacts.items():
    with open(filename, "wb") as f:
        pickle.dump(obj, f)
    print(f"saved → {filename}")

print("\n✅ Done! Now run:  streamlit run app.py")
