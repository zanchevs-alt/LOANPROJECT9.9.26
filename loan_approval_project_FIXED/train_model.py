"""
שלב 1 - אימון מודל SVC לחיזוי אישור הלוואה (Loan_Status).

כל הטיפול בנתונים - מילוי ערכים חסרים, נרמול הפיצ'רים המספריים וקידוד
הפיצ'רים הקטגוריאליים - נמצא בתוך אותו Pipeline יחד עם המודל עצמו.
כך כשה-Pipeline נשמר לקובץ (joblib.dump) ונטען מחדש בשלב 3 כדי לבצע
predict על שורת נתונים בודדת שמילא משתמש בטופס, כל שלבי העיבוד המקדים
יקרו אוטומטית ובאופן עקבי - בלי צורך להתאים ידנית עמודות דמה.

איך מריצים:
    python train_model.py
הסקריפט קורא train.csv מהתיקייה הנוכחית, מאמן את המודל, ושומר שני קבצים:
    - svm_pipeline_model.pkl   (ה-Pipeline המאומן, לשימוש ב-model.py/app.py)
    - model_metadata.json     (כל נתוני המודל, ל-API ולדשבורד של שלב 2)
"""

import json
import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

MODEL_FILE_PATH = "svm_pipeline_model.pkl"
METADATA_FILE_PATH = "model_metadata.json"

# --- Step 1: טעינת ה-CSV למסגרת נתונים (DataFrame) ---
df = pd.read_csv("train.csv")

# --- Step 2: הגדרת קבוצות הפיצ'רים לפי סוג הטיפול הנדרש ---
# מספריים "רציפים" - חוסרים ימולאו לפי median
numeric_median_features = ["CoapplicantIncome", "ApplicantIncome", "LoanAmount"]

# מספריים "קטגוריאליים באופיים" (Credit_History הוא 0/1, Loan_Amount_Term
# מקבל בפועל מספר קטן של ערכים אפשריים) - חוסרים ימולאו לפי mode
numeric_mode_features = ["Credit_History", "Loan_Amount_Term"]

# קטגוריאליים טקסטואליים - חוסרים ימולאו לפי mode, ואז קידוד ל-0/1
categorical_features = ["Self_Employed", "Education"]

features = numeric_median_features + numeric_mode_features + categorical_features
target = "Loan_Status"

# --- Step 3: הפרדת הפיצ'רים (X) מהמטרה (y) ---
X = df[features].copy()
y = df[target].copy()

# --- Step 4: פיצול ל-Train ו-Test, עם stratify לשמירה על יחס המחלקות ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Step 5: הגדרת שלב העיבוד המקדים (ColumnTransformer) ---
numeric_median_transformer = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

numeric_mode_transformer = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num_median", numeric_median_transformer, numeric_median_features),
        ("num_mode", numeric_mode_transformer, numeric_mode_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# --- Step 6: בניית ה-Pipeline המלא: עיבוד מקדים + מודל SVC ---
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("model", SVC(kernel="rbf", random_state=42)),
    ]
)

# --- Step 7: אימון ה-Pipeline כולו ---
pipeline.fit(X_train, y_train)

# --- Step 8: בדיקה על קבוצת ה-Test וחישוב מדדי דיוק ---
y_pred = pipeline.predict(X_test)

print("--- Accuracy ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}\n")

print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# --- Step 9: שמירת ה-Pipeline המאומן (כולל העיבוד המקדים) לקובץ ---
joblib.dump(pipeline, MODEL_FILE_PATH)
print(f"\nהמודל נשמר בהצלחה לקובץ '{MODEL_FILE_PATH}'")

# --- Step 10: שמירת "מטא-נתוני" המודל לקובץ JSON, לשימוש שלב 2 (ה-API) ---
svc_model = pipeline.named_steps["model"]

sample_rows = (
    df[features + [target]]
    .head(10)
    .astype(object)
    .where(pd.notnull(df[features + [target]].head(10)), None)
    .to_dict(orient="records")
)

metadata = {
    "features": {
        "numeric_median_fill": numeric_median_features,
        "numeric_mode_fill": numeric_mode_features,
        "categorical": categorical_features,
        "all": features,
    },
    "target": target,
    "classes": svc_model.classes_.tolist(),
    "model_info": {
        "algorithm": "SVC (Support Vector Classifier)",
        "kernel": svc_model.kernel,
        "C": svc_model.C,
        "gamma": str(svc_model.gamma),
        "decision_function_shape": svc_model.decision_function_shape,
        "total_support_vectors": int(sum(svc_model.n_support_)),
        "support_vectors_per_class": {
            str(cls): int(count)
            for cls, count in zip(svc_model.classes_, svc_model.n_support_)
        },
        "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    },
    "metrics": {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "confusion_matrix_labels": svc_model.classes_.tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
    },
    "margins_on_test_set": pipeline.decision_function(X_test).tolist(),
    "samples": sample_rows,
    "model_file": {
        "name": MODEL_FILE_PATH,
        "size_kb": None,  # מתעדכן מטה, אחרי שהקובץ נשמר בפועל
    },
}

metadata["model_file"]["size_kb"] = round(os.path.getsize(MODEL_FILE_PATH) / 1024, 2)

with open(METADATA_FILE_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"מטא-הנתונים של המודל נשמרו בהצלחה לקובץ '{METADATA_FILE_PATH}'")
