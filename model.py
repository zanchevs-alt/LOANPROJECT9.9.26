"""
שכבת הגישה למודל (Model Layer).

הקובץ הזה טוען את ה-Pipeline המאומן (svm_pipeline_model.pkl) ואת
מטא-הנתונים (model_metadata.json) פעם אחת בזיכרון (lazy load), וחושף
פונקציות Python "נקיות" שה-API (app.py) רק עוטף ב-JSON - בלי שום לוגיקה
של קבצים או חישוב בתוך app.py עצמו.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE_PATH = BASE_DIR / "svm_pipeline_model.pkl"
METADATA_FILE_PATH = BASE_DIR / "model_metadata.json"

DEFAULT_FEATURES = [
    "CoapplicantIncome",
    "ApplicantIncome",
    "LoanAmount",
    "Credit_History",
    "Loan_Amount_Term",
    "Self_Employed",
    "Education",
]

# ערכים חוקיים לפיצ'רים הקטגוריאליים/בינאריים - לבדיקת תקינות לפני predict
ALLOWED_CATEGORICAL_VALUES = {
    "Self_Employed": {"Yes", "No"},
    "Education": {"Graduate", "Not Graduate"},
}
ALLOWED_CREDIT_HISTORY_VALUES = {0, 1}

# משתנים "פרטיים" למודול - נטענים פעם אחת בלבד, כדי שלא נטען מהדיסק בכל קריאה
_pipeline = None
_metadata = None


def _ensure_loaded() -> None:
    """טוען את המודל ואת מטא-הנתונים לזיכרון, פעם אחת בלבד (lazy load)."""
    global _pipeline, _metadata

    if _pipeline is None:
        if not MODEL_FILE_PATH.exists():
            raise FileNotFoundError(
                f"קובץ המודל '{MODEL_FILE_PATH.name}' לא נמצא. "
                "יש להריץ קודם את train_model.py כדי לאמן ולשמור את המודל."
            )
        _pipeline = joblib.load(MODEL_FILE_PATH)

    if _metadata is None:
        if not METADATA_FILE_PATH.exists():
            raise FileNotFoundError(
                f"קובץ המטא-נתונים '{METADATA_FILE_PATH.name}' לא נמצא. "
                "יש להריץ קודם את train_model.py כדי לייצר אותו."
            )
        _metadata = json.loads(METADATA_FILE_PATH.read_text(encoding="utf-8"))


def get_pipeline():
    """מחזיר את ה-Pipeline המאומן (StandardScaler/OneHotEncoder + SVC)."""
    _ensure_loaded()
    return _pipeline


def get_model_info() -> dict:
    """פרטי המודל עצמו: אלגוריתם, kernel, מספר Support Vectors וכו'."""
    _ensure_loaded()
    return _metadata["model_info"]


def get_features() -> dict:
    """רשימת הפיצ'רים שנבחרו לאימון, מחולקים לפי סוג הטיפול שקיבלו."""
    _ensure_loaded()
    return _metadata.get("features", {"all": DEFAULT_FEATURES})


def get_samples() -> list:
    """כמה שורות לדוגמה מתוך הדאטהסט המקורי, להצגה בדף."""
    _ensure_loaded()
    return _metadata["samples"]


def get_metrics() -> dict:
    """מדדי הדיוק של המודל: accuracy, confusion matrix, classification report."""
    _ensure_loaded()
    return _metadata["metrics"]


def get_margins_distribution() -> list:
    """ערכי ה-margin (decision_function) על קבוצת הבדיקה - להיסטוגרמה."""
    _ensure_loaded()
    return _metadata["margins_on_test_set"]


def get_model_file_info() -> dict:
    """שם קובץ המודל השמור וגודלו."""
    _ensure_loaded()
    return _metadata["model_file"]


def get_classes() -> list:
    """התוויות האפשריות שהמודל חוזה (למשל ['N', 'Y'])."""
    _ensure_loaded()
    return _metadata["classes"]


def compute_margin(input_df: pd.DataFrame) -> list:
    """
    מחשבת את ה-margin (decision_function) עבור שורת/שורות קלט חדשות.

    ה-margin הוא המרחק המסומן (signed distance) של הדוגמה ממישור המפריד
    (hyperplane) של ה-SVM: ערך חיובי משמעו שהדוגמה בצד המחלקה החיובית (Y),
    ערך שלילי - בצד המחלקה השנייה (N), וככל שהערך רחוק יותר מ-0 - המודל
    "בטוח" יותר בחיזוי.

    input_df: DataFrame עם אותן עמודות הפיצ'רים המקוריות (לא מקודדות) -
    הקידוד והנרמול קורים אוטומטית בתוך ה-Pipeline.
    """
    pipeline = get_pipeline()
    return pipeline.decision_function(input_df).tolist()


def get_full_model_overview() -> dict:
    """מרכזת את כל המידע על המודל למקום אחד - נוח ל-endpoint אחד מסכם."""
    return {
        "model_info": get_model_info(),
        "features": get_features(),
        "classes": get_classes(),
        "model_file": get_model_file_info(),
    }


# --- שלב 3: פונקציית החיזוי עבור בקשה בודדת מהמשתמש ---

APPROVED_CLASS = "Y"  # הערך ב-Loan_Status שמסמן הלוואה מאושרת


def _validate_and_build_row(request_data: dict) -> dict:
    """
    בודקת שכל הפיצ'רים הנדרשים קיימים, לא ריקים, ומהסוג/מהערכים הצפויים,
    ומחזירה dict "נקי" מוכן להפוך ל-DataFrame של שורה אחת.

    זורקת ValueError עם הודעה ברורה (בעברית, מתאימה להצגה למשתמש) על כל
    בעיה - שדה חסר, ריק, לא מספרי, או ערך קטגוריאלי לא מוכר - כדי שה-API
    יחזיר 400 נקי במקום להדליף שגיאת sklearn גולמית.
    """
    required_features = get_features()["all"]

    missing = [
        f for f in required_features
        if f not in request_data or request_data[f] in (None, "")
    ]
    if missing:
        raise ValueError(f"חסרים השדות הבאים בבקשה: {', '.join(missing)}")

    row = {}
    for feature in required_features:
        value = request_data[feature]

        if feature in ALLOWED_CATEGORICAL_VALUES:
            if value not in ALLOWED_CATEGORICAL_VALUES[feature]:
                allowed = "/".join(sorted(ALLOWED_CATEGORICAL_VALUES[feature]))
                raise ValueError(f"ערך לא חוקי לשדה {feature}: '{value}' (אפשריים: {allowed})")
            row[feature] = value
            continue

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"ערך לא מספרי לשדה {feature}: '{value}'")

        if feature == "Credit_History" and numeric_value not in ALLOWED_CREDIT_HISTORY_VALUES:
            raise ValueError("ערך לא חוקי לשדה Credit_History (אפשריים: 0/1)")

        row[feature] = numeric_value

    return row


def predict_loan_status(request_data: dict) -> dict:
    """
    מקבלת dict אחד עם פרטי בקשת הלוואה (כמו שמגיע מהטופס באתר), ומחזירה
    את חיזוי המודל: התוצאה הגולמית (Y/N), האם ההלוואה אושרה, ה-margin
    (מידת ה"ביטחון" של המודל), והודעה מוכנה להצגה למשתמש.

    request_data חייב להכיל ערך תקין לכל אחד מהפיצ'רים שהמודל אומן עליהם
    (הרשימה נמצאת ב-get_features()["all"]) - אחרת נזרקת ValueError עם
    הודעה ברורה, כדי שה-API יוכל להחזיר שגיאת לקוח (400) נקייה.
    """
    row = _validate_and_build_row(request_data)
    input_row = pd.DataFrame([row])

    pipeline = get_pipeline()
    try:
        prediction = pipeline.predict(input_row)[0]
        margin = float(pipeline.decision_function(input_row)[0])
    except Exception as exc:  # noqa: BLE001 - מתורגם להודעת לקוח ברורה
        raise ValueError(f"לא ניתן לבצע חיזוי עם הנתונים שסופקו: {exc}") from exc

    approved = bool(prediction == APPROVED_CLASS)

    return {
        "prediction": str(prediction),
        "approved": approved,
        "margin": round(margin, 4),
        "message": "Loan Approved" if approved else "Loan Not Approved",
    }
