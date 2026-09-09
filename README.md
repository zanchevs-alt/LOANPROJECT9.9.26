# פרויקט – חיזוי אישור הלוואה

מערכת לחיזוי אישור/דחייה של בקשת הלוואה, מבוססת על מודל **SVC (Support
Vector Classifier)** בתוך `Pipeline` (עיבוד מקדים + מודל), עם REST API
ב-Flask ושני דפי אתר.

## התאמה לשלבי ההגשה (לפי קובץ ההנחיות)

| שלב בהנחיות | קבצים רלוונטיים | הערה |
|---|---|---|
| **שלב 1 – אימון המודל** | `STEP1_original_notebook.ipynb` (העבודה המקורית, ב-Jupyter) + `train_model.py` (הגרסה המתקדמת שבפועל רצה בפרויקט הסופי, עם `ColumnTransformer` לטיפול אוטומטי בערכים חסרים וקידוד קטגוריות) | שני הקבצים מאמנים את **אותו** מודל SVC ומגיעים לאותה דיוק (0.8455). ה-notebook הוא נקודת ההתחלה; `train_model.py` הוא הגרסה שמשולבת בפועל עם השרת. |
| **שלב 2 – דף נתוני המודל** | `model.py` (פונקציות Python שחושפות את נתוני המודל) + `app.py` (ה-REST API: `/model/info`, `/model/features`, `/model/samples`, `/model/metrics`, `/model/margins`) + `dashboard.html` (הדף המוצג בכתובת `/dashboard.html`) | |
| **שלב 3 – טופס משתמש + Predict** | `index.html` + `app.js` + `styles.css` (הטופס, בכתובת `/`) + `app.py` (`/model/predict`) + `model.py` (`predict_loan_status`) | |

## מבנה הפרויקט

| קובץ                     | תיאור                                                              |
|--------------------------|---------------------------------------------------------------------|
| `STEP1_original_notebook.ipynb` | שלב 1 (מקורי) – ה-Jupyter notebook הראשוני, בו נבחרו הפיצ'רים ואומן המודל לראשונה |
| `train_model.py`         | שלב 1 (סופי) – מאמן את המודל, שומר `svm_pipeline_model.pkl` ו-`model_metadata.json` |
| `model.py`                | שכבת גישה למודל: טעינה, שליפת נתונים, ופונקציית `predict_loan_status` |
| `app.py`                  | שרת Flask יחיד – חושף את כל ה-REST API ומגיש את שני דפי ה-HTML     |
| `index.html` + `app.js` + `styles.css` | שלב 3 – טופס בדיקת זכאות להלוואה (הדף הראשי, `/`)         |
| `dashboard.html`          | שלב 2 – דשבורד נתוני המודל (`/dashboard.html`)                     |
| `train.csv`                | דאטהסט האימון (Kaggle – Analytics Vidhya Loan Prediction)          |

## איך מריצים

```bash
pip install -r requirements.txt

# שלב 1 – אימון (מריץ פעם אחת, או בכל פעם שמעדכנים את train.csv)
python train_model.py

# שלב 2+3 – מפעיל שרת אחד שמגיש גם את ה-API וגם את שני הדפים
python app.py
```

לאחר ההרצה, השרת עולה בכתובת `http://127.0.0.1:5000`:
- `/` – טופס בדיקת זכאות להלוואה
- `/dashboard.html` – דשבורד נתוני המודל

## REST API

| Method | Path              | תיאור                                             |
|--------|-------------------|-----------------------------------------------------|
| GET    | `/model/info`     | סקירה כללית: אלגוריתם, פיצ'רים, מחלקות, קובץ המודל |
| GET    | `/model/features` | רשימת הפיצ'רים                                     |
| GET    | `/model/samples`  | דוגמאות מהדאטהסט                                   |
| GET    | `/model/metrics`  | accuracy / confusion matrix / classification report |
| GET    | `/model/margins`  | התפלגות ערכי margin על קבוצת הבדיקה                |
| POST   | `/model/predict`  | חיזוי לבקשה בודדת (JSON עם 7 הפיצ'רים)             |

`/model/predict` מחזירה שגיאת `400` עם הודעה ברורה על שדה חסר, ריק,
לא מספרי, או ערך קטגוריאלי לא מוכר. אם קובץ המודל לא קיים (למשל אם
`train_model.py` לא הורץ), כל ה-endpoints מחזירים `503` עם הודעה מתאימה.

## פריסה (Render וכו')

`app.js` פונה ל-API בכתובת **יחסית** (`/model/predict`), ולא ל-
`127.0.0.1` קבוע — כך שהאתר עובד גם מקומית וגם אחרי פריסה לדומיין
אמיתי, בלי שינויי קוד.
