"""
שרת Flask אחד המשרת את כל הפרויקט: גם את ה-REST API שחושף את נתוני
המודל (שלב 2) וגם את endpoint ה-predict (שלב 3), וגם את שני דפי ה-HTML.

כל endpoint הוא "עטיפה" דקה סביב פונקציה מ-model.py - השרת עצמו לא מכיל
שום לוגיקה של טעינת קבצים או חישוב, רק ממיר את הפלט של הפונקציות ל-JSON.

דפי האתר:
    /                -> index.html      (טופס בדיקת זכאות להלוואה - שלב 3)
    /dashboard.html  -> dashboard.html  (דשבורד נתוני המודל - שלב 2)

איך מריצים:
    python app.py
השרת יעלה בכתובת: http://127.0.0.1:5000
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import model

app = Flask(__name__)

# מאפשר לדף ה-HTML (שיכול לרוץ מכתובת/פורט אחרים, או כקובץ מקומי)
# לבצע בקשות fetch לשרת הזה בלי שהדפדפן יחסום אותן (CORS)
CORS(app)

# שמות הקבצים הסטטיים שמותר להגיש דרך ה-route הכללי למטה
STATIC_ASSETS = {"styles.css", "app.js", "dashboard.html"}


@app.errorhandler(FileNotFoundError)
def handle_model_not_found(error):
    """
    אם קובץ המודל (svm_pipeline_model.pkl) או המטא-נתונים
    (model_metadata.json) עדיין לא נוצרו - לדוגמה אם האתר נפתח לפני
    שהריצו את train_model.py - מחזירים הודעה ברורה במקום שגיאת שרת
    "מכוערת" (500 גנרית).
    """
    return jsonify({"error": str(error)}), 503


@app.route("/", methods=["GET"])
def index():
    """Serve the loan-eligibility application form (Step 3)."""
    return send_from_directory(".", "index.html")


@app.route("/<path:asset>", methods=["GET"])
def static_asset(asset):
    """Serve the frontend stylesheet, client script, and dashboard page."""
    if asset in STATIC_ASSETS:
        return send_from_directory(".", asset)
    return jsonify({"error": "Not found"}), 404


@app.route("/model/info", methods=["GET"])
def model_info():
    """מחזיר סקירה כללית של המודל: אלגוריתם, פיצ'רים, מחלקות, קובץ המודל."""
    return jsonify(model.get_full_model_overview())


@app.route("/model/features", methods=["GET"])
def model_features():
    """מחזיר את רשימת הפיצ'רים שנבחרו לאימון המודל."""
    return jsonify(model.get_features())


@app.route("/model/samples", methods=["GET"])
def model_samples():
    """מחזיר כמה שורות לדוגמה מתוך הדאטהסט המקורי."""
    return jsonify(model.get_samples())


@app.route("/model/metrics", methods=["GET"])
def model_metrics():
    """מחזיר את מדדי הדיוק של המודל (accuracy / confusion matrix / classification report)."""
    return jsonify(model.get_metrics())


@app.route("/model/margins", methods=["GET"])
def model_margins():
    """מחזיר את התפלגות ערכי ה-margin על קבוצת הבדיקה (לגרף היסטוגרמה)."""
    return jsonify(model.get_margins_distribution())


@app.route("/model/predict", methods=["POST"])
def model_predict():
    """
    שלב 3 - מקבלת מהטופס באתר JSON עם פרטי בקשת הלוואה בודדת, ומחזירה
    את חיזוי המודל (Loan Approved / Loan Not Approved).

    גוף הבקשה (JSON) חייב להכיל ערך תקין לכל אחד מהפיצ'רים הבאים:
    CoapplicantIncome, ApplicantIncome, LoanAmount, Credit_History,
    Loan_Amount_Term, Self_Employed, Education
    (אפשר לקבל את הרשימה המדויקת גם מ-GET /model/features)
    """
    request_data = request.get_json(silent=True) or {}

    try:
        result = model.predict_loan_status(request_data)
    except ValueError as error:
        # שדות חסרים/לא תקינים בבקשה - שגיאת לקוח (400), לא שגיאת שרת
        return jsonify({"error": str(error)}), 400

    return jsonify(result)


if __name__ == "__main__":
    # debug=True נוח לפיתוח (מציג שגיאות מפורטות, טוען מחדש אוטומטית) -
    # יש לכבות (debug=False) לפני פריסה אמיתית לאינטרנט (למשל ב-Render)
    app.run(host="127.0.0.1", debug=False, port=5000)
