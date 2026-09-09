const form = document.querySelector("#loan-form");
const submitButton = document.querySelector("#submit-button");
const result = document.querySelector("#result");
const resultTitle = document.querySelector("#result-title");
const resultDetail = document.querySelector("#result-detail");
const dismissResult = document.querySelector("#dismiss-result");

// כתובת יחסית: עובדת גם מקומית (השרת מגיש גם את index.html וגם את ה-API
// מאותו origin) וגם אחרי פריסה לאינטרנט (למשל Render) - בלי לקבע 127.0.0.1.
// אם הדף נפתח כקובץ מקומי (file://) עדיין נצטרך את הכתובת המלאה.
const isLocalHost = ["localhost", "127.0.0.1"].includes(window.location.hostname);
const API_URL = window.location.protocol === "file:" || (isLocalHost && window.location.port !== "5000")
  ? "http://127.0.0.1:5000/model/predict"
  : "/model/predict";

function showResult(type, title, detail) {
  result.className = `result ${type}`;
  resultTitle.textContent = title;
  resultDetail.textContent = detail;
  result.hidden = false;
}

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  submitButton.querySelector(".button-text").textContent = isLoading ? "Reading the signal..." : "Check Loan Eligibility";
  submitButton.querySelector(".button-arrow").textContent = isLoading ? "…" : "↗";
}

function requestPayload() {
  const formData = new FormData(form);
  return {
    ApplicantIncome: Number(formData.get("ApplicantIncome")),
    CoapplicantIncome: Number(formData.get("CoapplicantIncome")),
    LoanAmount: Number(formData.get("LoanAmount")),
    Loan_Amount_Term: Number(formData.get("Loan_Amount_Term")),
    Credit_History: Number(formData.get("Credit_History")),
    Self_Employed: formData.get("Self_Employed"),
    Education: formData.get("Education")
  };
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!form.reportValidity()) return;

  setLoading(true);
  result.hidden = true;

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestPayload())
    });
    const data = await response.json().catch(() => ({}));

    if (response.ok) {
      const approved = data.approved === true;
      showResult(
        approved ? "approved" : "rejected",
        approved ? "Loan Approved" : "Loan Not Approved",
        approved ? "The model found a positive eligibility signal." : "The model did not find enough signal for approval."
      );
    } else if (response.status === 400) {
      showResult("error", "We need one more detail", data.error || "Please check the application fields and try again.");
    } else if (response.status === 503) {
      showResult("error", "The system is initializing", data.error || "The model file was not found. Please train the model and save it before running the app.");
    } else {
      showResult("error", "Something went wrong", data.error || "We could not complete the check right now. Please try again.");
    }
  } catch (error) {
    showResult("error", "The model is out of reach", "Please make sure the API is running, then try again.");
  } finally {
    setLoading(false);
  }
});

dismissResult.addEventListener("click", () => {
  result.hidden = true;
});
