const predictBtn = document.getElementById("predict-btn");
const resultCard = document.getElementById("result-card");

predictBtn.addEventListener("click", async () => {
  predictBtn.disabled = true;
  predictBtn.textContent = "Predicting...";

  const payload = {
    loan_amnt: parseFloat(document.getElementById("loan_amnt").value),
    int_rate: parseFloat(document.getElementById("int_rate").value),
    installment: parseFloat(document.getElementById("installment").value),
    annual_inc: parseFloat(document.getElementById("annual_inc").value),
    dti: parseFloat(document.getElementById("dti").value),
    open_acc: parseInt(document.getElementById("open_acc").value),
    pub_rec: parseInt(document.getElementById("pub_rec").value),
    revol_bal: parseFloat(document.getElementById("revol_bal").value),
    revol_util: parseFloat(document.getElementById("revol_util").value),
    total_acc: parseInt(document.getElementById("total_acc").value),
    pub_rec_bankruptcies: parseInt(document.getElementById("pub_rec_bankruptcies").value),
    emp_length: document.getElementById("emp_length").value,
    term: document.getElementById("term").value,
    grade: document.getElementById("grade").value,
    sub_grade: document.getElementById("sub_grade").value,
    home_ownership: document.getElementById("home_ownership").value,
    verification_status: document.getElementById("verification_status").value,
    purpose: document.getElementById("purpose").value,
    initial_list_status: "w",
    application_type: "Individual"
  };

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed");
    }

    showResult(data);

  } catch (error) {
    alert("Error: " + error.message);
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Predict Default Risk";
  }
});


function showResult(data) {
  const isDefault = data.prediction === "Default";
  const prob = data.default_probability;
  const pct = Math.round(prob * 100);
  const confidence = Math.round((1 - prob) * 100);

  resultCard.classList.remove("hidden");

  document.getElementById("prob-value").textContent = pct + "%";
  document.getElementById("prob-bar").style.width = pct + "%";
  document.getElementById("prob-bar").style.background = isDefault ? "#ef4444" : "#10b981";

  const badge = document.getElementById("result-badge");
  badge.textContent = isDefault ? "High Risk — Default" : "Low Risk — No Default";
  badge.className = "result-badge " + (isDefault ? "badge-default" : "badge-safe");

  document.getElementById("risk-label").textContent = isDefault ? "High Risk" : "Low Risk";
  document.getElementById("confidence").textContent = confidence + "%";

  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}