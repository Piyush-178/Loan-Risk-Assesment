"use strict";

const $ = (id) => document.getElementById(id);
const CIRC = 527.8; // 2 * pi * r(84)

const COLORS = { LOW: "#16a34a", MODERATE: "#d97706", HIGH: "#dc2626" };

const SAFE = {
  age: 32, annual_income: 850000, employment_years: 6, employment_type: "Salaried",
  credit_score: 742, existing_loans: 1, loan_amount: 500000,
  loan_tenure_months: 36, loan_purpose: "Personal", dti_percent: 28,
};
const RISKY = {
  age: 26, annual_income: 300000, employment_years: 1, employment_type: "Self-Employed",
  credit_score: 540, existing_loans: 4, loan_amount: 900000,
  loan_tenure_months: 60, loan_purpose: "Personal", dti_percent: 62,
};

/* ---------- slider <-> number sync ---------- */
function bindPair(rangeEl, numEl, onChange) {
  rangeEl.addEventListener("input", () => { numEl.value = rangeEl.value; onChange(); });
  numEl.addEventListener("input", () => { rangeEl.value = numEl.value; onChange(); });
}
function refreshLabels() {
  $("creditVal").textContent = $("credit_score").value;
  $("dtiVal").textContent = $("dti_percent").value + "%";
}
bindPair($("credit_score_range"), $("credit_score"), refreshLabels);
bindPair($("dti_range"), $("dti_percent"), refreshLabels);
refreshLabels();

/* ---------- samples ---------- */
function fillForm(s) {
  $("age").value = s.age;
  $("annual_income").value = s.annual_income;
  $("employment_years").value = s.employment_years;
  $("employment_type").value = s.employment_type;
  $("credit_score").value = s.credit_score;
  $("credit_score_range").value = s.credit_score;
  $("existing_loans").value = s.existing_loans;
  $("loan_amount").value = s.loan_amount;
  $("loan_tenure_months").value = String(s.loan_tenure_months);
  $("loan_purpose").value = s.loan_purpose;
  $("dti_percent").value = s.dti_percent;
  $("dti_range").value = s.dti_percent;
  refreshLabels();
}
$("sampleSafe").addEventListener("click", () => fillForm(SAFE));
$("sampleRisky").addEventListener("click", () => fillForm(RISKY));

/* ---------- model badge ---------- */
fetch("/api/model-info")
  .then((r) => r.json())
  .then((info) => {
    if (info && info.model) {
      $("modelBadge").textContent =
        `Model: ${info.model} · ROC-AUC ${info.roc_auc} · ${info.trained_on_records.toLocaleString()} records`;
    } else {
      $("modelBadge").textContent = "Model info unavailable";
    }
  })
  .catch(() => { $("modelBadge").textContent = "Model info unavailable"; });

/* ---------- helpers ---------- */
function showError(msg) {
  const box = $("errorBox");
  box.textContent = "⚠ " + msg;
  box.hidden = false;
}
function clearError() { $("errorBox").hidden = true; }

function setLoading(on) {
  $("submitBtn").disabled = on;
  $("btnSpinner").hidden = !on;
  $("btnText").textContent = on ? "Scoring…" : "Assess Risk";
}

function animateNumber(el, to, suffix) {
  const start = performance.now(), dur = 900;
  function frame(now) {
    const t = Math.min((now - start) / dur, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = (to * eased).toFixed(1) + suffix;
    if (t < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

function renderResult(data) {
  $("placeholder").hidden = true;
  $("result").hidden = false;

  const fill = $("gaugeFill");
  fill.style.stroke = COLORS[data.category] || COLORS.MODERATE;
  // force reflow so the transition animates from the previous value
  void fill.getBoundingClientRect();
  fill.style.strokeDashoffset = String(CIRC * (1 - data.risk_probability));

  animateNumber($("riskPct"), data.risk_percent, "%");

  const chip = $("categoryChip");
  chip.textContent = data.category + " RISK";
  chip.className = "category " + data.category;
  $("recommendation").textContent = data.recommendation;

  const list = $("factorList");
  list.innerHTML = "";
  if (data.factors && data.factors.length) {
    for (const f of data.factors) {
      const li = document.createElement("li");
      li.className = f.type;
      const mark = document.createElement("span");
      mark.className = "mark";
      mark.textContent = f.type === "positive" ? "✔" : "✖";
      li.appendChild(mark);
      li.appendChild(document.createTextNode(f.text));
      list.appendChild(li);
    }
  } else {
    const li = document.createElement("li");
    li.className = "none";
    li.textContent = "No standout factors — a balanced profile.";
    list.appendChild(li);
  }

  if (window.innerWidth <= 880) {
    document.querySelector(".result-card").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

/* ---------- submit ---------- */
$("riskForm").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  clearError();

  const payload = {
    age: Number($("age").value),
    annual_income: Number($("annual_income").value),
    employment_years: Number($("employment_years").value),
    employment_type: $("employment_type").value,
    credit_score: Number($("credit_score").value),
    existing_loans: Number($("existing_loans").value),
    loan_amount: Number($("loan_amount").value),
    loan_tenure_months: Number($("loan_tenure_months").value),
    loan_purpose: $("loan_purpose").value,
    debt_to_income_ratio: Number($("dti_percent").value) / 100,
  };

  setLoading(true);
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      showError(data.error || `Server returned ${res.status}. Please check the inputs.`);
      return;
    }
    renderResult(data);
  } catch (err) {
    showError("Could not reach the server. Is the backend running?");
  } finally {
    setLoading(false);
  }
});
