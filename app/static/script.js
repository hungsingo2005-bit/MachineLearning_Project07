const form = document.getElementById("prediction-form");
const result = document.getElementById("result");

const numberFields = ["age", "balance", "day", "campaign", "pdays", "previous"];
const categoricalFields = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"];

function showLoading() {
  result.innerHTML = `
    <div class="result-empty" role="status">
      <div class="result-empty-icon loading-icon">◌</div>
      <h4>Đang phân tích...</h4>
      <p>Mô hình đang xử lý thông tin khách hàng.</p>
    </div>`;
}

function showError(message) {
  result.innerHTML = "";
  const box = document.createElement("div");
  box.className = "error-box";
  box.setAttribute("role", "alert");
  box.textContent = message;
  result.appendChild(box);
}

function renderPrediction(data) {
  const probability = Math.max(0, Math.min(1, Number(data.probability) || 0));
  const percent = (probability * 100).toFixed(2);
  const thresholdPercent = (Number(data.threshold) * 100).toFixed(1);
  const isPositive = Number(data.prediction) === 1;
  const decisionText = data.decision_vi || (isPositive ? "Có khả năng đăng ký" : "Chưa được xếp vào nhóm có khả năng đăng ký");

  result.innerHTML = `
    <div class="result-box">
      <span class="result-label">XÁC SUẤT DỰ ĐOÁN ĐĂNG KÝ</span>
      <div class="result-probability">${percent}%</div>
      <div class="probability-track" aria-label="Xác suất ${percent}%"><div class="probability-fill" style="width:${probability * 100}%"></div></div>
      <div class="result-scale"><span>0%</span><span>100%</span></div>
      <div class="result-threshold">Ngưỡng quyết định: <strong>${thresholdPercent}%</strong></div>
      <div class="result-decision ${isPositive ? "yes" : "no"}">${decisionText}</div>
      <p class="result-model">Mô hình: ${escapeHtml(String(data.model || "PER2 selected model"))}</p>
    </div>`;
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[character]);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!form.reportValidity()) return;

  const data = {};
  for (const field of numberFields) {
    const input = document.getElementById(field);
    data[field] = Number(input.value);
    if (input.value.trim() === "" || !Number.isFinite(data[field])) {
      showError(`Vui lòng nhập giá trị số hợp lệ cho trường “${input.labels?.[0]?.textContent || field}”.`);
      input.focus();
      return;
    }
  }
  for (const field of categoricalFields) {
    data[field] = document.getElementById(field).value;
  }

  showLoading();
  const submitButton = form.querySelector("button[type='submit']");
  submitButton.disabled = true;
  submitButton.style.opacity = "0.7";

  try {
    const response = await fetch("/api/score", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify(data)
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const missing = Array.isArray(payload.missing_fields) ? ` Thiếu: ${payload.missing_fields.join(", ")}.` : "";
      throw new Error((payload.error || "Không thể thực hiện dự đoán.") + missing);
    }
    if (typeof payload.probability !== "number" || typeof payload.threshold !== "number") {
      throw new Error("Phản hồi API không có xác suất hoặc ngưỡng hợp lệ.");
    }
    renderPrediction(payload);
  } catch (error) {
    showError(error instanceof Error ? error.message : "Đã xảy ra lỗi không xác định.");
  } finally {
    submitButton.disabled = false;
    submitButton.style.opacity = "1";
  }
});
