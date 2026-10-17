from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

app = Flask(__name__)

# Locate the project root whether app.py lives in the root or one directory below it.
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(
    (parent for parent in _THIS_FILE.parents if (parent / "paths.py").exists()),
    _THIS_FILE.parent,
)
MODEL_PATH = PROJECT_ROOT / "models" / "project07_selected_model.joblib"
EVALUATION_PATH = PROJECT_ROOT / "reports" / "final_evaluation.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy mô hình PER2: {MODEL_PATH}. "
        "Hãy chạy src/PER2/compare_models.py trước."
    )
if not EVALUATION_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy báo cáo PER2: {EVALUATION_PATH}. "
        "Hãy chạy src/PER2/compare_models.py trước."
    )

model = joblib.load(MODEL_PATH)
with EVALUATION_PATH.open("r", encoding="utf-8") as file:
    evaluation: dict[str, Any] = json.load(file)

# PER2 stores the selected threshold at the report's top level.
THRESHOLD = float(evaluation["selected_threshold"])
TEST_METRICS = evaluation.get("test_metrics", {})
CONFUSION_MATRIX = evaluation.get("test_confusion_matrix", {})
SELECTED_MODEL = evaluation.get("selected_model", "Mô hình đã chọn")

# Các biểu đồ được tạo bởi PER1/PER2 trong reports/figures.
# Chỉ hiển thị biểu đồ thực sự tồn tại để trang web không bị ảnh lỗi.
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURE_CATALOG = [
    {"file": "threshold_metrics_validation.png", "title": "Thử nghiệm ngưỡng dự đoán", "description": "Precision, Recall và F1 theo threshold trên Validation.", "group": "Thí nghiệm mô hình"},
    {"file": "baseline_model_comparison_validation.png", "title": "So sánh mô hình và baseline", "description": "So sánh Logistic Regression, Decision Tree và các baseline trên Validation.", "group": "Thí nghiệm mô hình"},
    {"file": "precision_recall_curve_validation_models.png", "title": "Đường Precision–Recall", "description": "Đánh giá khả năng phát hiện lớp đăng ký trên Validation.", "group": "Thí nghiệm mô hình"},
    {"file": "confusion_matrix_test.png", "title": "Confusion Matrix trên Test", "description": "Số lượng TN, FP, FN và TP trên tập Test độc lập.", "group": "Kết quả trên Test"},
    {"file": "error_analysis_age_test.png", "title": "Phân tích lỗi theo độ tuổi", "description": "Phân bố lỗi dự đoán theo nhóm tuổi trên Test.", "group": "Kết quả trên Test"},
    {"file": "error_analysis_job_test.png", "title": "Phân tích lỗi theo nghề nghiệp", "description": "Phân bố lỗi dự đoán theo nghề nghiệp trên Test.", "group": "Kết quả trên Test"},
    {"file": "eda_train_target_distribution.png", "title": "Phân bố nhãn trên Train", "description": "Tỷ lệ khách hàng đăng ký và không đăng ký trong tập Train.", "group": "Khám phá dữ liệu (Train)"},
    {"file": "eda_train_age_distribution.png", "title": "Phân bố độ tuổi", "description": "Phân phối độ tuổi của khách hàng trong tập Train.", "group": "Khám phá dữ liệu (Train)"},
    {"file": "eda_train_balance_distribution.png", "title": "Phân bố số dư tài khoản", "description": "Phân phối số dư tài khoản trong tập Train.", "group": "Khám phá dữ liệu (Train)"},
    {"file": "eda_train_subscription_by_age.png", "title": "Đăng ký theo nhóm tuổi", "description": "So sánh kết quả đăng ký giữa các nhóm tuổi trong Train.", "group": "Khám phá dữ liệu (Train)"},
    {"file": "eda_train_subscription_by_job.png", "title": "Đăng ký theo nghề nghiệp", "description": "So sánh kết quả đăng ký theo nghề nghiệp trong Train.", "group": "Khám phá dữ liệu (Train)"},
]

REQUIRED_FIELDS = [
    "age", "job", "marital", "education", "default", "balance",
    "housing", "loan", "contact", "day", "month", "campaign",
    "pdays", "previous", "poutcome",
]
NUMERIC_FIELDS = ["age", "balance", "day", "campaign", "pdays", "previous"]
INTEGER_FIELDS = ["age", "day", "campaign", "pdays", "previous"]
VALID_VALUES = {
    "job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"],
    "marital": ["divorced", "married", "single"],
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "default": ["no", "yes"],
    "housing": ["no", "yes"],
    "loan": ["no", "yes"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "poutcome": ["failure", "other", "success", "unknown"],
}


def metric_value(name: str, digits: int = 3) -> str:
    value = TEST_METRICS.get(name)
    try:
        return f"{float(value):.{digits}f}" if value is not None else "—"
    except (TypeError, ValueError):
        return "—"


@app.route("/")
def home():
    return render_template(
        "index.html",
        threshold=f"{THRESHOLD:.2f}",
        evaluation=TEST_METRICS,
        metrics={
            "pr_auc": metric_value("pr_auc"),
            "accuracy": metric_value("accuracy"),
            "precision": metric_value("precision"),
            "recall": metric_value("recall"),
            "f1": metric_value("f1"),
            "threshold": f"{THRESHOLD:.2f}",
        },
        confusion=CONFUSION_MATRIX,
        selected_model=SELECTED_MODEL,
        figures=[item for item in FIGURE_CATALOG if (FIGURES_DIR / item["file"]).is_file()],
    )


@app.route("/reports/figures/<path:filename>")
def report_figure(filename: str):
    """Phục vụ biểu đồ đánh giá từ thư mục reports/figures."""
    return send_from_directory(FIGURES_DIR, filename)


@app.route("/api/score", methods=["POST"])
def score():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Nội dung yêu cầu phải là JSON hợp lệ."}), 400

    missing = [field for field in REQUIRED_FIELDS if field not in data]
    extra = [field for field in data if field not in REQUIRED_FIELDS]
    if missing:
        return jsonify({"error": "Thiếu trường dữ liệu bắt buộc.", "missing_fields": missing}), 400
    if extra:
        return jsonify({"error": "Có trường dữ liệu không được hỗ trợ.", "extra_fields": extra}), 400

    for field in NUMERIC_FIELDS:
        value = data[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return jsonify({"error": f"Trường '{field}' phải là số."}), 400
        if field in INTEGER_FIELDS and int(value) != value:
            return jsonify({"error": f"Trường '{field}' phải là số nguyên."}), 400

    for field, allowed in VALID_VALUES.items():
        if not isinstance(data[field], str) or data[field] not in allowed:
            return jsonify({"error": f"Giá trị của '{field}' không hợp lệ.", "allowed_values": allowed}), 400

    if not 18 <= data["age"] <= 100:
        return jsonify({"error": "Tuổi phải nằm trong khoảng 18–100."}), 400
    if not 1 <= data["day"] <= 31:
        return jsonify({"error": "Ngày liên hệ phải nằm trong khoảng 1–31."}), 400
    if data["campaign"] < 1:
        return jsonify({"error": "Số lần liên hệ hiện tại phải lớn hơn hoặc bằng 1."}), 400
    if data["pdays"] < -1:
        return jsonify({"error": "pdays phải lớn hơn hoặc bằng -1."}), 400
    if data["previous"] < 0:
        return jsonify({"error": "Số lần liên hệ trước không được âm."}), 400

    # Use a stable feature order and pass raw fields to the saved sklearn Pipeline.
    features = pd.DataFrame([{field: data[field] for field in REQUIRED_FIELDS}], columns=REQUIRED_FIELDS)
    try:
        probability = float(model.predict_proba(features)[0][1])
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Không thể dự đoán với dữ liệu này.", "detail": str(exc)}), 500

    prediction = int(probability >= THRESHOLD)
    return jsonify({
        "probability": round(probability, 6),
        "threshold": THRESHOLD,
        "prediction": prediction,
        "decision": "yes" if prediction else "no",
        "decision_vi": "Có khả năng đăng ký" if prediction else "Chưa được xếp vào nhóm có khả năng đăng ký",
        "model": SELECTED_MODEL,
    })


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model": SELECTED_MODEL, "threshold": THRESHOLD})


if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Model: {MODEL_PATH.name}")
    print(f"Selected model: {SELECTED_MODEL}")
    print(f"Threshold: {THRESHOLD:.2f}")
    app.run(debug=True)
