"""Vẽ Precision, Recall và F1 theo threshold từ kết quả Validation."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paths import PROJECT07

REPORTS_DIR = PROJECT07 / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"


def main():
    source = TABLES_DIR / "threshold_results_validation.csv"
    report_path = REPORTS_DIR / "final_evaluation.json"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        raise FileNotFoundError(f"Chưa có {source}; hãy chạy src/PER2/compare_models.py trước.")
    results = pd.read_csv(source)
    selected_model = "Logistic Regression"
    selected_threshold = None
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        selected_model = report.get("selected_model", selected_model)
        selected_threshold = report.get("selected_threshold")
    data = results[results["model"] == selected_model].sort_values("threshold")
    if data.empty:
        raise ValueError(f"Không tìm thấy mô hình {selected_model!r} trong {source}")
    for metric in ["precision", "recall", "f1"]:
        plt.plot(data["threshold"], data[metric], label=metric.upper(), linewidth=2)
    if selected_threshold is not None:
        plt.axvline(selected_threshold, linestyle="--", label=f"Threshold đã chọn: {selected_threshold:.2f}")
    plt.title(f"Precision, Recall, F1 theo threshold — Validation\n{selected_model}")
    plt.xlabel("Threshold")
    plt.ylabel("Điểm số")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output = FIGURES_DIR / "threshold_metrics_validation.png"
    plt.savefig(output, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Đã lưu: {output.resolve()}")


if __name__ == "__main__":
    main()
