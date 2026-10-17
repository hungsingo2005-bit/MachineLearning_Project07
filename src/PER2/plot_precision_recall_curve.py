"""Vẽ Precision–Recall Curve cho các mô hình trên Validation."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paths import PROJECT07

TABLES_DIR = PROJECT07 / "reports" / "tables"
FIGURES_DIR = PROJECT07 / "reports" / "figures"


def main():
    source = TABLES_DIR / "validation_predictions.csv"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        raise FileNotFoundError(f"Chưa có {source}; hãy chạy src/PER2/compare_models.py trước.")
    data = pd.read_csv(source)
    if data.empty:
        raise ValueError(f"File không có dữ liệu: {source}")
    for model_name, group in data.groupby("model"):
        precision, recall, _ = precision_recall_curve(group["actual"], group["probability_yes"])
        average_precision = average_precision_score(group["actual"], group["probability_yes"])
        plt.plot(recall, precision, linewidth=2, label=f"{model_name} (AP={average_precision:.3f})")
    # Tỷ lệ lớp dương trên Validation là baseline precision của mô hình không phân biệt.
    positive_rate = data[data["model"] == data["model"].iloc[0]]["actual"].mean()
    plt.axhline(positive_rate, linestyle="--", color="gray", label=f"Tỷ lệ lớp dương = {positive_rate:.3f}")
    plt.title("Precision–Recall Curve — Validation")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output = FIGURES_DIR / "precision_recall_curve_validation_models.png"
    plt.savefig(output, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Đã lưu: {output.resolve()}")


if __name__ == "__main__":
    main()
