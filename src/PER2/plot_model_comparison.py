"""Vẽ biểu đồ so sánh mô hình và baseline trên Validation."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paths import PROJECT07

TABLES_DIR = PROJECT07 / "reports" / "tables"
FIGURES_DIR = PROJECT07 / "reports" / "figures"


def main():
    source = TABLES_DIR / "model_baseline_comparison_validation.csv"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        raise FileNotFoundError(f"Chưa có {source}; hãy chạy src/PER2/compare_models.py trước.")
    data = pd.read_csv(source).set_index("model")
    metrics = ["precision", "recall", "f1"]
    ax = data[metrics].plot(kind="bar", figsize=(11, 6), width=0.8)
    ax.set_title("So sánh mô hình và baseline — Validation")
    ax.set_xlabel("Phương pháp")
    ax.set_ylabel("Điểm số")
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="x", rotation=18)
    ax.legend(["Precision", "Recall", "F1"])
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    output = FIGURES_DIR / "baseline_model_comparison_validation.png"
    plt.savefig(output, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Đã lưu: {output.resolve()}")


if __name__ == "__main__":
    main()
