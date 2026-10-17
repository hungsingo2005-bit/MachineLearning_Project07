"""Vẽ Confusion Matrix trên Test độc lập của mô hình đã chọn."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paths import PROJECT07

TABLES_DIR = PROJECT07 / "reports" / "tables"
FIGURES_DIR = PROJECT07 / "reports" / "figures"


def main():
    source = TABLES_DIR / "test_predictions_and_errors.csv"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        raise FileNotFoundError(f"Chưa có {source}; hãy chạy src/PER2/compare_models.py trước.")
    data = pd.read_csv(source)
    matrix = confusion_matrix(data["actual"], data["predicted"], labels=[0, 1])
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["NO (0)", "YES (1)"])
    display.plot(values_format="d", cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix — Test độc lập")
    plt.tight_layout()
    output = FIGURES_DIR / "confusion_matrix_test.png"
    plt.savefig(output, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Đã lưu: {output.resolve()}")


if __name__ == "__main__":
    main()
