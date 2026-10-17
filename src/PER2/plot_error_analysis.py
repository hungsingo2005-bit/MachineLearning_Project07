"""Vẽ số FP/FN theo nhóm tuổi và nghề nghiệp trên Test độc lập."""
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


def age_group(age):
    if pd.isna(age):
        return "Unknown"
    if age <= 30:
        return "18–30"
    if age <= 40:
        return "31–40"
    if age <= 50:
        return "41–50"
    if age <= 60:
        return "51–60"
    return "61+"


def main():
    source = TABLES_DIR / "test_predictions_and_errors.csv"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        raise FileNotFoundError(f"Chưa có {source}; hãy chạy src/PER2/compare_models.py trước.")
    data = pd.read_csv(source)

    if "age" in data.columns:
        data["age_group"] = data["age"].apply(age_group)
        age = data.groupby("age_group", dropna=False).agg(
            samples=("actual", "size"),
            FP=("error_type", lambda values: (values == "FP").sum()),
            FN=("error_type", lambda values: (values == "FN").sum()),
            TP=("error_type", lambda values: (values == "TP").sum()),
            TN=("error_type", lambda values: (values == "TN").sum()),
        ).reindex(["18–30", "31–40", "41–50", "51–60", "61+", "Unknown"]).dropna(how="all")
        age.to_csv(TABLES_DIR / "error_analysis_age_test.csv", index_label="age_group")
        ax = age[["FP", "FN"]].plot(kind="bar", figsize=(8, 5), width=0.8)
        ax.set_title("FP và FN theo nhóm tuổi — Test")
        ax.set_xlabel("Nhóm tuổi")
        ax.set_ylabel("Số trường hợp")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        output = FIGURES_DIR / "error_analysis_age_test.png"
        plt.savefig(output, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"Đã lưu: {output.resolve()}")

    if "job" in data.columns:
        job = data.groupby("job", dropna=False).agg(
            samples=("actual", "size"),
            FP=("error_type", lambda values: (values == "FP").sum()),
            FN=("error_type", lambda values: (values == "FN").sum()),
            TP=("error_type", lambda values: (values == "TP").sum()),
            TN=("error_type", lambda values: (values == "TN").sum()),
        ).sort_values("samples", ascending=False)
        job.to_csv(TABLES_DIR / "error_analysis_job_test.csv", index_label="job")
        ax = job[["FP", "FN"]].plot(kind="bar", figsize=(10, 6), width=0.8)
        ax.set_title("FP và FN theo nghề nghiệp — Test")
        ax.set_xlabel("Nghề nghiệp")
        ax.set_ylabel("Số trường hợp")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        output = FIGURES_DIR / "error_analysis_job_test.png"
        plt.savefig(output, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"Đã lưu: {output.resolve()}")


if __name__ == "__main__":
    main()
