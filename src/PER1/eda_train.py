
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


# ==================================================
# IMPORT PATHS.PY
# ==================================================

# Lấy thư mục gốc dự án:
# eda_train.py nằm trong src/PER1/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Đảm bảo Python tìm được paths.py ở thư mục gốc
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paths import DATA_FILE, PROJECT07


# ==================================================
# CONFIGURATION
# ==================================================

SEED = 42

# Đường dẫn dữ liệu được quản lý tập trung trong paths.py
DATA_PATH = DATA_FILE

# Các thư mục lưu kết quả
REPORTS_DIR = PROJECT07 / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def age_group(age):
    """Phân nhóm tuổi để phục vụ phân tích dữ liệu."""

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


def save_fig(path: Path):
    """Lưu biểu đồ với độ phân giải cao."""

    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


# ==================================================
# MAIN
# ==================================================

def main():

    # --------------------------------------------------
    # 1. KIỂM TRA ĐƯỜNG DẪN VÀ TẠO THƯ MỤC
    # --------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu: {DATA_PATH}"
        )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PROJECT 07 - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print(f"Dataset: {DATA_PATH}")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Tables:  {TABLES_DIR}")

    # --------------------------------------------------
    # 2. ĐỌC VÀ KIỂM TRA DỮ LIỆU
    # --------------------------------------------------

    # Bank Marketing sử dụng dấu chấm phẩy để phân tách cột
    df = pd.read_csv(DATA_PATH, sep=";")

    print("\nKích thước dữ liệu ban đầu:", df.shape)

    print("\nDanh sách cột:")
    print(df.columns.tolist())

    # Kiểm tra các cột bắt buộc
    required_columns = {"y", "age", "balance"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset thiếu các cột bắt buộc: {missing_columns}"
        )

    # --------------------------------------------------
    # 3. LOẠI DATA LEAKAGE VÀ XỬ LÝ TARGET
    # --------------------------------------------------

    # duration chỉ biết được sau hoặc trong cuộc gọi.
    # Mục tiêu là dự đoán trước cuộc gọi nên loại duration.
    duration_removed = "duration" in df.columns

    if duration_removed:
        df = df.drop(columns=["duration"])

    # Loại các dòng không có nhãn mục tiêu
    df = df.dropna(subset=["y"]).copy()

    # Chuyển target:
    # no  -> 0
    # yes -> 1
    target_mapping = {
        "no": 0,
        "yes": 1,
        0: 0,
        1: 1,
    }

    y = df["y"].map(target_mapping)

    # Không âm thầm chuyển đổi nhãn không hợp lệ
    if y.isna().any():
        invalid_values = df.loc[y.isna(), "y"].unique().tolist()

        raise ValueError(
            f"Target chứa giá trị không hợp lệ: {invalid_values}"
        )

    y = y.astype(int)

    # X chỉ chứa các đặc trưng đầu vào
    X = df.drop(columns=["y"])

    print("\nKích thước sau xử lý:", df.shape)
    print("Số lượng features:", X.shape[1])
    print("Đã loại duration:", duration_removed)

    # --------------------------------------------------
    # 4. CHIA TRAIN / VALIDATION / TEST
    # --------------------------------------------------

    # 70% Train, 30% phần còn lại
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=SEED,
        stratify=y,
    )

    # Chia 30% còn lại thành 15% Validation và 15% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=SEED,
        stratify=y_temp,
    )

    print("\nKích thước các tập dữ liệu:")
    print(f"Train:      {len(X_train):,} mẫu")
    print(f"Validation: {len(X_val):,} mẫu")
    print(f"Test:       {len(X_test):,} mẫu")

    # EDA phục vụ quyết định tiền xử lý chỉ sử dụng Train
    train = X_train.copy()

    # Gắn target đã mã hóa để thống kê
    train["y"] = y_train.map({
        0: "no",
        1: "yes",
    })

    # --------------------------------------------------
    # 5. BIỂU ĐỒ PHÂN BỐ TARGET
    # --------------------------------------------------

    counts = train["y"].value_counts().reindex(
        ["no", "yes"],
        fill_value=0,
    )

    ax = counts.plot(
        kind="bar",
        figsize=(7, 5),
        color=["#2563EB", "#F59E0B"],
    )

    ax.set_title("Phân bố target - tập Train")
    ax.set_xlabel("Đăng ký tiền gửi")
    ax.set_ylabel("Số lượng mẫu")
    ax.tick_params(axis="x", rotation=0)

    save_fig(
        FIGURES_DIR / "eda_train_target_distribution.png"
    )

    # --------------------------------------------------
    # 6. BIỂU ĐỒ PHÂN BỐ ĐỘ TUỔI
    # --------------------------------------------------

    ax = train["age"].plot(
        kind="hist",
        bins=30,
        figsize=(7, 5),
        color="#2563EB",
        edgecolor="white",
    )

    ax.set_title("Phân bố độ tuổi - tập Train")
    ax.set_xlabel("Tuổi")
    ax.set_ylabel("Số lượng mẫu")

    save_fig(
        FIGURES_DIR / "eda_train_age_distribution.png"
    )

    # --------------------------------------------------
    # 7. BIỂU ĐỒ PHÂN BỐ SỐ DƯ
    # --------------------------------------------------

    ax = train["balance"].plot(
        kind="hist",
        bins=50,
        figsize=(7, 5),
        color="#0D9488",
        edgecolor="white",
    )

    ax.set_title("Phân bố số dư tài khoản - tập Train")
    ax.set_xlabel("Số dư tài khoản")
    ax.set_ylabel("Số lượng mẫu")

    save_fig(
        FIGURES_DIR / "eda_train_balance_distribution.png"
    )

    # --------------------------------------------------
    # 8. TỶ LỆ ĐĂNG KÝ THEO NHÓM TUỔI
    # --------------------------------------------------

    train["age_group"] = train["age"].apply(age_group)

    age_order = [
        "18–30",
        "31–40",
        "41–50",
        "51–60",
        "61+",
        "Unknown",
    ]

    age_rate = (
        train.groupby("age_group", observed=False)["y"]
        .apply(lambda s: (s == "yes").mean() * 100)
        .reindex(age_order)
        .dropna()
    )

    ax = age_rate.plot(
        kind="bar",
        figsize=(8, 5),
        color="#2563EB",
    )

    ax.set_title("Tỷ lệ đăng ký theo nhóm tuổi - tập Train")
    ax.set_xlabel("Nhóm tuổi")
    ax.set_ylabel("Tỷ lệ đăng ký (%)")
    ax.tick_params(axis="x", rotation=0)

    save_fig(
        FIGURES_DIR / "eda_train_subscription_by_age.png"
    )

    age_rate.rename(
        "subscription_rate_percent"
    ).to_csv(
        TABLES_DIR / "eda_train_subscription_by_age.csv",
        encoding="utf-8-sig",
    )

    # --------------------------------------------------
    # 9. TỶ LỆ ĐĂNG KÝ THEO NGHỀ NGHIỆP
    # --------------------------------------------------

    if "job" in train.columns:

        job_rate = (
            train.groupby("job", observed=False)["y"]
            .apply(lambda s: (s == "yes").mean() * 100)
            .sort_values()
        )

        ax = job_rate.plot(
            kind="barh",
            figsize=(9, 6),
            color="#0D9488",
        )

        ax.set_title(
            "Tỷ lệ đăng ký theo nghề nghiệp - tập Train"
        )
        ax.set_xlabel("Tỷ lệ đăng ký (%)")
        ax.set_ylabel("Nghề nghiệp")

        save_fig(
            FIGURES_DIR / "eda_train_subscription_by_job.png"
        )

        job_rate.rename(
            "subscription_rate_percent"
        ).to_csv(
            TABLES_DIR / "eda_train_subscription_by_job.csv",
            encoding="utf-8-sig",
        )

    # --------------------------------------------------
    # 10. TỔNG HỢP CHẤT LƯỢNG DỮ LIỆU
    # --------------------------------------------------

    summary = {
        "dataset_path": str(DATA_PATH),
        "total_rows": int(len(df)),
        "total_features": int(X.shape[1]),
        "duration_removed": duration_removed,
        "random_state": SEED,

        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),

        "train_positive_rate": float(y_train.mean()),
        "validation_positive_rate": float(y_val.mean()),
        "test_positive_rate": float(y_test.mean()),

        "missing_values": {
            column: int(count)
            for column, count in df.isna().sum().items()
        },

        "duplicate_rows": int(df.duplicated().sum()),
    }

    summary_path = (
        TABLES_DIR / "eda_train_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # 11. THÔNG BÁO KẾT QUẢ
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("HOÀN THÀNH EDA")
    print("=" * 60)

    print(f"Biểu đồ: {FIGURES_DIR.resolve()}")
    print(f"Bảng kết quả: {TABLES_DIR.resolve()}")
    print(f"Summary: {summary_path.resolve()}")

    print(
        f"\nTỷ lệ YES trên Train: "
        f"{y_train.mean() * 100:.2f}%"
    )

    print("\nĐã tạo các biểu đồ:")
    print("- eda_train_target_distribution.png")
    print("- eda_train_age_distribution.png")
    print("- eda_train_balance_distribution.png")
    print("- eda_train_subscription_by_age.png")
    print("- eda_train_subscription_by_job.png")


if __name__ == "__main__":
    main()
