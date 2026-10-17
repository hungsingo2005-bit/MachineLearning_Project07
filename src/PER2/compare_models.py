"""Huấn luyện, so sánh baseline và mô hình; chỉ dùng Test sau khi chốt trên Validation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from paths import DATA_FILE, PROJECT07

REPORTS_DIR = PROJECT07 / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
MODELS_DIR = PROJECT07 / "models"
SEED = 42
THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.01), 2)


def make_pipeline(X_train: pd.DataFrame, classifier):
    """Tạo preprocessing và model trong cùng Pipeline để tránh leakage."""
    numeric = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in X_train.columns if column not in numeric]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers = []
    if numeric:
        transformers.append(("num", numeric_pipeline, numeric))
    if categorical:
        transformers.append(("cat", categorical_pipeline, categorical))
    preprocessor = ColumnTransformer(transformers=transformers)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def metric_row(y_true, y_pred, y_proba=None):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_proba)) if y_proba is not None else float("nan"),
    }


def encode_target(series: pd.Series) -> pd.Series:
    mapped = series.map({"no": 0, "yes": 1, 0: 0, 1: 1})
    if mapped.isna().any():
        invalid = series[mapped.isna()].unique().tolist()
        raise ValueError(f"Target y có giá trị không hợp lệ: {invalid}")
    return mapped.astype(int)


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {DATA_FILE}")

    # Bank Marketing CSV dùng dấu ; để phân tách cột.
    df = pd.read_csv(DATA_FILE, sep=";")
    if "y" not in df.columns:
        raise ValueError("Dataset cần có cột target 'y'.")

    # duration chỉ có sau/trong cuộc gọi, nên loại khỏi feature.
    df = df.drop(columns=["duration"], errors="ignore")
    df = df.dropna(subset=["y"]).copy()
    y = encode_target(df["y"])
    X = df.drop(columns=["y"])

    # 70% train, 15% validation, 15% test; giữ tỷ lệ lớp bằng stratify.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Decision Tree (depth=5)": DecisionTreeClassifier(max_depth=5, random_state=SEED),
    }
    fitted_models = {}
    threshold_rows = []
    comparison_rows = []
    validation_predictions = []
    best_thresholds = {}

    for model_name, classifier in classifiers.items():
        pipeline = make_pipeline(X_train, classifier)
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_val)[:, 1]
        fitted_models[model_name] = pipeline

        model_rows = []
        for threshold in THRESHOLDS:
            predictions = (probabilities >= threshold).astype(int)
            row = {
                "model": model_name,
                "threshold": float(threshold),
                **metric_row(y_val, predictions, probabilities),
            }
            threshold_rows.append(row)
            model_rows.append(row)

        # Chọn threshold chỉ trên Validation: F1 cao nhất; tie-break Recall rồi threshold thấp hơn.
        best = max(model_rows, key=lambda row: (row["f1"], row["recall"], -row["threshold"]))
        best_thresholds[model_name] = float(best["threshold"])
        best_predictions = (probabilities >= best["threshold"]).astype(int)
        comparison_rows.append({
            "model": model_name,
            "selected_threshold": float(best["threshold"]),
            **metric_row(y_val, best_predictions, probabilities),
        })

        for actual, probability, prediction in zip(y_val.to_numpy(), probabilities, best_predictions):
            validation_predictions.append({
                "model": model_name,
                "actual": int(actual),
                "probability_yes": float(probability),
                "selected_threshold": float(best["threshold"]),
                "predicted": int(prediction),
            })

    # Baseline 1: dự đoán tất cả là NO.
    all_negative = np.zeros(len(y_val), dtype=int)
    comparison_rows.append({
        "model": "Baseline — all negative",
        "selected_threshold": np.nan,
        **metric_row(y_val, all_negative),
    })

    # Baseline 2: chọn ngẫu nhiên cùng số lượng liên hệ như Logistic Regression trên Validation.
    lr_probabilities = fitted_models["Logistic Regression"].predict_proba(X_val)[:, 1]
    lr_predictions = (lr_probabilities >= best_thresholds["Logistic Regression"]).astype(int)
    contact_count = int(lr_predictions.sum())
    rng = np.random.default_rng(SEED)
    random_predictions = np.zeros(len(y_val), dtype=int)
    if contact_count > 0:
        selected_indices = rng.choice(len(y_val), size=min(contact_count, len(y_val)), replace=False)
        random_predictions[selected_indices] = 1
    comparison_rows.append({
        "model": "Baseline — random contact",
        "selected_threshold": np.nan,
        **metric_row(y_val, random_predictions),
    })

    pd.DataFrame(threshold_rows).to_csv(TABLES_DIR / "threshold_results_validation.csv", index=False)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(TABLES_DIR / "model_baseline_comparison_validation.csv", index=False)
    pd.DataFrame(validation_predictions).to_csv(TABLES_DIR / "validation_predictions.csv", index=False)

    # Chọn model theo F1 Validation; nếu bằng nhau, dùng PR-AUC Validation.
    ml_comparison = comparison[comparison["model"].isin(best_thresholds.keys())].set_index("model")
    selected_name = max(
        best_thresholds,
        key=lambda name: (ml_comparison.loc[name, "f1"], ml_comparison.loc[name, "pr_auc"]),
    )
    selected_threshold = best_thresholds[selected_name]
    selected_model = fitted_models[selected_name]

    # Test chỉ được dùng sau khi model và threshold đã chốt.
    test_probabilities = selected_model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= selected_threshold).astype(int)
    test_metrics = metric_row(y_test, test_predictions, test_probabilities)

    model_path = MODELS_DIR / "project07_selected_model.joblib"
    joblib.dump(selected_model, model_path)

    test_output = X_test.reset_index(drop=True).copy()
    test_output["actual"] = y_test.reset_index(drop=True).astype(int)
    test_output["probability_yes"] = test_probabilities
    test_output["predicted"] = test_predictions
    test_output["error_type"] = np.select(
        [
            (test_output["actual"] == 1) & (test_output["predicted"] == 1),
            (test_output["actual"] == 0) & (test_output["predicted"] == 0),
            (test_output["actual"] == 0) & (test_output["predicted"] == 1),
            (test_output["actual"] == 1) & (test_output["predicted"] == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="UNKNOWN",
    )
    test_output.to_csv(TABLES_DIR / "test_predictions_and_errors.csv", index=False)

    cm = {
        "TN": int(((y_test.to_numpy() == 0) & (test_predictions == 0)).sum()),
        "FP": int(((y_test.to_numpy() == 0) & (test_predictions == 1)).sum()),
        "FN": int(((y_test.to_numpy() == 1) & (test_predictions == 0)).sum()),
        "TP": int(((y_test.to_numpy() == 1) & (test_predictions == 1)).sum()),
    }
    report = {
        "selected_model": selected_name,
        "selected_threshold": selected_threshold,
        "selection_rule": "Chọn F1 cao nhất trên Validation; nếu hòa, dùng PR-AUC Validation. Test chỉ đánh giá sau khi chốt lựa chọn.",
        "validation_model_comparison": comparison_rows,
        "test_metrics": test_metrics,
        "test_confusion_matrix": cm,
        "seed": SEED,
        "dataset_path": str(DATA_FILE),
        "duration_removed": True,
        "split_rows": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
    }
    (REPORTS_DIR / "final_evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nSO SÁNH TRÊN VALIDATION")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nMô hình được chọn: {selected_name}; threshold={selected_threshold:.2f}")
    print("KẾT QUẢ TEST CUỐI CÙNG")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}" if pd.notna(value) else f"{metric}: N/A")
    print(f"\nBảng kết quả: {TABLES_DIR.resolve()}")
    print(f"Báo cáo cuối: {(REPORTS_DIR / 'final_evaluation.json').resolve()}")
    print(f"Mô hình đã lưu: {model_path.resolve()}")


if __name__ == "__main__":
    main()
