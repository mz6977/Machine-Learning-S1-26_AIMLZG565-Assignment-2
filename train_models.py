"""Train the five assignment classifiers and create the held-out test CSV."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET_COLUMN = "target"
TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_SPECS = {
    "Logistic Regression": (
        "logistic_regression.joblib",
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]
        ),
    ),
    "Decision Tree": (
        "decision_tree.joblib",
        DecisionTreeClassifier(max_depth=5, min_samples_leaf=2, random_state=RANDOM_STATE),
    ),
    "K-Nearest Neighbors": (
        "knn.joblib",
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
    ),
    "Gaussian Naive Bayes": (
        "naive_bayes.joblib",
        GaussianNB(),
    ),
    "Support Vector Machine": (
        "support_vector_machine.joblib",
        CalibratedClassifierCV(
            estimator=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", SVC(C=1.0, random_state=RANDOM_STATE)),
                ]
            ),
            method="sigmoid",
            cv=5,
            ensemble=False,
        ),
    ),
}


def main() -> None:
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    target = dataset.target.rename(TARGET_COLUMN)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    MODEL_DIR.mkdir(exist_ok=True)

    for model_name, (filename, estimator) in MODEL_SPECS.items():
        estimator.fit(x_train, y_train)
        joblib.dump(estimator, MODEL_DIR / filename, compress=3)
        print(f"Trained and saved {model_name}: {filename}")

    test_data = x_test.copy()
    test_data[TARGET_COLUMN] = y_test
    test_data.to_csv(ROOT / "test_data.csv", index=False)
    metadata = {
        "dataset": "Wisconsin Diagnostic Breast Cancer",
        "source": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        "target_column": TARGET_COLUMN,
        "target_labels": {"0": "malignant", "1": "benign"},
        "feature_columns": list(features.columns),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "training_rows": len(x_train),
        "test_rows": len(x_test),
        "models": {name: filename for name, (filename, _) in MODEL_SPECS.items()},
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Dataset rows: {len(features)}; features: {features.shape[1]}")
    print(f"Training rows: {len(x_train)}; held-out test rows: {len(x_test)}")
    print(f"Target counts in test data:\n{y_test.value_counts().sort_index().to_string()}")


if __name__ == "__main__":
    main()
