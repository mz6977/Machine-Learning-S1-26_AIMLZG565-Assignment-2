"""Interactive Streamlit evaluation dashboard for the assignment classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TEST_DATA_PATH = ROOT / "test_data.csv"

st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="📊",
    layout="wide",
)

st.title("Breast Cancer Classifier")
st.caption("Compare five models on labeled test data.")

with (MODEL_DIR / "metadata.json").open(encoding="utf-8") as metadata_file:
    metadata = json.load(metadata_file)

model_choices = list(metadata["models"])
model_paths = {
    display_name: MODEL_DIR / filename
    for display_name, filename in metadata["models"].items()
}
feature_columns = metadata["feature_columns"]
target_column = metadata["target_column"]


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


@st.cache_data
def load_default_test_data() -> pd.DataFrame:
    return pd.read_csv(TEST_DATA_PATH)


def evaluate_model(estimator, test_frame: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    x_test = test_frame[feature_columns]
    y_true = test_frame[target_column]
    y_pred = estimator.predict(x_test)
    classes = estimator.classes_ if hasattr(estimator, "classes_") else estimator[-1].classes_

    if len(classes) == 2:
        average = "binary"
        auc_scores = estimator.predict_proba(x_test)[:, list(classes).index(classes[1])]
        auc = roc_auc_score(y_true, auc_scores, labels=classes)
        metric_options = {"average": average, "pos_label": classes[1], "zero_division": 0}
    else:
        average = "weighted"
        auc_scores = estimator.predict_proba(x_test)
        auc = roc_auc_score(
            y_true,
            auc_scores,
            labels=classes,
            multi_class="ovr",
            average=average,
        )
        metric_options = {"average": average, "zero_division": 0}

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, **metric_options),
        "Recall": recall_score(y_true, y_pred, **metric_options),
        "F1": f1_score(y_true, y_pred, **metric_options),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    labels = sorted(set(y_true) | set(y_pred))
    matrix = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=[f"Actual {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels],
    )
    report = pd.DataFrame(
        classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    ).transpose()
    return metrics, matrix, report


with st.container():
    setup_column, upload_column = st.columns([1, 2])
    with setup_column:
        selected_model = st.selectbox("Model details", model_choices)
    with upload_column:
        uploaded_file = st.file_uploader(
            "Test CSV (optional)",
            type=["csv"],
            help=(
                f"Include all {len(feature_columns)} numeric features and the "
                f"`{target_column}` target column."
            ),
        )

try:
    if uploaded_file is None:
        test_data = load_default_test_data()
        data_source = "Bundled held-out test set"
    else:
        test_data = pd.read_csv(uploaded_file)
        data_source = f"Uploaded file: {uploaded_file.name}"
except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
    st.error(f"Could not read the test CSV: {error}")
    st.stop()

if test_data.columns.duplicated().any():
    st.error("The CSV contains duplicate column names. Please correct the file and upload again.")
    st.stop()

missing_columns = [column for column in [*feature_columns, target_column] if column not in test_data]
if missing_columns:
    st.error(f"The CSV is missing required columns: {', '.join(missing_columns)}")
    st.stop()
if test_data.empty:
    st.error("The test CSV contains no rows.")
    st.stop()

test_data = test_data.loc[:, [*feature_columns, target_column]].copy()
for column in feature_columns:
    test_data[column] = pd.to_numeric(test_data[column], errors="coerce")
if test_data[feature_columns].isna().any().any():
    st.error("Feature columns must contain numeric values with no missing entries.")
    st.stop()

try:
    y_values = pd.to_numeric(test_data[target_column], errors="raise")
except (TypeError, ValueError):
    st.error(f"The `{target_column}` target column must contain numeric class labels.")
    st.stop()
if y_values.isna().any():
    st.error("The target column contains missing values.")
    st.stop()
test_data[target_column] = y_values

try:
    loaded_models = {
        model_name: load_model(str(model_path))
        for model_name, model_path in model_paths.items()
    }
except (OSError, ValueError, EOFError, ImportError) as error:
    st.error(f"Could not load a saved classifier: {error}")
    st.stop()

valid_classes = set(loaded_models[model_choices[0]].classes_)
unknown_classes = set(test_data[target_column].unique()) - valid_classes
if unknown_classes:
    st.error(
        "The target column contains class labels not present in the trained dataset: "
        f"{sorted(unknown_classes)}"
    )
    st.stop()

if uploaded_file is None:
    st.info(
        f"Showing results for the bundled test data ({len(test_data)} rows). "
        "Upload a different labeled CSV above to evaluate that data instead."
    )
else:
    st.success(
        f"Showing results for {uploaded_file.name} ({len(test_data)} rows). "
        "All metrics below are recalculated for this file; models are not retrained."
    )

results = {}
for model_name, estimator in loaded_models.items():
    try:
        results[model_name] = evaluate_model(estimator, test_data)[0]
    except (ValueError, AttributeError) as error:
        st.error(f"Could not evaluate {model_name}: {error}")
        st.stop()

comparison = pd.DataFrame(results).transpose()
st.subheader("Model comparison")
st.dataframe(
    comparison.style.format("{:.4f}"),
    width="stretch",
)

selected_metrics, selected_matrix, selected_report = evaluate_model(
    loaded_models[selected_model],
    test_data,
)
st.subheader(selected_model)
metric_columns = st.columns(len(selected_metrics))
for column, (metric_name, value) in zip(metric_columns, selected_metrics.items()):
    column.metric(metric_name, f"{value:.4f}")

with st.expander("Confusion matrix and classification report"):
    st.markdown("**Confusion matrix**")
    st.dataframe(selected_matrix, width="stretch")
    st.markdown("**Classification report**")
    st.dataframe(selected_report.style.format("{:.4f}"), width="stretch")
