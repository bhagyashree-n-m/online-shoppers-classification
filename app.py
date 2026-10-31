import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Application configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Online Shopping Purchase Prediction",
    page_icon="🛍️",
    layout="wide"
)

st.title("Online Shoppers Purchase Intention Prediction 🛍️")

st.write(
    "Compare five machine learning classification models "
    "for predicting whether an online shopping session "
    "results in a purchase."
)

st.info(
    "Upload a small test CSV containing the input features "
    "and the actual Revenue column to evaluate a model."
)

MODEL_DIR = Path(__file__).parent / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_nb.joblib",
    "Support Vector Machine": "svm.joblib"
}


# --------------------------------------------------
# Load saved model
# --------------------------------------------------

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


# --------------------------------------------------
# Dataset upload
# --------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

if uploaded_file is None:
    st.warning(
        "Please upload test_data.csv to evaluate the models."
    )
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Could not read the CSV file: {error}")
    st.stop()

if data.empty:
    st.error("The uploaded CSV is empty.")
    st.stop()

if "Revenue" not in data.columns:
    st.error(
        "The uploaded CSV must contain the actual Revenue column "
        "to calculate evaluation metrics."
    )
    st.stop()

if data["Revenue"].isnull().any():
    st.error("The Revenue column contains missing values.")
    st.stop()

# Support the boolean labels used by the original dataset.
if pd.api.types.is_bool_dtype(data["Revenue"]):
    y_true = data["Revenue"].astype(int)

elif pd.api.types.is_numeric_dtype(data["Revenue"]):
    if not set(data["Revenue"].unique()).issubset({0, 1}):
        st.error("Revenue must contain only 0 and 1.")
        st.stop()

    y_true = data["Revenue"].astype(int)

else:
    target_values = (
        data["Revenue"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    target_mapping = {
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "yes": 1,
        "no": 0
    }

    if not set(target_values.unique()).issubset(target_mapping):
        st.error(
            "Revenue must contain Boolean values or labels "
            "representing 0 and 1."
        )
        st.stop()

    y_true = target_values.map(target_mapping).astype(int)

if y_true.nunique() != 2:
    st.error(
        "The uploaded test dataset must contain both purchase "
        "and no-purchase examples to calculate all metrics."
    )
    st.stop()

# Separate features from the target.
X_test = data.drop(columns=["Revenue"])

st.success("Test dataset uploaded successfully.")

col1, col2 = st.columns(2)

col1.metric("Test Samples", len(data))
col2.metric("Input Features", X_test.shape[1])

with st.expander("Preview uploaded dataset"):
    st.dataframe(data.head(10), use_container_width=True)


# --------------------------------------------------
# Model selection
# --------------------------------------------------

st.header("2. Select Classification Model")

selected_model = st.selectbox(
    "Choose a model",
    list(MODEL_FILES.keys())
)

model_path = MODEL_DIR / MODEL_FILES[selected_model]

if not model_path.exists():
    st.error(
        f"Model file not found: {model_path.name}. "
        "Please upload the saved model to the model/ directory "
        "in your GitHub repository."
    )
    st.stop()

try:
    model = load_model(str(model_path))
except Exception as error:
    st.error(f"Could not load the selected model: {error}")
    st.stop()


# --------------------------------------------------
# Validate input features
# --------------------------------------------------

try:
    expected_features = list(
        model.named_steps["preprocessor"].feature_names_in_
    )
except (KeyError, AttributeError):
    st.error(
        "The saved model does not contain the expected "
        "preprocessing pipeline."
    )
    st.stop()

missing_features = [
    feature for feature in expected_features
    if feature not in X_test.columns
]

if missing_features:
    st.error(
        "The uploaded CSV is missing required features: "
        + ", ".join(missing_features)
    )
    st.stop()

# Select features in the same order used during training.
X_test = X_test[expected_features]


# --------------------------------------------------
# Make predictions
# --------------------------------------------------

if st.button("Evaluate Model", type="primary"):

    try:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        precision = precision_score(
            y_true, y_pred, zero_division=0
        )
        recall = recall_score(
            y_true, y_pred, zero_division=0
        )
        f1 = f1_score(
            y_true, y_pred, zero_division=0
        )
        mcc = matthews_corrcoef(y_true, y_pred)

    except Exception as error:
        st.error(f"Evaluation failed: {error}")
        st.stop()

    # --------------------------------------------------
    # Evaluation metrics
    # --------------------------------------------------

    st.header("3. Evaluation Metrics")

    st.subheader(f"Results: {selected_model}")

    metric_columns = st.columns(3)

    metric_columns[0].metric("Accuracy", f"{accuracy:.4f}")
    metric_columns[1].metric("AUC", f"{auc:.4f}")
    metric_columns[2].metric("Precision", f"{precision:.4f}")

    metric_columns = st.columns(3)

    metric_columns[0].metric("Recall", f"{recall:.4f}")
    metric_columns[1].metric("F1 Score", f"{f1:.4f}")
    metric_columns[2].metric("MCC", f"{mcc:.4f}")

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    st.header("4. Confusion Matrix")

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual No Purchase", "Actual Purchase"],
        columns=["Predicted No Purchase", "Predicted Purchase"]
    )

    st.dataframe(cm_df, use_container_width=True)

    st.caption(
        "Rows represent actual classes; columns represent "
        "predicted classes."
    )

    # --------------------------------------------------
    # Classification report
    # --------------------------------------------------

    st.header("5. Classification Report")

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["No Purchase", "Purchase"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )

    # --------------------------------------------------
    # Prediction preview
    # --------------------------------------------------

    st.header("6. Sample Predictions")

    prediction_results = X_test.copy()

    prediction_results["Actual Revenue"] = y_true.to_numpy()
    prediction_results["Predicted Revenue"] = y_pred
    prediction_results["Purchase Probability"] = y_prob

    st.dataframe(
        prediction_results.head(20),
        use_container_width=True
    )

    csv_data = prediction_results.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Predictions as CSV",
        data=csv_data,
        file_name="purchase_predictions.csv",
        mime="text/csv"
    )