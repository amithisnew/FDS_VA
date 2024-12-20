# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import json

# Load datasets
raw_data_path = "https://raw.githubusercontent.com/amithisnew/FDS_VA/refs/heads/main/raw_data_dementia.csv"
preprocessed_data_path = "https://raw.githubusercontent.com/amithisnew/FDS_VA/refs/heads/main/preprocessed_data_dementia.csv"

# Read raw and preprocessed data
try:
    df_raw = pd.read_csv(raw_data_path)
    df_preprocessed = pd.read_csv(preprocessed_data_path)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Title and Sidebar
st.title("Dementia Prediction System Dashboard")
st.sidebar.title("Options")

# Section 1: Data Distribution Visualization
st.header("1. Data Distribution Visualization")

data_selection = st.sidebar.radio("Choose data to display:", ("Raw Data", "Preprocessed Data"))
selected_data = df_raw if data_selection == "Raw Data" else df_preprocessed

st.write(f"### {data_selection} Distribution")
for col in selected_data.select_dtypes(include=["object", "int64", "float64"]).columns:
    fig = px.histogram(selected_data, x=col, title=f"Distribution of {col}")
    st.plotly_chart(fig, use_container_width=True)

# Section 2: Model Performance Metrics
st.header("2. Model Performance Metrics")

# Classification report for raw data
raw_report_json = """
{
    "0": {"precision": 0.73, "recall": 0.98, "f1-score": 0.84, "support": 332},
    "1": {"precision": 0.50, "recall": 0.05, "f1-score": 0.09, "support": 127},
    "accuracy": 0.7233115468409586,
    "macro avg": {"precision": 0.61, "recall": 0.51, "f1-score": 0.46, "support": 459},
    "weighted avg": {"precision": 0.67, "recall": 0.72, "f1-score": 0.63, "support": 459}
}
"""

# Classification report for preprocessed data
preprocessed_report_json = """
{
    "0": {"precision": 0.78, "recall": 0.87, "f1-score": 0.82, "support": 343},
    "1": {"precision": 0.84, "recall": 0.75, "f1-score": 0.79, "support": 326},
    "accuracy": 0.8071748878923767,
    "macro avg": {"precision": 0.81, "recall": 0.81, "f1-score": 0.81, "support": 669},
    "weighted avg": {"precision": 0.81, "recall": 0.81, "f1-score": 0.81, "support": 669}
}
"""

# Load JSON reports from strings
try:
    raw_report = json.loads(raw_report_json)
    preprocessed_report = json.loads(preprocessed_report_json)
except Exception as e:
    st.error(f"Error parsing classification reports: {e}")
    st.stop()

# Display metrics for raw data
st.write("### Training Metrics (Raw Data)")
raw_metrics = ["precision", "recall", "f1-score"]
raw_metric_data = {metric: [raw_report[str(i)][metric] for i in range(2)] for metric in raw_metrics}
raw_metric_data["Class"] = ["Class 0", "Class 1"]

raw_df_metrics = pd.DataFrame(raw_metric_data)
fig = px.bar(raw_df_metrics, x="Class", y=raw_metrics, barmode="group", title="Classification Metrics for Raw Data")
st.plotly_chart(fig, use_container_width=True)

# Display metrics for preprocessed data
st.write("### Training Metrics (Preprocessed Data)")
preprocessed_metrics = ["precision", "recall", "f1-score"]
preprocessed_metric_data = {metric: [preprocessed_report[str(i)][metric] for i in range(2)] for metric in preprocessed_metrics}
preprocessed_metric_data["Class"] = ["Class 0", "Class 1"]

preprocessed_df_metrics = pd.DataFrame(preprocessed_metric_data)
fig = px.bar(preprocessed_df_metrics, x="Class", y=preprocessed_metrics, barmode="group", title="Classification Metrics for Preprocessed Data")
st.plotly_chart(fig, use_container_width=True)

# Section 3: Model Comparison
st.header("3. Model Comparison")

# Model comparison table
comparison_data = {
    "Model": ["Raw Data", "Preprocessed Data"],
    "Accuracy": [0.7233, 0.8072],
    "Precision": [0.67, 0.81],
    "Recall": [0.72, 0.81],
    "F1-Score": [0.63, 0.81]
}
df_comparison = pd.DataFrame(comparison_data)

# Visualization of model comparison
fig = px.bar(df_comparison, x="Model", y=["Accuracy", "Precision", "Recall", "F1-Score"],
             barmode="group", title="Performance Comparison Between Raw and Preprocessed Data")
st.plotly_chart(fig, use_container_width=True)

# Display comparison table
st.write("### Comparison Table")
st.dataframe(df_comparison)

# Section 4: Insights
st.header("4. Insights")
st.markdown("""
- **Raw Data**: The model's performance on raw data shows an accuracy of 72.33%. However, the recall for Class 1 is very low (5%), indicating significant imbalance.
- **Preprocessed Data**: After preprocessing, the accuracy improves to 80.72%, and the metrics for Class 1 improve significantly, demonstrating balanced performance.
- **Significance**: Preprocessing has markedly improved the model's performance, especially for the minority class (Class 1). The balanced precision and recall result in a higher overall f1-score.
- Use the bar charts and tables above to analyze and compare performance metrics interactively.
""")
