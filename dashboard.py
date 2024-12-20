# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import json
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load datasets
raw_data_path = "https://raw.githubusercontent.com/amithisnew/FDS_VA/refs/heads/main/raw_data_dementia.csv"
preprocessed_data_path = "https://raw.githubusercontent.com/amithisnew/FDS_VA/refs/heads/main/preprocessed_data_dementia.csv"

try:
    df_raw = pd.read_csv(raw_data_path)
    df_preprocessed = pd.read_csv(preprocessed_data_path)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Title and Sidebar
st.title("Dementia Prediction System Dashboard")
st.sidebar.title("Options")

# --- Preprocessing for Raw Data (for Confusion Matrix) ---
X_raw = df_raw.drop(columns=['Patient_ID', 'Dementia_Label'])
y_raw = df_raw['Dementia_Label']

num_features_raw = X_raw.select_dtypes(include=[np.number]).columns
cat_features_raw = X_raw.select_dtypes(include=[object]).columns

# Impute
num_imputer_raw = SimpleImputer(strategy='median')
X_raw[num_features_raw] = num_imputer_raw.fit_transform(X_raw[num_features_raw])
cat_imputer_raw = SimpleImputer(strategy='most_frequent')
X_raw[cat_features_raw] = cat_imputer_raw.fit_transform(X_raw[cat_features_raw])

# Scale
scaler_raw = StandardScaler()
X_raw[num_features_raw] = scaler_raw.fit_transform(X_raw[num_features_raw])

# Encode
encoder_raw = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded_raw = pd.DataFrame(encoder_raw.fit_transform(X_raw[cat_features_raw]), columns=encoder_raw.get_feature_names_out(cat_features_raw))
X_raw = X_raw.drop(cat_features_raw, axis=1)
X_raw = pd.concat([X_raw, X_encoded_raw], axis=1)


# --- Prediction for Raw Data (using a simple heuristic for demonstration) ---
X_raw['prediction'] = X_raw.apply(lambda x: 1 if x.sum() > x.shape[0] / 2 else 0, axis=1)

# --- Preprocessing for Preprocessed Data (for Confusion Matrix) ---
X_preprocessed_cm = df_preprocessed.drop(columns=['Patient_ID', 'Dementia_Label'])
y_preprocessed_cm = df_preprocessed['Dementia_Label']

num_features_preprocessed = X_preprocessed_cm.select_dtypes(include=[np.number]).columns
# Scale
scaler_preprocessed = StandardScaler()
X_preprocessed_cm[num_features_preprocessed] = scaler_preprocessed.fit_transform(X_preprocessed_cm[num_features_preprocessed])

# --- Prediction for Preprocessed Data (using a simple heuristic for demonstration) ---
X_preprocessed_cm['prediction'] = X_preprocessed_cm.apply(lambda x: 1 if x.sum() > x.shape[0] / 2 else 0, axis=1)


# Section 1: Data Distribution Visualization (unchanged)
# ... (rest of the code for data distribution, metrics, comparison, and insights)

# Confusion Matrix for Raw data
cm_raw = confusion_matrix(y_raw, X_raw['prediction'])
fig_cm_raw = px.imshow(cm_raw, labels=dict(x="Predicted", y="Actual"),
                    x=['No Dementia', 'Dementia'],
                    y=['No Dementia', 'Dementia'],
                    title="Confusion Matrix (Raw Data)")
st.plotly_chart(fig_cm_raw, use_container_width=True)

# Confusion Matrix for Preprocessed data
cm_preprocessed = confusion_matrix(y_preprocessed_cm, X_preprocessed_cm['prediction'])
fig_cm_preprocessed = px.imshow(cm_preprocessed, labels=dict(x="Predicted", y="Actual"),
                    x=['No Dementia', 'Dementia'],
                    y=['No Dementia', 'Dementia'],
                    title="Confusion Matrix (Preprocessed Data)")
st.plotly_chart(fig_cm_preprocessed, use_container_width=True)

# ... (rest of the code)v
# Load JSON reports from strings (if available)
try:
    raw_report = json.loads(raw_report_json)
    preprocessed_report = json.loads(preprocessed_report_json)
except Exception as e:
    st.error(f"Error parsing classification reports: {e}")
    st.stop()

# Display metrics for raw data (if report available)
if raw_report:
    st.write("### Training Metrics (Raw Data)")
    raw_metrics = ["precision", "recall", "f1-score"]
    raw_metric_data = {metric: [raw_report[str(i)][metric] for i in range(2)] for metric in raw_metrics}
    raw_metric_data["Class"] = ["Class 0", "Class 1"]

    raw_df_metrics = pd.DataFrame(raw_metric_data)
    fig = px.bar(raw_df_metrics, x="Class", y=raw_metrics, barmode="group", title="Classification Metrics for Raw Data")
    st.plotly_chart(fig, use_container_width=True)

    # Confusion Matrix for Raw data
    cm_raw = confusion_matrix(df_raw['Dementia_Label'], (df_raw.drop(columns=['Patient_ID', 'Dementia_Label']).select_dtypes(include=[np.number]).fillna(0).apply(lambda x: 1 if x.sum() > x.shape[0] / 2 else 0, axis=1)))
    fig_cm_raw = px.imshow(cm_raw, labels=dict(x="Predicted", y="Actual"),
                        x=['No Dementia', 'Dementia'],
                        y=['No Dementia', 'Dementia'],
                        title="Confusion Matrix (Raw Data)")
    st.plotly_chart(fig_cm_raw, use_container_width=True)

# Display metrics for preprocessed data (if report available)
if preprocessed_report:
    st.write("### Training Metrics (Preprocessed Data)")
    preprocessed_metrics = ["precision", "recall", "f1-score"]
    preprocessed_metric_data = {metric: [preprocessed_report[str(i)][metric] for i in range(2)] for metric in preprocessed_metrics}
    preprocessed_metric_data["Class"] = ["Class 0", "Class 1"]

    preprocessed_df_metrics = pd.DataFrame(preprocessed_metric_data)
    fig = px.bar(preprocessed_df_metrics, x="Class", y=preprocessed_metrics, barmode="group", title="Classification Metrics for Preprocessed Data")
    st.plotly_chart(fig, use_container_width=True)

    # Confusion Matrix for Preprocessed data
    cm_preprocessed = confusion_matrix(df_preprocessed['Dementia_Label'], (df_preprocessed.drop(columns=['Patient_ID', 'Dementia_Label']).select_dtypes(include=[np.number]).apply(lambda x: 1 if x.sum() > x.shape[0] / 2 else 0, axis=1)))
    fig_cm_preprocessed = px.imshow(cm_preprocessed, labels=dict(x="Predicted", y="Actual"),
                        x=['No Dementia', 'Dementia'],
                        y=['No Dementia', 'Dementia'],
                        title="Confusion Matrix (Preprocessed Data)")
    st.plotly_chart(fig_cm_preprocessed, use_container_width=True)


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
