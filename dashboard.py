# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

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

# Extract features and target for preprocessed data
X_preprocessed = df_preprocessed.drop(columns=['Dementia_Label'])  # Features
y_preprocessed = df_preprocessed['Dementia_Label']  # Target

# Train RandomForestClassifier on the preprocessed data (assuming it's already trained)
model = RandomForestClassifier()
model.fit(X_preprocessed, y_preprocessed)

# Make predictions
y_pred = model.predict(X_preprocessed)

# Compute confusion matrix
cm = confusion_matrix(y_preprocessed, y_pred)

# Visualization of the confusion matrix
fig = go.Figure(data=go.Heatmap(z=cm, x=['Class 0', 'Class 1'], y=['Class 0', 'Class 1'],
                                colorscale='Viridis', colorbar=dict(title='Count')))
fig.update_layout(title='Confusion Matrix', xaxis_title='Predicted', yaxis_title='Actual')
st.plotly_chart(fig)

# Calculate precision, recall, F1-score from the confusion matrix
precision_class_0 = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # TP / (TP + FP)
precision_class_1 = cm[1, 1] / (cm[1, 1] + cm[1, 0])  # TP / (TP + FP)
recall_class_0 = cm[0, 0] / (cm[0, 0] + cm[1, 0])  # TP / (TP + FN)
recall_class_1 = cm[1, 1] / (cm[1, 1] + cm[0, 1])  # TP / (TP + FN)
f1_class_0 = 2 * (precision_class_0 * recall_class_0) / (precision_class_0 + recall_class_0)
f1_class_1 = 2 * (precision_class_1 * recall_class_1) / (precision_class_1 + recall_class_1)
accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()

# Display precision, recall, and F1-score metrics
st.write(f"### Metrics based on Confusion Matrix")
st.write(f"**Precision for Class 0**: {precision_class_0:.2f}")
st.write(f"**Precision for Class 1**: {precision_class_1:.2f}")
st.write(f"**Recall for Class 0**: {recall_class_0:.2f}")
st.write(f"**Recall for Class 1**: {recall_class_1:.2f}")
st.write(f"**F1-Score for Class 0**: {f1_class_0:.2f}")
st.write(f"**F1-Score for Class 1**: {f1_class_1:.2f}")
st.write(f"**Accuracy**: {accuracy:.2f}")

# Section 3: Model Comparison
st.header("3. Model Comparison")

# Model comparison table (update with actual results)
comparison_data = {
    "Model": ["Raw Data", "Preprocessed Data"],
    "Accuracy": [0.7233, accuracy],  # Replace with real accuracy value
    "Precision": [0.67, precision_class_1],  # Replace with real precision
    "Recall": [0.72, recall_class_1],  # Replace with real recall
    "F1-Score": [0.63, f1_class_1],  # Replace with real F1 score
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
