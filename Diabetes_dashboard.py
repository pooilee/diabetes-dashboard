import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report, ConfusionMatrixDisplay
)

# Set Streamlit Page Config
st.set_page_config(page_title="Diabetes Prediction Dashboard", layout="wide")

# Title
st.title("🔬 Diabetes Prediction Dashboard")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    return df

df = load_data()
st.sidebar.subheader("🔍 Data Overview")
st.sidebar.write(f"📊 Shape: {df.shape}")
st.sidebar.write(f"🔢 Data Types: {df.dtypes}")

# Handling missing values
zero_columns = ['SkinThickness', 'Insulin', 'Glucose', 'BloodPressure', 'BMI']
df[zero_columns] = df[zero_columns].replace(0, np.nan)

imputer = SimpleImputer(strategy='median')
df[zero_columns] = imputer.fit_transform(df[zero_columns])

# Remove duplicates
df = df.drop_duplicates()

# Handling Outliers using IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Standardization
X = df.drop('Outcome', axis=1)
y = df['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Data Splitting & Balancing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Sidebar - Model Selection
st.sidebar.subheader("📌 Select a Model")
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "Neural Network": MLPClassifier(random_state=42, max_iter=500)
}
model_choice = st.sidebar.selectbox("Choose Model", list(models.keys()))
selected_model = models[model_choice]

# Train Model
selected_model.fit(X_resampled, y_resampled)
y_pred = selected_model.predict(X_test)

# Model Performance Metrics
st.sidebar.subheader("📊 Model Performance")
st.sidebar.write(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")
st.sidebar.write(f"🎯 Precision: {precision_score(y_test, y_pred):.2f}")
st.sidebar.write(f"🔁 Recall: {recall_score(y_test, y_pred):.2f}")
st.sidebar.write(f"📈 F1 Score: {f1_score(y_test, y_pred):.2f}")

# Exploratory Data Analysis (EDA)
st.subheader("📊 Exploratory Data Analysis (EDA)")
col1, col2 = st.columns(2)

with col1:
    st.write("### 🔍 Feature Distributions")
    fig, ax = plt.subplots(figsize=(10, 6))
    df.hist(bins=20, ax=ax)
    st.pyplot(fig)

with col2:
    st.write("### 🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

# Prediction Section
st.subheader("🤖 Predict Diabetes for a New Patient")
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, step=1)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, step=1)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0, max_value=500, step=1)
    bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, step=0.1)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
    age = st.number_input("Age", min_value=1, max_value=120, step=1)

# Store inputs into a dataframe
user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
user_data_scaled = scaler.transform(user_data)  # Standardize input

# Prediction Button
if st.button("🔮 Predict"):
    prediction = selected_model.predict(user_data_scaled)
    prediction_prob = selected_model.predict_proba(user_data_scaled)[:, 1] if hasattr(selected_model, "predict_proba") else None

    if prediction[0] == 1:
        st.error(f"🚨 The model predicts **Diabetes** with probability: {prediction_prob[0]:.2f}" if prediction_prob is not None else "🚨 The model predicts **Diabetes**.")
    else:
        st.success(f"✅ The model predicts **No Diabetes** with probability: {1 - prediction_prob[0]:.2f}" if prediction_prob is not None else "✅ The model predicts **No Diabetes**.")

# ROC Curve
st.subheader("📈 ROC Curve Analysis")
if hasattr(selected_model, "predict_proba"):
    y_prob = selected_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"{model_choice} (AUC = {roc_auc_score(y_test, y_prob):.2f})")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)