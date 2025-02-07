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

# Streamlit Title
st.title("Diabetes Prediction Dashboard")

# Load dataset
def load_data():
    df = pd.read_csv("diabetes.csv")
    return df

df = load_data()
st.write("### Data Preview")
st.dataframe(df.head())

# Show dataset info
st.write("### Dataset Information")
st.write(f"Shape: {df.shape}")
st.write(df.dtypes)

# Handling missing values
st.write("### Handling Missing Values")
zero_columns = ['SkinThickness', 'Insulin', 'Glucose', 'BloodPressure', 'BMI']
df[zero_columns] = df[zero_columns].replace(0, np.nan)
st.write("Missing Values Before Imputation:")
st.write(df.isnull().sum())

# Impute missing values
imputer = SimpleImputer(strategy='median')
df[zero_columns] = imputer.fit_transform(df[zero_columns])
st.write("Missing Values After Imputation:")
st.write(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()
st.write(f"Dataset shape after removing duplicates: {df.shape}")

# Handling Outliers
Q1 = df.quantile(0.20)
Q3 = df.quantile(0.80)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_no_outliers = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]
st.write(f"Dataset shape after removing outliers: {df_no_outliers.shape}")

# Standardization
X = df_no_outliers.drop('Outcome', axis=1)
y = df_no_outliers['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Exploratory Data Analysis (EDA)
st.write("### Exploratory Data Analysis (EDA)")

# Feature Distributions
st.write("### Feature Distributions")
fig, ax = plt.subplots(figsize=(15, 10))
df_no_outliers.hist(bins=20, ax=ax)
st.pyplot(fig)

# Boxplots
st.write("### Boxplots of Features")
fig, ax = plt.subplots(figsize=(15, 10))
sns.boxplot(data=df_no_outliers, orient="h", ax=ax)
st.pyplot(fig)

# Visualization
st.write("### Data Visualization")
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df_no_outliers, ax=ax)
st.pyplot(fig)

# Correlation Heatmap
st.write("### Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df_no_outliers.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
st.pyplot(fig)

# Pairplot
st.write("### Pairplot of Features")
st.write("This visualization helps understand relationships between features.")
pairplot_fig = sns.pairplot(df_no_outliers, hue="Outcome", palette="Set1")
st.pyplot(pairplot_fig)

# SMOTE Balancing
df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(df_X_train, df_y_train)
st.write("### Class Distribution After SMOTE")
st.write(y_resampled.value_counts())

# Model Training
st.write("### Model Training and Evaluation")
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "Neural Network": MLPClassifier(random_state=42, max_iter=500)
}

model_choice = st.selectbox("Select a Model", list(models.keys()))
selected_model = models[model_choice]
selected_model.fit(X_resampled, y_resampled)
y_pred = selected_model.predict(df_X_test)

st.write("### Model Performance")
st.write(f"Accuracy: {accuracy_score(df_y_test, y_pred):.2f}")
st.write(f"Precision: {precision_score(df_y_test, y_pred):.2f}")
st.write(f"Recall: {recall_score(df_y_test, y_pred):.2f}")
st.write(f"F1 Score: {f1_score(df_y_test, y_pred):.2f}")

# Confusion Matrix
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay.from_estimator(selected_model, df_X_test, df_y_test, cmap='Blues', ax=ax)
st.pyplot(fig)

# Classification Report
st.write("### Classification Report")
st.text(classification_report(df_y_test, y_pred))

# ROC Curve
if hasattr(selected_model, "predict_proba"):
    y_prob = selected_model.predict_proba(df_X_test)[:, 1]
    fpr, tpr, _ = roc_curve(df_y_test, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"{model_choice} (AUC = {roc_auc_score(df_y_test, y_prob):.2f})")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)