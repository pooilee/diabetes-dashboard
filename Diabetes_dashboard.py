import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

df = pd.read_csv(r"C:\UTM Master\Agile Data Science\diabetes.csv")
df.head()
print(df.head())

df.dtypes
print(df.dtypes)
print('Shape of the dataset is ', df.shape)

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Check for duplicates
print(f"Duplicates before removal: {df.duplicated().sum()}")

# Drop duplicates if any
df = df.drop_duplicates()

# Verify duplicates are removed
print(f"Duplicates after removal: {df.duplicated().sum()}")

# Statistical summary
print("\nStatistical Summary:\n", df.describe())

# Replace zero values in SkinThickness, Insulin, Glucose, BloodPressure, and BMI with NaN for proper handling
df['SkinThickness'] = df['SkinThickness'].replace(0, np.nan)
df['Insulin'] = df['Insulin'].replace(0, np.nan)
df['Glucose'] = df['Glucose'].replace(0, np.nan)
df['BloodPressure'] = df['BloodPressure'].replace(0, np.nan)
df['BMI'] = df['BMI'].replace(0, np.nan)

# Check for missing values again
print("\nMissing Values After Zero Replacement:\n", df.isnull().sum())

# Impute missing values (optional) with median values of the respective columns
imputer = SimpleImputer(strategy='median')
df[['SkinThickness', 'Insulin', 'Glucose', 'BloodPressure', 'BMI']] = imputer.fit_transform(df[['SkinThickness', 'Insulin', 'Glucose', 'BloodPressure', 'BMI']])

# Check if there are still missing values after imputation
print("\nMissing Values After Imputation:\n", df.isnull().sum())

# Statistical summary
print("\nStatistical Summary:\n", df.describe())
print('Shape of the dataset is ', df.shape)

# Handling outlier
# Calculate IQR for each numeric column
Q1 = df.quantile(0.20)
Q3 = df.quantile(0.80)
IQR = Q3 - Q1

# Define lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers by filtering data that falls within the bounds
df_no_outliers = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

# Display the shape of the dataset before and after removing outliers
print(f"Original dataset shape: {df.shape}")
print(f"Dataset shape after removing outliers: {df_no_outliers.shape}")

# Standardization
# Separate the features and target variable
X = df_no_outliers.drop('Outcome', axis=1)  # Features
y = df_no_outliers['Outcome']  # Target variable

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the features to standardize them
X_scaled = scaler.fit_transform(X)

# Convert the scaled features back to a DataFrame for better readability
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Display the first few rows of the standardized features
print(X_scaled_df.head())

# Visualize the distribution before and after outlier removal
plt.figure(figsize=(12, 6))

# Before removing outliers
plt.subplot(1, 2, 1)
sns.boxplot(data=df)
plt.title("Before Outlier Removal")

# After removing outliers
plt.subplot(1, 2, 2)
sns.boxplot(data=df_no_outliers)
plt.title("After Outlier Removal")

plt.tight_layout()
plt.show()

# Exploratory Data Analysis (EDA)
# Visualizing Feature Distributions (Histograms)
# Plot histograms for each feature
df_no_outliers.hist(bins=20, figsize=(15, 10))
plt.suptitle('Feature Distributions', fontsize=16)
plt.show()

# Plot boxplots for each feature
plt.figure(figsize=(15, 10))
sns.boxplot(data=df_no_outliers, orient="h")
plt.title('Feature Spread and Outliers')
plt.show()

# Correlation matrix
corr_matrix = df_no_outliers.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Pairplot of features to check for relationships between them
sns.pairplot(df_no_outliers, hue="Outcome", palette="Set1")
plt.suptitle("Pairplot of Features", y=1.02)
plt.tight_layout()
plt.show()

# Plot the distribution of the target variable (Outcome)
plt.figure(figsize=(6, 4))
sns.countplot(x='Outcome', data=df_no_outliers, palette='Set1')
plt.title('Distribution of Outcome Variable')
plt.show()

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X = df_no_outliers.drop('Outcome', axis=1)  # Features
y = df_no_outliers['Outcome']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Apply SMOTE to the training data
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Check the new distribution of the Outcome variable
print(f"Resampled Outcome distribution:\n{y_resampled.value_counts()}")

# Classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report, ConfusionMatrixDisplay
)

# Define classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "Neural Network": MLPClassifier(random_state=42, max_iter=500)
}

# Prepare to store results
results = {}

# Train and evaluate models
for name, clf in classifiers.items():
    print(f"\nTraining {name}...")

    # Train the model
    clf.fit(X_resampled, y_resampled)

    # Predict on the test set
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

    # Store results
    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC AUC": roc_auc
    }

    # Print classification report
    print(f"\nClassification Report for {name}:\n")
    print(classification_report(y_test, y_pred))

    # Plot ROC Curve
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

# Finalize and show ROC curves
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="best")
plt.grid()
plt.show()

# Summarize results
print("\nSummary of Results:")
for name, metrics in results.items():
    print(f"\n{name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")