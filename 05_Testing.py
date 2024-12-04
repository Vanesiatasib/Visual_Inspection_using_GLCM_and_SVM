import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix

# Paths to model and input data
model_svm_path = r"D:\Proposal Skripsi gas 2024\Skripsi Lancar Jaya\SKENARIO_FINAL_FIX\02_Data_Lama\Model_Mandiri.joblib"
input_data_path = r"D:\Proposal Skripsi gas 2024\Skripsi Lancar Jaya\SKENARIO_FINAL_FIX\01_Data_Baru\05_Normalisasi\hasil_normalisasi_baru_d_15_t_315.csv"

# Load input data
input_data = pd.read_csv(input_data_path)

# Extract target labels
y = input_data['label']

# Remove unnecessary columns and ensure feature matching
X = input_data.drop(columns=['label', 'filename'])

# Load model
model_svm = load(model_svm_path)

# Ensure test data matches training features
trained_features = model_svm.feature_names_in_

# Align test data columns with model's expected features
X = X[trained_features]

# Make predictions using the features
y_pred = model_svm.predict(X)

# Calculate accuracy
accuracy = (y_pred == y).mean()
print(f"Accuracy: {accuracy:.2%}")

# Evaluate classification performance
print("\nClassification Report:\n")
print(classification_report(y, y_pred))

# Generate confusion matrix
conf_matrix = confusion_matrix(y, y_pred)

# Normalize confusion matrix to get percentages
conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

tn, fp, fn, tp = conf_matrix.ravel()
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")

# Plot confusion matrix in percentages
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Confusion Matrix (Percentage)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
# Identify misclassified samples
misclassified = input_data.copy()
misclassified['Ground Truth'] = y
misclassified['Prediction'] = y_pred
misclassified = misclassified[misclassified['Ground Truth'] != misclassified['Prediction']]

# Display misclassified filenames and details
print("\nMisclassified Samples with Filenames:")
print(misclassified[['filename', 'Ground Truth', 'Prediction']].head(100))


# Plot class distribution in test data (optional)
plt.figure(figsize=(8, 4))
sns.histplot(y, label="Test Data", color="blue", alpha=0.5)
plt.title("Class Distribution in Test Data")
plt.legend()
plt.show()
