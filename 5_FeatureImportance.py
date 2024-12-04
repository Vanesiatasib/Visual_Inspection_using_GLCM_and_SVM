import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load your dataset
data_path = r"D:\Proposal Skripsi gas 2024\Skripsi Lancar Jaya\SKENARIO_FINAL_FIX\02_Data_Lama\02_Hasil_GLCM\Histogram_random_d_15_t_315.csv"
data = pd.read_csv(data_path)

# Replace 'NG' with 1 and 'OK' with 0
data.replace({'NG': 1, 'OK': 0}, inplace=True)

# Drop the 'filename' column
data.drop(columns=['filename'], inplace=True)

# Define features (X) and target (y)
X = data.drop(columns=["label"])
y = data["label"]

# Split data into train (70%) and temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Split temp into validation (20%) and test (10%)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

# Train Random Forest model on the training set
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Select the top 3 most important features
top_3_features = feature_importances.head(3)['Feature'].tolist()
print(f"Top 3 features: {top_3_features}")

# Filter the dataset to only include the top 3 features
X_top3 = X[top_3_features]

# Split data again with top 3 features
X_train_top3, X_temp_top3, y_train, y_temp = train_test_split(X_top3, y, test_size=0.3, random_state=42)
X_valid_top3, X_test_top3, y_valid, y_test = train_test_split(X_temp_top3, y_temp, test_size=1/3, random_state=42)

# Train a new model using only the top 3 features
rf_top3 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_top3.fit(X_train_top3, y_train)

# Display final feature importances for the reduced feature set
print("Feature importances for top 3 features:")
print(feature_importances.head(6))
