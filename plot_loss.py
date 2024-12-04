import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hinge_loss

# Compute feature importances using Random Forest
def compute_feature_importances(file_path):
    try:
        df = pd.read_csv(file_path)
        X = df[['contrast', 'correlation', 'energy', 'ASM', 'homogeneity', 'dissimilarity']]
        y = df['label']

        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        return feature_importance_df['Feature'].head(3), df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None, None

# Train and evaluate SVM using top features
def train_svm(top_features, df):
    if top_features is None or df is None:
        return None

    scaler = MinMaxScaler()
    df[top_features] = scaler.fit_transform(df[top_features])

    X = df[top_features]
    y = df['label']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

    param_grid = {
        'C': np.logspace(-3, 2, 6),
        'kernel': ['rbf', 'linear', 'sigmoid', 'poly'],
        'gamma': ['scale', 'auto'],
        'degree': [2, 3]
    }

    grid_search = GridSearchCV(SVC(), param_grid, cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    train_losses, val_losses, C_values = [], [], param_grid['C']
    for C in C_values:
        model = SVC(C=C, kernel='rbf')
        model.fit(X_train, y_train)
        train_losses.append(hinge_loss(y_train, model.decision_function(X_train)))
        val_losses.append(hinge_loss(y_val, model.decision_function(X_val)))

    plt.figure(figsize=(8, 6))
    plt.plot(C_values, train_losses, marker='o', label='Training Loss')
    plt.plot(C_values, val_losses, marker='s', label='Validation Loss')
    plt.xscale('log')
    plt.xlabel('Value of C')
    plt.ylabel('Hinge Loss')
    plt.title('Hinge Loss vs. C')
    plt.legend()
    plt.grid(True)
    plt.show()

    return grid_search.best_params_

# Main function
def main():
    dataset_path = "D:\\Proposal Skripsi gas 2024\\Skripsi Lancar Jaya\\SKENARIO_FINAL_FIX\\03_Gabungan\\02_Gabungan Normalisasi\\setelah_normalisasi_gabungan15_t_315.csv"
    top_features, df = compute_feature_importances(dataset_path)
    if top_features is not None:
        best_params = train_svm(top_features, df)
        print(f"Best SVM Parameters: {best_params}")

if __name__ == "__main__":
    main()
