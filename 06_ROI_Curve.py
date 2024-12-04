import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Compute feature importances
def compute_feature_importances(d, t):
    dataset_path = f"D:\\Proposal Skripsi gas 2024\\Skripsi Lancar Jaya\\SKENARIO_FINAL_FIX\\03_Gabungan\\02_Gabungan Normalisasi\\setelah_normalisasi_gabungan15_t_315.csv"
    
    try:
        df = pd.read_csv(dataset_path)
        X = df[['homogeneity', 'dissimilarity', 'contrast']]
        y = df['label']

        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        top_features = feature_importance_df['Feature'].head(3)
        return top_features.tolist(), df
    except FileNotFoundError:
        print(f"File not found: {dataset_path}")
        return None, None

# Train and evaluate SVM model
def train_svm(top_features, df):
    if not top_features or df is None:
        return None, None, None

    df['label'] = df['label'].map({'OK': 0, 'NG': 1})

    X = df[top_features]
    y = df['label']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

    param_grid = {'C': [10], 'kernel': ['rbf'], 'gamma': ['scale'], 'degree': [2]}
    grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    val_score = grid_search.best_estimator_.score(X_val, y_val)
    test_score = grid_search.best_estimator_.score(X_test, y_test)

    print(f"Validation Accuracy: {val_score:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Test Accuracy: {test_score:.4f}")

    y_test_proba = grid_search.best_estimator_.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return grid_search.best_params_, val_score, test_score

# Main function
def main():
    d, t = 15, 315
    top_features, df = compute_feature_importances(d, t)
    if top_features:
        best_params, val_accuracy, test_accuracy = train_svm(top_features, df)
        if best_params:
            print(f"Results: d={d}, t={t}, Features={top_features}, Params={best_params}, Validation={val_accuracy:.4f}, Test={test_accuracy:.4f}")

if __name__ == "__main__":
    main()
