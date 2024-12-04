import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from joblib import dump  # For saving the model

# Parameter values
d_values = [5, 10, 15]
t_values = [0, 45, 90, 315]

# Function to compute feature importances using Random Forest
def compute_feature_importances(d, t):
    """Computes feature importances and returns the top 3 features."""
    try:
        dataset_path = f"D:\\Proposal Skripsi gas 2024\\Skripsi Lancar Jaya\\SKENARIO_FINAL_FIX\\03_Gabungan\\02_Gabungan Normalisasi\\setelah_normalisasi_gabungan{d}_t_{t}.csv"
        df = pd.read_csv(dataset_path)
        
        # Select features and target
        X = df[['contrast', 'correlation', 'energy', 'ASM', 'homogeneity', 'dissimilarity']]
        y = df['label']

        # Train Random Forest model
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        
        # Compute feature importances
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        # Return top 3 features
        top_features = feature_importance_df['Feature'].head(3)
        return top_features, df
    
    except FileNotFoundError:
        print(f"File not found: {dataset_path}")
        return None, None

# Function to train and evaluate an SVM model using top 3 features
def train_svm(top_features, df):
    """Trains an SVM model using the top 3 features and returns evaluation results."""
    if top_features is None or df is None:
        return None, None, None, None

    # Prepare data for training
    X = df[top_features]
    y = df['label']

    # Split data into training, validation, and test sets (70:20:10 split)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

    # SVM Grid Search parameters
    param_grid = {
        'C': np.logspace(-3, 2, num=6),
        'kernel': ['rbf', 'linear', 'sigmoid', 'poly'],
        'gamma': ['scale', 'auto'],
        'degree': [2, 3]
    }

    # Train SVM with GridSearchCV
    svm_model = SVC()
    grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Evaluate the model
    val_score = grid_search.best_estimator_.score(X_val, y_val)
    test_score = grid_search.best_estimator_.score(X_test, y_test)
    
    print(f"Validation Accuracy: {val_score:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Test Accuracy: {test_score:.4f}")
    
    # Return results along with the best model
    return grid_search.best_params_, 1 - val_score, test_score, grid_search.best_estimator_

# Function to save results to a CSV file
def save_results_to_csv(data):
    """Saves results to a CSV file."""
    df_results = pd.DataFrame(data, columns=['d', 't', 'top_features', 'best_params', 'val_loss', 'test_accuracy'])
    df_results.to_csv('x_Hasil_Gabungan.csv', index=False)
    print("Results saved to x_Hasil_Mandiri.csv")

# Function to plot loss
def plot_loss(val_losses):
    """Plots the validation loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(val_losses, marker='o', color='r', linestyle='-', label='Validation Loss (1 - Accuracy)')
    plt.title('SVM Model Validation Loss vs Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function to process datasets and train models
def main():
    results = []
    val_losses = []  # Store validation losses for plotting
    best_model = None
    best_accuracy = 0
    best_model_info = None
    
    for d in d_values:
        for t in t_values:
            print(f"Processing d={d}, t={t}...")
            top_features, df = compute_feature_importances(d, t)
            if top_features is not None:
                best_params, val_loss, test_accuracy, model = train_svm(top_features, df)
                if best_params is not None:
                    results.append([d, t, top_features.tolist(), best_params, val_loss, test_accuracy])
                    val_losses.append(val_loss)  # Collect validation loss
                    
                    # Save the model if it has the greatest accuracy
                    if test_accuracy > best_accuracy:
                        best_accuracy = test_accuracy
                        best_model = model
                        best_model_info = {'d': d, 't': t, 'accuracy': test_accuracy, 'params': best_params}
    
    if results:
        save_results_to_csv(results)
        plot_loss(val_losses)
        
        # Save the best model
        if best_model:
            dump(best_model, 'Model_Gabungan.joblib')
            print(f"Best model saved with accuracy: {best_accuracy:.4f}")
            print(f"Best model info: {best_model_info}")

# Run the main function
if __name__ == "__main__":
    main()
