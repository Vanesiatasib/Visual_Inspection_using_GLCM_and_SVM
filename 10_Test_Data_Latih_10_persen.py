import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# Function to train and evaluate an SVM model using top 3 features
def train_svm(top_features, df):
    """Trains an SVM model using the top 3 features and returns evaluation results."""
    # Replace "NG" with 1 and "OK" with 0 in the 'label' column
    df['label'] = df['label'].replace({'NG': 1, 'OK': 0})

    # Prepare data for training
    X = df[top_features]
    y = df['label']

    # Split data into training, validation, and test sets (70:20:10 split)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

    # Train SVM with predefined parameters
    svm_model = SVC(C=10, kernel='rbf', gamma='scale', degree=2)
    svm_model.fit(X_train, y_train)

    # Evaluate the model
    val_score = svm_model.score(X_val, y_val) * 100  # Convert to percentage
    test_score = svm_model.score(X_test, y_test) * 100  # Convert to percentage

    print(f"Validation Accuracy: {val_score:.2f}%")
    print(f"Test Accuracy: {test_score:.2f}%")

    # Confusion Matrix
    y_pred = svm_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_percentage = (cm / cm.sum(axis=1, keepdims=True)) * 100  # Convert to percentage

    # Extract TP, TN, FP, FN
    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")
    
    # Plot confusion matrix as percentages
    # Plot confusion matrix as absolute values
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=['NG', 'OK'], yticklabels=['NG', 'OK'])
    plt.title("Confusion Matrix (Absolute Values)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return val_score, test_score


# Main function to process datasets and train models
def main():
    # Load the dataset
    dataset_path = r"D:\Proposal Skripsi gas 2024\Skripsi Lancar Jaya\SKENARIO_FINAL_FIX\02_Data_Lama\02_Hasil_Normalisasi\hasil_normalisasi_mandiri_d_15_t_315.csv"
    df = pd.read_csv(dataset_path)
    
    # Top features
    top_features = ['contrast', 'homogeneity', 'dissimilarity']
    
    # Train and evaluate the SVM model
    train_svm(top_features, df)


# Run the main function
if __name__ == "__main__":
    main()
