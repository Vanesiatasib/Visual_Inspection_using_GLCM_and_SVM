import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Parameter values
d_values = [5, 10, 15]
t_values = [0, 45, 90, 315]

# Loop over parameter values
for d in d_values:
    for t in t_values:
        try:
            # Load dataset
            dataset_path = f"D:\\Proposal Skripsi gas 2024\\Skripsi Lancar Jaya\\SKENARIO_FINAL_FIX\\01_Data_Baru\\04_GLCM\\Histogram_random_databaru_d_{d}_t_{t}.csv"
            df = pd.read_csv(dataset_path)
            
            # Select features, target, and filename
            X = df[['contrast', 'correlation', 'energy', 'ASM', 'homogeneity', 'dissimilarity']]
            y = df['label']
            filenames = df['filename']
            
            # Scale features using MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_scaled = scaler.fit_transform(X)
            
            # Create a new DataFrame with scaled data
            scaled_df = pd.DataFrame(X_scaled, columns=['contrast', 'correlation', 'energy', 'ASM', 'homogeneity', 'dissimilarity'])
            scaled_df['label'] = y  # Add label column back
            scaled_df['filename'] = filenames  # Add filename column back
            
            # Save scaled data to a new CSV file
            output_path = f"D:\\Proposal Skripsi gas 2024\\Skripsi Lancar Jaya\\SKENARIO_FINAL_FIX\\01_Data_Baru\\05_Normalisasi\\hasil_normalisasi_baru_d_{d}_t_{t}.csv"
            scaled_df.to_csv(output_path, index=False)
            
            print(f"Scaled data saved for d={d}, t={t} at {output_path}")
        except Exception as e:
            print(f"Error processing d={d}, t={t}: {e}")
