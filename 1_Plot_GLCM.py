import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler

# File path to the CSV file containing GLCM histogram data
file_path = r"D:\Proposal Skripsi gas 2024\Skripsi Lancar Jaya\SKENARIO_FINAL_FIX\01_Data_Baru\04_GLCM\Histogram_random_databaru_d_5_t_315.csv"

# Read the CSV file
data = pd.read_csv(file_path)

# List of GLCM feature names to extract
feature_names = ['contrast', 'correlation', 'energy', 'homogeneity', 'ASM', 'dissimilarity']

# Apply MinMaxScaler to normalize the features between 0 and 1
# scaler = MinMaxScaler(feature_range=(0, 1))
# data[feature_names] = scaler.fit_transform(data[feature_names])

# Split the data into NG and OK groups based on the 'label' column
ng_glcm_features = data[data['label'] == 'NG']
ok_glcm_features = data[data['label'] == 'OK']

# Create a 3D scatter plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')  # Create 3D axes

# Plot NG features in red
ax.scatter(ng_glcm_features['homogeneity'], 
           ng_glcm_features['dissimilarity'], 
           ng_glcm_features['contrast'], 
           color='red', marker='o', label='NG Images')

# Plot OK features in green
ax.scatter(ok_glcm_features['homogeneity'], 
           ok_glcm_features['dissimilarity'], 
           ok_glcm_features['contrast'], 
           color='green', marker='^', label='OK Images')

# Set plot titles and axis labels
ax.set_title('GLCM Features : 3D Scatter Plot (NG vs OK)')
ax.set_xlabel('homogeneity')
ax.set_ylabel('dissimilarity')
ax.set_zlabel('contrast')

# Add legend
ax.legend()

# Display the plot
plt.show()
