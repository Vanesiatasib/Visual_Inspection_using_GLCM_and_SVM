import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Path dataset Anda
file_path = r"D:\Proposal Skripsi gas 2024\Skripsi Lancar Jaya\SKENARIO_FINAL_FIX\02_Data_Lama\02_Hasil_Normalisasi\hasil_normalisasi_mandiri_d_15_t_315.csv"

# Membaca data
data = pd.read_csv(file_path)

# Ganti nilai label NG menjadi 1 dan OK menjadi 0
data["label"] = data["label"].replace({"NG": 1, "OK": 0})

# Pilih kolom yang relevan
features = ["contrast", "correlation", "energy", "ASM", "homogeneity", "dissimilarity"]  # Nama kolom fitur
label = "label"  # Nama kolom label

# Hitung korelasi terhadap label
correlation = data[features + [label]].corr()[label].drop(label)

# Generate a heatmap similar to the uploaded image using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(
    data[features + [label]].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"label": "Korelasi"}
)
plt.title("Korelasi Antar Fitur dan Label")
plt.tight_layout()
plt.show()
