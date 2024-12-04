import pandas as pd

# Input file path
input_path = fr"D:\Proposal Skripsi gas 2024\Skripsi Lancar Jaya\05_Gabung_baru_lama_histogram_random\setelah_normalisasi_gabungan15_t_315.csv"

# Load the data from the input file
data = pd.read_csv(input_path)

# Extract a specific range of rows from index 3835 to 4296 (Python uses zero-based indexing)
new_data = data.iloc[3834:4299]

# Save the extracted data to a new CSV file
output_path = r"D:\Proposal Skripsi gas 2024\Skripsi Lancar Jaya\05_Gabung_baru_lama_histogram_random\0_Split_data_uji_fix_d_15_t_315.csv"
new_data.to_csv(output_path, index=False)
