import pandas as pd

# Parameters
d_values = [5, 10, 15]
t_values = [0, 45, 90, 315]

# Process each combination of d and t
for d in d_values:
    for t in t_values:
        input_path = fr"D:\\Proposal Skripsi gas 2024\\Skripsi Lancar Jaya\\05_Gabung_baru_lama_histogram_random\\setelah_normalisasi_gabungan{d}_t_{t}.csv"
        output_path = fr"D:\\Proposal Skripsi gas 2024\\Skripsi Lancar Jaya\\05_Gabung_baru_lama_histogram_random\\split_DataLama_3_d_{d}_t_{t}.csv"

        # Read input file
        data = pd.read_csv(input_path)

        # Extract rows 0 to 3835
        new_data = data.iloc[:3834]  # Include row 3835

        # Save to new CSV
        new_data.to_csv(output_path, index=False)
