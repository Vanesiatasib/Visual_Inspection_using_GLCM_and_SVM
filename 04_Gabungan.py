import pandas as pd

d = [5, 10, 15]
t = [0, 45, 90, 315]

for di in d:
    for ti in t:
        file1 = f"D:\\Proposal Skripsi gas 2024\\Skripsi Lancar Jaya\\04_Histogram_Random\\2_GLCM\\Histogram_random_d_{di}_t_{ti}.csv"
        file2 = f"D:\\Proposal Skripsi gas 2024\\Skripsi Lancar Jaya\\03_Data_Baru\\2_Histogram_random\\2_GLCM\Histogram_random_databaru_d_{di}_t_{ti}.csv"
        output_file = f"D:\\Proposal Skripsi gas 2024\\Skripsi Lancar Jaya\\05_Gabung_baru_lama_histogram_random\\Gabung_data_baru_lama_Histogram_random_d_{di}_t_{ti}.csv"
        
        # Read the CSV files
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        # Merge the files
        merged_df = pd.concat([df1, df2])
        
        # Save to output path
        merged_df.to_csv(output_file, index=False)
