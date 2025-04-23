import pandas as pd
import os

# Directory containing gesture CSVs
raw_data_dir = "data/raw"
output_path = "data/merged_hand_gesture_data_common_labels.csv"

# Collect all CSV files from the raw data directory
csv_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]

if not csv_files:
    print("[!] No CSV files found in 'data/raw'. Add gesture samples first.")
    exit()

merged_df = pd.DataFrame()

# Read and append each CSV file
for file in csv_files:
    file_path = os.path.join(raw_data_dir, file)
    df = pd.read_csv(file_path)

    # Make sure 'label' column exists
    if 'label' not in df.columns:
        print(f"[!] Skipping {file} - missing 'label' column.")
        continue

    merged_df = pd.concat([merged_df, df], ignore_index=True)

# Save merged data
os.makedirs("data", exist_ok=True)
merged_df.to_csv(output_path, index=False)
print(f"[+] Merged {len(csv_files)} files. Total samples: {len(merged_df)}")
print(f"[+] Saved to: {output_path}")
