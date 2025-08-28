#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#
#                                   CombineMyScript.py
#
# This script reads the raw accelerometer data collected from the smartphone,
# processes it, and organizes it into a structured 'Combined' folder.
# It splits the subjects into training and testing sets.
#
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

import os
import pandas as pd
import numpy as np
import shutil

# --- Configuration ---
# Path to the directory containing your raw CSV files (e.g., h_sit.csv)
raw_data_path = "mydata"

# Directory to save the processed data
output_path = "Combined"

# Dictionary to map file prefixes to activity names
ACTIVITIES = {
    'walk': 'WALKING',
    'walku': 'WALKING_UPSTAIRS',
    'walkd': 'WALKING_DOWNSTAIRS',
    'sit': 'SITTING',
    'stand': 'STANDING',
    'sleep': 'LAYING'
}

# --- Script Logic ---

# Clean up previous runs
if os.path.exists(output_path):
    shutil.rmtree(output_path)
print(f"Cleaned up old directory: {output_path}")

# Get all unique subjects from filenames (e.g., 'h', 'r', 'v')
all_files = [f for f in os.listdir(raw_data_path) if f.endswith('.csv')]
subjects = sorted(list(set([f.split('_')[0] for f in all_files])))

# Split subjects into training and testing sets (e.g., 80/20 split)
np.random.seed(42) # for reproducibility
np.random.shuffle(subjects)
split_idx = int(len(subjects) * 0.8)
train_subjects = subjects[:split_idx]
test_subjects = subjects[split_idx:]

print(f"Total Subjects: {len(subjects)}. Training: {train_subjects}, Testing: {test_subjects}")

# Process files for both training and testing sets
for subject_list, set_name in [(train_subjects, "Train"), (test_subjects, "Test")]:
    for subject in subject_list:
        # Find all files for the current subject
        subject_files = [f for f in all_files if f.startswith(f"{subject}_")]

        for file in subject_files:
            try:
                # Extract activity from filename (e.g., 'sit' from 'h_sit.csv')
                activity_prefix = file.replace('.csv', '').split('_')[1]
                activity_name = ACTIVITIES.get(activity_prefix)

                if not activity_name:
                    print(f"Warning: Skipping file with unknown activity prefix: {file}")
                    continue

                # Create the full path for saving the processed file
                save_dir = os.path.join(output_path, set_name, activity_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # Read the raw data
                file_path = os.path.join(raw_data_path, file)
                data = pd.read_csv(file_path)

                # --- FIX: Select and rename the correct columns ---
                # Your CSV has columns: 'time', 'gFx', 'gFy', 'gFz', 'TgF'
                # We need the 'gFx', 'gFy', 'gFz' columns
                processed_data = data[['gFx', 'gFy', 'gFz']].copy()
                processed_data.columns = ['accx', 'accy', 'accz']
                # --- END FIX ---

                # Save the processed data
                save_path = os.path.join(save_dir, f"Subject_{subject}.csv")
                processed_data.to_csv(save_path, index=False)

            except Exception as e:
                print(f"Error processing file {file}: {e}")

print(f"\nDone combining data. Processed data is in '{output_path}'")