#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#
#                                   MakeMyDataset.py
#
# This script creates the final dataset from the structured data in the 'Combined' folder.
# It slices the time series data, assigns labels, and saves them as .npy files.
#
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

import os
import numpy as np
import pandas as pd
import shutil

# --- Configuration ---
# Directory where the combined data is located
combined_dir = "Combined"
# Directory to save the final dataset files
dataset_dir = "Dataset"

# Constants mirroring the original script
TIME_PERIODS = 10  # 10 seconds of data
SAMPLING_RATE = 50 # 50Hz
SAMPLES = TIME_PERIODS * SAMPLING_RATE # 500 samples
OFFSET = 100 # Start reading from the 100th sample to avoid noise at the beginning

# Activity to integer label mapping
CLASSES = {
    "WALKING": 1,
    "WALKING_UPSTAIRS": 2,
    "WALKING_DOWNSTAIRS": 3,
    "SITTING": 4,
    "STANDING": 5,
    "LAYING": 6
}
FOLDERS = ["LAYING", "SITTING", "STANDING", "WALKING", "WALKING_DOWNSTAIRS", "WALKING_UPSTAIRS"]

# --- Script Logic ---

# Create dataset directory if it doesn't exist
if os.path.exists(dataset_dir):
    shutil.rmtree(dataset_dir)
os.makedirs(dataset_dir)
print(f"Created dataset directory: {dataset_dir}")

# --- Load Training Data ---
X_train_list = []
y_train_list = []
train_path = os.path.join(combined_dir, "Train")

if os.path.exists(train_path):
    for folder in FOLDERS:
        activity_path = os.path.join(train_path, folder)
        if os.path.exists(activity_path):
            files = os.listdir(activity_path)
            for file in files:
                try:
                    df = pd.read_csv(os.path.join(activity_path, file), header=0)
                    # Slice the dataframe to get the required samples
                    sliced_df = df.iloc[OFFSET : OFFSET + SAMPLES]
                    if len(sliced_df) == SAMPLES:
                        X_train_list.append(sliced_df.values)
                        y_train_list.append(CLASSES[folder])
                except Exception as e:
                    print(f"Error processing training file {file}: {e}")
else:
    print(f"Warning: Training directory not found at {train_path}")


# --- Load Testing Data ---
X_test_list = []
y_test_list = []
test_path = os.path.join(combined_dir, "Test")

if os.path.exists(test_path):
    for folder in FOLDERS:
        activity_path = os.path.join(test_path, folder)
        if os.path.exists(activity_path):
            files = os.listdir(activity_path)
            for file in files:
                try:
                    df = pd.read_csv(os.path.join(activity_path, file), header=0)
                    # Slice the dataframe
                    sliced_df = df.iloc[OFFSET : OFFSET + SAMPLES]
                    if len(sliced_df) == SAMPLES:
                        X_test_list.append(sliced_df.values)
                        y_test_list.append(CLASSES[folder])
                except Exception as e:
                    print(f"Error processing testing file {file}: {e}")
else:
    print(f"Warning: Testing directory not found at {test_path}")

# --- Convert lists to numpy arrays and Save ---
if X_train_list:
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    np.save(os.path.join(dataset_dir, "X_train.npy"), X_train)
    np.save(os.path.join(dataset_dir, "y_train.npy"), y_train)
    print(f"\nFinal training data shape: {X_train.shape}")
else:
    print("\nNo training data was generated.")

if X_test_list:
    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)
    np.save(os.path.join(dataset_dir, "X_test.npy"), X_test)
    np.save(os.path.join(dataset_dir, "y_test.npy"), y_test)
    print(f"Final testing data shape:  {X_test.shape}")
else:
    print("No testing data was generated.")


print(f"\nDone creating the dataset. Files saved in '{dataset_dir}'")