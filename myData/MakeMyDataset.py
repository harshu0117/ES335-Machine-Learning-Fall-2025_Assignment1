import os
import shutil
import pandas as pd

def process_and_split_custom_data(source_dir, dest_dir, train_subjects, test_subjects):
    """
    Reads custom HAR data, renames columns to the standard format (accx, accy, accz),
    splits it by subject into train/test sets, and organizes it into the
    'Combined' folder structure.

    Args:
        source_dir (str): Path to the directory with custom CSV files.
        dest_dir (str): Path to the output directory (e.g., 'MyData_Combined').
        train_subjects (list): List of participant IDs for the training set.
        test_subjects (list): List of participant IDs for the test set.
    """
    activity_map = {
        'sit': 'SITTING',
        'sleep': 'LAYING',
        'stand': 'STANDING',
        'walk': 'WALKING',
        'walkd': 'WALKING_DOWNSTAIRS',
        'walku': 'WALKING_UPSTAIRS'
    }

    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)
    print(f"Created clean directory: {dest_dir}")

    # --- THIS IS THE KEY FIX ---
    # Define the column name mapping
    column_rename_map = {'ax': 'accx', 'ay': 'accy', 'az': 'accz'}
    # --- END OF FIX ---

    for file_name in os.listdir(source_dir):
        if not file_name.endswith('.csv'):
            continue

        try:
            parts = file_name.replace('.csv', '').split('_')
            participant_id = parts[0]
            activity_key = parts[1]
            activity_name = activity_map.get(activity_key)

            if not activity_name:
                continue

            if participant_id in train_subjects:
                set_folder = 'Train'
            elif participant_id in test_subjects:
                set_folder = 'Test'
            else:
                continue

            activity_dest_dir = os.path.join(dest_dir, set_folder, activity_name)
            os.makedirs(activity_dest_dir, exist_ok=True)

            source_path = os.path.join(source_dir, file_name)
            dest_path = os.path.join(activity_dest_dir, f"Subject_{participant_id}.csv")

            # Read the original CSV
            data = pd.read_csv(source_path)
            # Rename the columns
            data.rename(columns=column_rename_map, inplace=True)
            # Save the new CSV with the correct column names
            data.to_csv(dest_path, index=False)
            
            print(f"Processed and saved '{file_name}' to '{dest_path}'")

        except (IndexError, KeyError):
            print(f"Warning: Skipping '{file_name}' as it does not match the expected format.")

    print("\nProcessing complete.")

if __name__ == '__main__':
    my_data_source = 'mydata'
    my_data_combined = 'MyData_Combined'
    
    train_participants = ['h', 'r']
    test_participants = ['v']
    
    process_and_split_custom_data(my_data_source, my_data_combined, train_participants, test_participants)