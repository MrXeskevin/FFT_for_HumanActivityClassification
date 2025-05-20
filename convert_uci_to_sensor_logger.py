import numpy as np
import pandas as pd
import os
from pathlib import Path

def convert_uci_to_sensor_logger(input_file, output_file, activity_label):
    """
    Convert UCI HAR dataset format to Sensor Logger format.
    
    Args:
        input_file (str): Path to UCI HAR data file
        output_file (str): Path to save converted file
        activity_label (str): Activity label for the data
    """
    # Read UCI HAR data
    data = pd.read_csv(input_file, header=None, delim_whitespace=True)
    
    # Create timestamp column (assuming 50Hz sampling rate)
    timestamps = np.arange(0, len(data) * 0.02, 0.02)
    
    # Create DataFrame in Sensor Logger format
    df = pd.DataFrame({
        'timestamp': timestamps,
        'acc_x': data[0],  # Total acceleration X
        'acc_y': data[1],  # Total acceleration Y
        'acc_z': data[2],  # Total acceleration Z
        'gyro_x': data[3],  # Gyroscope X
        'gyro_y': data[4],  # Gyroscope Y
        'gyro_z': data[5],  # Gyroscope Z
        'activity': activity_label
    })
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Converted {input_file} to {output_file}")

def create_test_folder():
    """Create a test folder with converted UCI HAR data."""
    # Create test folder
    test_folder = Path("test_data")
    test_folder.mkdir(exist_ok=True)
    
    # Activity labels
    activities = {
        'WALKING': 'walking',
        'WALKING_UPSTAIRS': 'walking_upstairs',
        'WALKING_DOWNSTAIRS': 'walking_downstairs',
        'SITTING': 'sitting',
        'STANDING': 'standing',
        'LAYING': 'laying'
    }
    
    # Convert files for each activity
    for activity, label in activities.items():
        # Create activity subfolder
        activity_folder = test_folder / label
        activity_folder.mkdir(exist_ok=True)
        
        # Convert 3 samples for each activity
        for i in range(1, 4):
            input_file = f"UCI_HAR_Dataset/train/Inertial Signals/body_acc_x_train.txt"
            output_file = activity_folder / f"sample_{i}.csv"
            
            # Note: In a real scenario, you would need the actual UCI HAR dataset files
            # This is just a placeholder to show the conversion process
            print(f"Converting {activity} sample {i}...")
            # convert_uci_to_sensor_logger(input_file, output_file, label)

if __name__ == "__main__":
    create_test_folder() 