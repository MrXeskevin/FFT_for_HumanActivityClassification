import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys

def generate_activity_data(activity, num_samples=128):
    """
    Generate synthetic accelerometer and gyroscope data for different activities.
    
    Args:
        activity (str): Activity type
        num_samples (int): Number of samples to generate
        
    Returns:
        np.ndarray: Generated sensor data
    """
    # Base frequencies and amplitudes for different activities
    activity_params = {
        'walking': {'freq': 2.0, 'amp': 1.5},
        'walking_upstairs': {'freq': 1.8, 'amp': 2.0},
        'walking_downstairs': {'freq': 1.9, 'amp': 2.2},
        'sitting': {'freq': 0.1, 'amp': 0.2},
        'standing': {'freq': 0.05, 'amp': 0.1},
        'laying': {'freq': 0.02, 'amp': 0.05}
    }
    
    params = activity_params.get(activity, {'freq': 0.1, 'amp': 0.2})
    t = np.linspace(0, 2*np.pi, num_samples)
    
    # Generate base signal
    base_signal = params['amp'] * np.sin(params['freq'] * t)
    
    # Add noise
    noise = np.random.normal(0, 0.1, num_samples)
    
    # Generate 6 channels of data (3 accelerometer + 3 gyroscope)
    data = np.zeros((num_samples, 6))
    
    # Accelerometer data
    data[:, 0] = base_signal + noise  # X-axis
    data[:, 1] = base_signal * 0.8 + noise  # Y-axis
    data[:, 2] = base_signal * 0.6 + noise  # Z-axis
    
    # Gyroscope data
    data[:, 3] = base_signal * 0.5 + noise  # X-axis
    data[:, 4] = base_signal * 0.4 + noise  # Y-axis
    data[:, 5] = base_signal * 0.3 + noise  # Z-axis
    
    return data

def create_test_folder():
    """Create a test folder with synthetic activity data."""
    try:
        # Create test folder
        test_folder = Path("test_data")
        test_folder.mkdir(exist_ok=True)
        
        # Activity labels
        activities = [
            'walking',
            'walking_upstairs',
            'walking_downstairs',
            'sitting',
            'standing',
            'laying'
        ]
        
        # Generate data for each activity
        for activity in activities:
            try:
                # Create activity subfolder
                activity_folder = test_folder / activity
                activity_folder.mkdir(exist_ok=True)
                
                # Generate 3 samples for each activity
                for i in range(1, 4):
                    try:
                        # Generate synthetic data
                        data = generate_activity_data(activity)
                        
                        # Create timestamp column (assuming 50Hz sampling rate)
                        timestamps = np.arange(0, len(data) * 0.02, 0.02)
                        
                        # Create DataFrame
                        df = pd.DataFrame({
                            'timestamp': timestamps,
                            'acc_x': data[:, 0],
                            'acc_y': data[:, 1],
                            'acc_z': data[:, 2],
                            'gyro_x': data[:, 3],
                            'gyro_y': data[:, 4],
                            'gyro_z': data[:, 5],
                            'activity': activity
                        })
                        
                        # Save to CSV
                        output_file = activity_folder / f"sample_{i}.csv"
                        df.to_csv(output_file, index=False)
                        print(f"Generated {activity} sample {i}")
                    except Exception as e:
                        print(f"Error generating sample {i} for {activity}: {str(e)}")
                        continue
            except Exception as e:
                print(f"Error processing activity {activity}: {str(e)}")
                continue
                
        print("\nTest data generation completed successfully!")
        return True
    except Exception as e:
        print(f"Error creating test folder: {str(e)}")
        return False

if __name__ == "__main__":
    # Suppress numpy warnings
    np.seterr(all='ignore')
    
    # Run the data generation
    success = create_test_folder()
    sys.exit(0 if success else 1) 