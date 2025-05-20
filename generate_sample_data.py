import numpy as np
import pandas as pd
import os

def generate_walking_data(duration=10, sampling_rate=100):
    """Generate sample walking data."""
    t = np.linspace(0, duration, duration * sampling_rate)
    # Walking pattern: regular oscillations in all axes
    x = 0.5 * np.sin(2 * np.pi * 1.5 * t) + np.random.normal(0, 0.1, len(t))
    y = 0.3 * np.sin(2 * np.pi * 1.5 * t + np.pi/2) + np.random.normal(0, 0.1, len(t))
    z = 0.4 * np.sin(2 * np.pi * 1.5 * t + np.pi/4) + np.random.normal(0, 0.1, len(t))
    return create_dataframe(t, x, y, z)

def generate_running_data(duration=10, sampling_rate=100):
    """Generate sample running data."""
    t = np.linspace(0, duration, duration * sampling_rate)
    # Running pattern: higher frequency and amplitude oscillations
    x = 0.8 * np.sin(2 * np.pi * 2.5 * t) + np.random.normal(0, 0.15, len(t))
    y = 0.6 * np.sin(2 * np.pi * 2.5 * t + np.pi/2) + np.random.normal(0, 0.15, len(t))
    z = 0.7 * np.sin(2 * np.pi * 2.5 * t + np.pi/4) + np.random.normal(0, 0.15, len(t))
    return create_dataframe(t, x, y, z)

def generate_climbing_stairs_data(duration=10, sampling_rate=100):
    """Generate sample climbing stairs data."""
    t = np.linspace(0, duration, duration * sampling_rate)
    # Climbing pattern: periodic vertical movements
    x = 0.3 * np.sin(2 * np.pi * 0.8 * t) + np.random.normal(0, 0.1, len(t))
    y = 0.2 * np.sin(2 * np.pi * 0.8 * t + np.pi/2) + np.random.normal(0, 0.1, len(t))
    z = 0.6 * np.sin(2 * np.pi * 0.8 * t) + np.random.normal(0, 0.1, len(t))
    return create_dataframe(t, x, y, z)

def generate_standing_data(duration=10, sampling_rate=100):
    """Generate sample standing data."""
    t = np.linspace(0, duration, duration * sampling_rate)
    # Standing pattern: minimal movement with small variations
    x = np.random.normal(0, 0.05, len(t))
    y = np.random.normal(0, 0.05, len(t))
    z = np.random.normal(0, 0.05, len(t))
    return create_dataframe(t, x, y, z)

def generate_sitting_data(duration=10, sampling_rate=100):
    """Generate sample sitting data."""
    t = np.linspace(0, duration, duration * sampling_rate)
    # Sitting pattern: very minimal movement
    x = np.random.normal(0, 0.03, len(t))
    y = np.random.normal(0, 0.03, len(t))
    z = np.random.normal(0, 0.03, len(t))
    return create_dataframe(t, x, y, z)

def create_dataframe(t, x, y, z):
    """Create a DataFrame with timestamp and accelerometer data."""
    # Convert numpy arrays to Python native types
    df = pd.DataFrame({
        'timestamp': t.astype(float),
        'acc_x': x.astype(float),
        'acc_y': y.astype(float),
        'acc_z': z.astype(float)
    })
    return df

def main():
    # Create sample data directory if it doesn't exist
    os.makedirs('sample_data', exist_ok=True)
    
    # Generate data for each activity
    activities = {
        'walking': generate_walking_data,
        'running': generate_running_data,
        'climbing_stairs': generate_climbing_stairs_data,
        'standing': generate_standing_data,
        'sitting': generate_sitting_data
    }
    
    # Generate and save data for each activity
    for activity, generator in activities.items():
        df = generator()
        filename = f'sample_data/{activity}_data.csv'
        # Save with explicit float formatting
        df.to_csv(filename, index=False, float_format='%.6f')
        print(f"Generated {filename}")

if __name__ == "__main__":
    main() 