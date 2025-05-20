import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

def load_uci_data(data_dir):
    """Load UCI HAR dataset and convert to our format."""
    # Load activity labels
    activity_labels = pd.read_csv(
        os.path.join(data_dir, 'activity_labels.txt'),
        sep=' ',
        header=None,
        names=['label_id', 'activity']
    )
    
    # Map activity names to our format
    activity_mapping = {
        'WALKING': 'walking',
        'WALKING_UPSTAIRS': 'climbing_stairs',
        'WALKING_DOWNSTAIRS': 'climbing_stairs',
        'SITTING': 'sitting',
        'STANDING': 'standing',
        'LAYING': 'sitting'  # We'll map LAYING to sitting as it's similar
    }
    
    # Process both train and test data
    for dataset in ['train', 'test']:
        # Load subject IDs
        subject_file = os.path.join(data_dir, dataset, f'subject_{dataset}.txt')
        subjects = pd.read_csv(subject_file, header=None, names=['subject_id'])
        
        # Load activity labels
        y_file = os.path.join(data_dir, dataset, f'y_{dataset}.txt')
        y = pd.read_csv(y_file, header=None, names=['label_id'])
        
        # Load accelerometer data
        X_file = os.path.join(data_dir, dataset, f'X_{dataset}.txt')
        X = pd.read_csv(X_file, sep='\s+', header=None)
        
        # Get the relevant columns (tBodyAcc-mean()-X, tBodyAcc-mean()-Y, tBodyAcc-mean()-Z)
        acc_cols = [0, 1, 2]  # These are the indices for tBodyAcc-mean()-X,Y,Z
        acc_data = X.iloc[:, acc_cols]
        acc_data.columns = ['acc_x', 'acc_y', 'acc_z']
        
        # Create timestamp column (assuming 50Hz sampling rate)
        start_time = datetime.now()
        timestamps = [start_time + timedelta(seconds=i/50) for i in range(len(acc_data))]
        
        # Combine all data
        df = pd.DataFrame({
            'timestamp': timestamps,
            'acc_x': acc_data['acc_x'],
            'acc_y': acc_data['acc_y'],
            'acc_z': acc_data['acc_z'],
            'label_id': y['label_id']
        })
        
        # Merge with activity labels
        df = df.merge(activity_labels, on='label_id')
        
        # Map activities to our format
        df['label'] = df['activity'].map(activity_mapping)
        
        # Drop unnecessary columns
        df = df.drop(['label_id', 'activity'], axis=1)
        
        # Save to training_data directory
        output_dir = 'training_data'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save each activity type separately
        for activity in df['label'].unique():
            activity_data = df[df['label'] == activity]
            output_file = os.path.join(output_dir, f'{activity}.csv')
            activity_data.to_csv(output_file, index=False)
            print(f"Saved {len(activity_data)} samples for {activity} to {output_file}")

if __name__ == "__main__":
    uci_dir = "UCI HAR dataset"
    print("Converting UCI HAR dataset to our format...")
    load_uci_data(uci_dir)
    print("Conversion complete!") 