import os
import pandas as pd
import numpy as np
from data_loader import DataLoader
from fft_processor import FFTProcessor
from feature_extraction import FeatureExtractor
from classifier import ActivityClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_training_data(data_dir: str) -> tuple:
    """
    Load training data from CSV files in the specified directory.
    Each file should be named as 'activity_name.csv' and contain a 'label' column.
    
    Args:
        data_dir (str): Directory containing training CSV files
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the label array
    """
    all_features = []
    all_labels = []
    
    # Initialize processors
    data_loader = DataLoader()
    fft_processor = FFTProcessor()
    feature_extractor = FeatureExtractor()
    
    # Process each file in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            
            try:
                # Load and process data
                df = data_loader.load_csv(file_path)
                sampling_rate = data_loader.get_sampling_rate(df)
                windows = data_loader.segment_data(df, int(sampling_rate))
                
                # Extract features from each window
                for window in windows:
                    window_data = window[['acc_x', 'acc_y', 'acc_z']].values
                    fft_features = fft_processor.process_window(window_data)
                    features = feature_extractor.extract_features(fft_features)
                    
                    all_features.append(features)
                    all_labels.append(window['label'].iloc[0])  # Assuming label is constant within window
                    
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
    
    return np.array(all_features), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, labels):
    """Plot confusion matrix for model evaluation."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels,
                yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def main():
    # Configuration
    DATA_DIR = "training_data"  # Directory containing labeled CSV files
    MODEL_PATH = "activity_classifier.pkl"
    
    # Create training data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created {DATA_DIR} directory. Please add your training CSV files there.")
        print("Each file should be named as 'activity_name.csv' and contain:")
        print("- timestamp, acc_x, acc_y, acc_z columns for accelerometer data")
        print("- label column with activity labels (walking, running, etc.)")
        return
    
    # Load and process training data
    print("Loading training data...")
    X, y = load_training_data(DATA_DIR)
    
    if len(X) == 0:
        print("No training data found. Please add CSV files to the training_data directory.")
        return
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and train classifier
    print("Training classifier...")
    classifier = ActivityClassifier()
    classifier.train(X_train, y_train)
    
    # Evaluate model
    print("\nModel Evaluation:")
    y_pred = classifier.model.predict(X_val)
    print(classification_report(y_val, y_pred))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_val, y_pred, classifier.activity_labels.values())
    
    # Save trained model
    classifier.save_model(MODEL_PATH)
    print(f"\nTrained model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main() 