import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import List, Dict, Union

class ActivityClassifier:
    def __init__(self, model_path: str = None):
        """
        Initialize the activity classifier.
        
        Args:
            model_path: Path to a pre-trained model file
        """
        if model_path:
            self.model = self.load_model(model_path)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Define activity labels as Python strings
        self.activity_labels = {
            0: 'walking',
            1: 'running',
            2: 'climbing_stairs',
            3: 'standing',
            4: 'sitting'
        }
        
        # Create reverse mapping for string to integer conversion
        self.label_to_int = {label: idx for idx, label in self.activity_labels.items()}
    
    def load_model(self, model_path: str) -> RandomForestClassifier:
        """
        Load a pre-trained model from file.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            RandomForestClassifier: Loaded model
        """
        return joblib.load(model_path)
    
    def save_model(self, model_path: str) -> None:
        """
        Save the trained model to file.
        
        Args:
            model_path: Path to save the model
        """
        joblib.dump(self.model, model_path)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the classifier on labeled data.
        
        Args:
            X: Feature matrix
            y: Activity labels
        """
        # Convert string labels to integers if necessary
        if isinstance(y[0], str):
            y = np.array([self.label_to_int[label] for label in y])
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> List[str]:
        """
        Predict activity labels for input features.
        
        Args:
            X: Feature matrix
            
        Returns:
            List[str]: Predicted activity labels
        """
        if not hasattr(self, 'model'):
            raise ValueError("Model not loaded or trained")
        
        # Ensure input is 2D
        X = np.array(X).reshape(1, -1) if len(X.shape) == 1 else X
        
        # Get numeric predictions
        y_pred = self.model.predict(X)
        
        # Convert numeric predictions to string labels
        predictions = []
        for pred in y_pred:
            if isinstance(pred, (np.int64, np.int32, int)):
                predictions.append(self.activity_labels[int(pred)])
            else:
                predictions.append(str(pred))
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability estimates for each activity.
        
        Args:
            X: Feature matrix
            
        Returns:
            np.ndarray: Probability estimates for each class
        """
        if not hasattr(self, 'model'):
            raise ValueError("Model not loaded or trained")
        
        # Ensure input is 2D
        X = np.array(X).reshape(1, -1) if len(X.shape) == 1 else X
        
        return self.model.predict_proba(X)
    
    def get_activity_labels(self) -> List[str]:
        """
        Get the list of activity labels in order.
        
        Returns:
            List[str]: List of activity labels
        """
        return [str(label) for label in self.activity_labels.values()] 