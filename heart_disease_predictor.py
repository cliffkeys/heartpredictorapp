import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score
import joblib

# --- 1. OOP Data Handler Class ---
class DataHandler:
    """Handles data loading, cleaning, and preprocessing."""
    def __init__(self, url):
        self.data_url = url
        self.dataframe = None
        self.scaler = None
        self.load_data()

    def load_data(self):
        """Loads data from the URL, handles missing values, and converts target to binary."""
        # Using the standard UCI Heart Disease Dataset URL
        column_names = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
        ]
        try:
            # Load data, treating '?' as missing values
            self.dataframe = pd.read_csv(
                self.data_url, 
                names=column_names, 
                na_values='?', 
                header=None,
                skiprows=1 
            )
            
            # **FIX 1: Clean Imputation**
            # Fill missing numerical values with the mode
            mode_ca = self.dataframe['ca'].mode()[0]
            mode_thal = self.dataframe['thal'].mode()[0]
            self.dataframe['ca'] = self.dataframe['ca'].fillna(mode_ca)
            self.dataframe['thal'] = self.dataframe['thal'].fillna(mode_thal)

            # **FIX 2: Convert Target to Binary (0 or 1)**
            # Re-code target: 0 (no disease) remains 0. 
            # Values 1, 2, 3, 4 (various stages of disease) are converted to 1 (disease).
            self.dataframe['target'] = self.dataframe['target'].apply(lambda x: 1 if x > 0 else 0)

        except Exception as e:
            print(f"Error loading or cleaning data: {e}")
            self.dataframe = None

    def preprocess_data(self):
        """Splits data and applies standardization (scaling)."""
        if self.dataframe is None:
            raise ValueError("Dataframe is not loaded. Cannot preprocess.")
        
        # Features (X) and Target (y)
        X = self.dataframe.drop('target', axis=1)
        y = self.dataframe['target']
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply Scaling (Standardization) to all numerical features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

# --- 2. OOP Model Trainer Class ---
class ModelTrainer:
    """Trains and evaluates the machine learning model."""
    def __init__(self, model):
        self.model = model

    def train_model(self, X_train, y_train):
        """Fits the model to the training data."""
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        """Calculates and prints model performance metrics."""
        y_pred = self.model.predict(X_test)
        
        print("\n--- Model Performance Metrics ---")
        # Now that the target is strictly binary, these metrics work correctly
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
        print("---------------------------------")

    def save_model(self, model_path, scaler_path, scaler):
        """Saves the trained model and scaler for deployment."""
        joblib.dump(self.model, model_path)
        joblib.dump(scaler, scaler_path)
        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")

# --- 3. OOP Prediction Class (Not fully shown, but included for completeness) ---
# Note: The app.py loads the artifacts directly, but this structure remains
# for reference to the Object-Oriented Design in the report.
class Predictor:
    """Loads artifacts and makes real-time predictions."""
    def __init__(self, model_path, scaler_path):
        self.model = self._load_artifact(model_path)
        self.scaler = self._load_artifact(scaler_path)

    def _load_artifact(self, path):
        """Helper method to load a serialized object."""
        try:
            return joblib.load(path)
        except Exception as e:
            print(f"Error loading artifact from {path}: {e}")
            return None

    def predict(self, patient_features):
        """
        Predicts heart disease risk for a single patient.
        patient_features must be a list of 13 features.
        """
        if self.model is None or self.scaler is None:
            return 0, 0.0 # Return default if model is not loaded

        # Reshape for scaling (expects 2D array: [1, 13])
        features_array = np.array(patient_features).reshape(1, -1)
        
        # Apply the loaded scaler
        scaled_features = self.scaler.transform(features_array)
        
        # Make prediction
        prediction_result = self.model.predict(scaled_features)[0]
        prediction_proba = self.model.predict_proba(scaled_features)[0][1]
        
        return int(prediction_result), prediction_proba


# --- Main execution block to train and save the model/scaler ---
def train_and_save_artifacts():
    """Main function to execute the training process."""
    DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    MODEL_PATH = 'rf_heart_model.joblib'
    SCALER_PATH = 'scaler.joblib'

    print("Starting data loading and model training...")
    
    # 1. Data Handling
    handler = DataHandler(DATA_URL)
    if handler.dataframe is None:
        print("Training aborted due to data error.")
        return

    X_train_scaled, X_test_scaled, y_train, y_test = handler.preprocess_data()
    
    # 2. Model Training
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    trainer = ModelTrainer(rf_model)
    trainer.train_model(X_train_scaled, y_train)
    
    # 3. Evaluation
    trainer.evaluate_model(X_test_scaled, y_test)
    
    # 4. Save Artifacts
    trainer.save_model(MODEL_PATH, SCALER_PATH, handler.scaler)
    print("\nTraining complete. Artifacts are ready for deployment (app.py).")

if __name__ == '__main__':
    train_and_save_artifacts()
