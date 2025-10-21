import numpy as np
import joblib
import os 
from flask import Flask, render_template, request, jsonify

# IMPORTANT: The model and scaler must be trained and saved by running 
# the heart_disease_predictor.py script first.
MODEL_PATH = 'rf_heart_model.joblib'
SCALER_PATH = 'scaler.joblib'

model = None
scaler = None

# --- CRITICAL DEBUGGING SECTION ---
# Check if the model files exist before attempting to load them
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    print("--- FATAL ERROR: DEPLOYMENT FILES MISSING ---")
    print("One or both model artifacts were not found in this folder:")
    print(f"Model Path: {os.path.abspath(MODEL_PATH)} (Exists: {os.path.exists(MODEL_PATH)})")
    print(f"Scaler Path: {os.path.abspath(SCALER_PATH)} (Exists: {os.path.exists(SCALER_PATH)})")
    print("ACTION REQUIRED: Please ensure you ran 'python heart_disease_predictor.py' successfully.")
    # Exit cleanly, preventing silent failure later
    import sys
    sys.exit(1)
    
try:
    # Load the model and scaler artifacts globally when the server starts
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("--- SUCCESS: Model artifacts loaded from .joblib files. ---")

except Exception as e:
    print(f"--- FATAL ERROR: Error during artifact loading: {e} ---")
    import sys
    sys.exit(1)
# ---------------------------------

app = Flask(__name__)

def predict_heart_disease(patient_features):
    """
    Takes raw patient features (list/array) and returns prediction.
    This simulates the core logic of the OOP Predictor class.
    """
    if model is None or scaler is None:
        return "ERROR: Model not loaded.", 0.0

    try:
        # Convert list of features to numpy array and reshape for scaling
        features_array = np.array(patient_features).reshape(1, -1)
        
        # Apply the same scaler used during training
        scaled_features = features_array.copy()
        scaled_features = scaler.transform(features_array)
        
        # Make prediction and get probability (confidence score)
        prediction_result = model.predict(scaled_features)[0]
        prediction_proba = model.predict_proba(scaled_features)[0][1] # Probability of Class 1 (Disease)
        
        return int(prediction_result), round(prediction_proba * 100, 2)
    
    except Exception as e:
        print(f"Prediction Error: {e}")
        return "ERROR: Internal Prediction Failed.", 0.0

@app.route('/')
def home():
    """Renders the HTML form for user input."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the form submission, processes data, and returns the prediction."""
    try:
        # The order of these values MUST match the column order in the dataset:
        feature_keys = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        
        # Collect all 13 numerical feature values from the submitted form
        features = [float(request.form.get(key, 0)) for key in feature_keys]

        # Get the prediction and probability
        result, probability = predict_heart_disease(features)
        
        diagnosis = "Positive (High Risk)" if result == 1 else "Negative (Low Risk)"
        
        response = {
            'diagnosis': diagnosis,
            'probability': f"{probability}%",
            'input_features': dict(zip(feature_keys, features))
        }
        
        # Return the results to the user via JSON 
        return jsonify(response)

    except ValueError:
        return jsonify({'diagnosis': 'ERROR', 'probability': 'Invalid input. Please ensure all fields are numerical.', 'input_features': None}), 400
    except Exception as e:
        return jsonify({'diagnosis': 'ERROR', 'probability': f'An unknown error occurred: {e}', 'input_features': None}), 500

if __name__ == '__main__':
    # Run the web server
    app.run(debug=True)
