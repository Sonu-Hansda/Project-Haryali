import os
import sys
import subprocess
import pickle
import numpy as np
import pandas as pd

def install_packages(requirements_file='requirements.txt'):
    """Install required packages from requirements.txt."""
    if not os.path.isfile(requirements_file):
        print(f"Requirements file '{requirements_file}' not found.")
        return
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
        print("All required packages have been installed.")
    except Exception as e:
        print(f"An error occurred while installing packages: {e}")

def load_model_and_features():
    """Load the crop recommendation model and feature names."""
    try:
        with open('models/crop_recommendation_model.pkl', 'rb') as file:
            crop_recommendation_model = pickle.load(file)

        with open('models/features/feature_names.pkl', 'rb') as file:
            feature_names = pickle.load(file)

        return crop_recommendation_model, feature_names
    except Exception as e:
        print(f"Error while loading models: {e}")
        sys.exit(1)

def get_user_input():
    """Get user input for crop prediction features."""
    n_val = int(input('Nitrogen (N)  : '))
    p_val = int(input('Phosphorus (P): '))
    k_val = int(input('Potassium (K) : '))
    t_val = float(input('Temperature (T)(in °C): '))
    h_val = int(input('Humidity (H) : '))
    ph_val = float(input('PH value : '))
    r_val = float(input('Rainfall (in mm) : '))
    
    return np.array([n_val, p_val, k_val, t_val, h_val, ph_val, r_val])

def predict_crop(crop_recommendation_model, feature_array, feature_names):
    """Make a prediction using the crop recommendation model."""
    feature_array = pd.DataFrame([feature_array], columns=feature_names)
    print('Predicting ....')
    prediction = crop_recommendation_model.predict(feature_array)
    probability = crop_recommendation_model.predict_proba(feature_array)
    recommended_crop = prediction[0]
    confidence = float(max(probability[0])) * 100

    return recommended_crop, confidence

def main():
    print('WELCOME TO CROP RECOMMENDATION SYSTEM')
    install_packages()
    crop_recommendation_model, feature_names = load_model_and_features()
    feature_array = get_user_input()

    recommended_crop, confidence = predict_crop(crop_recommendation_model, feature_array, feature_names)

    print(f'Recommended Crop: {recommended_crop} ({confidence:.2f}%)')

if __name__ == "__main__":
    main()