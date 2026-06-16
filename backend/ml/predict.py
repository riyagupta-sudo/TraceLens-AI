import os
import sys
import pickle
import pandas as pd
import numpy as np

# Load model globally for efficiency
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
MODEL_PATH = os.path.join(MODEL_DIR, "tracelens_rf.pkl")

_rf_model = None

def get_model():
    global _rf_model
    if _rf_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first.")
        with open(MODEL_PATH, "rb") as f:
            _rf_model = pickle.load(f)
    return _rf_model

def predict_from_features(features: dict) -> dict:
    """
    Accepts a dictionary of features and returns ML tampering prediction.
    
    Expected keys:
      - manipulation_risk: int (0-100)
      - screenshot_probability: int (0-100)
      - metadata_trust_score: int (0-100)
      - blockiness: float
      - ai_generation_probability: int (0-100)
      - stego_suspicion: int (0-100)
      - compression_status: str ("CLEAN", "LOW", "HEAVY") OR compression_status_num: int (0, 1, 2)
    """
    rf = get_model()
    
    # Map compression status if present as string
    comp_val = 0
    if "compression_status_num" in features:
        comp_val = features["compression_status_num"]
    elif "compression_status" in features:
        comp_status = features["compression_status"]
        comp_map = {"CLEAN": 0, "LOW": 1, "HEAVY": 2}
        comp_val = comp_map.get(str(comp_status).upper(), 0)
        
    # Standardize feature values in the exact order the classifier was trained on
    ordered_features = {
        "manipulation_risk": features.get("manipulation_risk", 0),
        "screenshot_probability": features.get("screenshot_probability", 0),
        "metadata_trust_score": features.get("metadata_trust_score", 100),
        "blockiness": features.get("blockiness", 1.0),
        "ai_generation_probability": features.get("ai_generation_probability", 0),
        "stego_suspicion": features.get("stego_suspicion", 0),
        "compression_status_num": comp_val
    }
    
    # Create single-row DataFrame with exact training columns
    df = pd.DataFrame([ordered_features])
    
    # Run prediction
    prob = float(rf.predict_proba(df)[0, 1])
    pred = int(rf.predict(df)[0])
    
    classification = "TAMPERED" if pred == 1 else "CLEAN"
    
    return {
        "ml_tampering_probability": prob,
        "ml_classification": classification
    }

def main():
    print("Executing predict.py sample verification...")
    
    # Sample 1: Mock Clean image profile (high metadata trust, low risk, low ELA blockiness)
    clean_sample = {
        "manipulation_risk": 15,
        "screenshot_probability": 10,
        "metadata_trust_score": 100,
        "blockiness": 1.01,
        "ai_generation_probability": 2,
        "stego_suspicion": 0,
        "compression_status": "CLEAN"
    }
    
    # Sample 2: Mock Tampered image profile (low metadata trust, high risk, high ELA blockiness)
    tampered_sample = {
        "manipulation_risk": 75,
        "screenshot_probability": 45,
        "metadata_trust_score": 15,
        "blockiness": 1.82,
        "ai_generation_probability": 10,
        "stego_suspicion": 15,
        "compression_status": "HEAVY"
    }
    
    try:
        res_clean = predict_from_features(clean_sample)
        print(f"\nSample Clean Prediction Output:\n{res_clean}")
        
        res_tampered = predict_from_features(tampered_sample)
        print(f"\nSample Tampered Prediction Output:\n{res_tampered}")
        
        print("\nSUCCESS: Inference engine verified successfully.")
    except Exception as e:
        print(f"ERROR: Inference verification failed: {e}")

if __name__ == "__main__":
    main()
