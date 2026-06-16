import os
import sys
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def main():
    # 1. Load feature_vectors.csv
    csv_path = "feature_vectors.csv"
    if not os.path.exists(csv_path):
        # Fallback if run from evaluate folder
        csv_path = "../feature_vectors.csv"
        if not os.path.exists(csv_path):
            print("ERROR: feature_vectors.csv not found. Please run evaluate_tracelens.py first.")
            return

    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")

    # 4. Convert compression_status into numeric values
    comp_map = {"CLEAN": 0, "LOW": 1, "HEAVY": 2}
    df["compression_status_num"] = df["compression_status"].map(comp_map).fillna(0)

    # 3. Features to use
    feature_cols = [
        "manipulation_risk",
        "screenshot_probability",
        "metadata_trust_score",
        "blockiness",
        "ai_generation_probability",
        "stego_suspicion",
        "compression_status_num"
    ]

    # Target
    y = df["label"]
    X = df[feature_cols]

    # 6. Perform train_test_split (stratified by label)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # 7. Train RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # 8. Calculate test metrics
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob))
    }

    # 11. Print metrics
    print("\n--- Test Set Metrics ---")
    for k, v in metrics.items():
        print(f"{k.upper()}: {v:.4f}")

    # 9. Save model and metrics
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    model_path = os.path.join(models_dir, "tracelens_rf.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(rf, f)
    print(f"\nSaved Random Forest model to: {model_path}")

    metrics_path = os.path.join(models_dir, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")

    # 12. Verify model can be loaded successfully
    print("\nVerifying model loading...")
    try:
        with open(model_path, "rb") as f:
            loaded_rf = pickle.load(f)
        
        # Test loaded model on a single mock row
        mock_row = X_test.iloc[[0]]
        test_pred = loaded_rf.predict(mock_row)
        print(f"SUCCESS: Model loaded successfully. Test prediction on mock row: {test_pred[0]}")
    except Exception as e:
        print(f"ERROR loading model: {e}")

if __name__ == "__main__":
    main()
