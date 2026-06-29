import os
import sys
import json
import numpy as np
import torch
import timm
from PIL import Image
from torchvision import transforms

# Setup paths
backend_dir = r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI\backend"
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

MODEL_PATH = os.path.join(backend_dir, "models", "ai_detector.pth")
CONFIG_DIR = os.path.join(backend_dir, "config")
CONFIG_PATH = os.path.join(CONFIG_DIR, "model_calibration.json")
TEST_DIR = r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\ai_detection\test"
REAL_DIR = os.path.join(TEST_DIR, "REAL")
FAKE_DIR = os.path.join(TEST_DIR, "FAKE")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("AI Detector Calibration Script")
    print(f"Loading model from: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        sys.exit(1)
        
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Get test files
    if not os.path.exists(REAL_DIR) or not os.path.exists(FAKE_DIR):
        print(f"ERROR: Test folders not found at {REAL_DIR} or {FAKE_DIR}")
        sys.exit(1)
        
    fake_files = sorted([os.path.join(FAKE_DIR, f) for f in os.listdir(FAKE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    real_files = sorted([os.path.join(REAL_DIR, f) for f in os.listdir(REAL_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"Found {len(fake_files)} FAKE files and {len(real_files)} REAL files.")
    
    def extract_logits(files):
        logits = []
        batch_size = 128
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i+batch_size]
            tensors = []
            for fpath in batch_files:
                try:
                    img = Image.open(fpath).convert("RGB")
                    tensors.append(transform(img))
                except Exception as e:
                    print(f"Error loading {fpath}: {e}")
            if not tensors:
                continue
            batch_tensor = torch.stack(tensors).to(device)
            with torch.no_grad():
                outputs = model(batch_tensor).squeeze(1).cpu().numpy()
                logits.extend(outputs.tolist())
        return logits

    print("Extracting logits...")
    fake_logits = np.array(extract_logits(fake_files))
    real_logits = np.array(extract_logits(real_files))

    # Ground truth: FAKE = 1 (target class for AI probability), REAL = 0
    # Note: Model's raw sigmoid output corresponds to REAL (class 1).
    # Thus: REAL probability = sigmoid(logit)
    # FAKE probability = 1.0 - REAL probability = 1.0 - sigmoid(logit)
    # The true labels: FAKE is the positive class we want to calibrate, so y_true = 1 for FAKE, 0 for REAL.
    y_true = np.concatenate([np.ones_like(fake_logits), np.zeros_like(real_logits)])
    all_logits = np.concatenate([fake_logits, real_logits])

    def calculate_ece(y_true, y_prob, n_bins=10):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        bin_stats = []
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            # Use inclusive upper bound for the last bin
            if i == n_bins - 1:
                in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
            else:
                in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
                
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin])
                avg_confidence_in_bin = np.mean(y_prob[in_bin])
                ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
                bin_stats.append({
                    "bin": i,
                    "range": f"[{bin_lower:.1f}, {bin_upper:.1f})",
                    "count": int(np.sum(in_bin)),
                    "confidence": float(avg_confidence_in_bin),
                    "accuracy": float(accuracy_in_bin)
                })
            else:
                bin_stats.append({
                    "bin": i,
                    "range": f"[{bin_lower:.1f}, {bin_upper:.1f})",
                    "count": 0,
                    "confidence": 0.0,
                    "accuracy": 0.0
                })
        return ece, bin_stats

    temperatures = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    results = {}
    best_t = None
    min_ece = float('inf')

    print("\n--- Temperature Scaling Evaluation ---")
    
    for T in temperatures:
        scaled_logits = all_logits / T
        # prob of class 1 (REAL)
        prob_real = 1.0 / (1.0 + np.exp(-scaled_logits))
        # prob of class 0 (FAKE / AI)
        prob_fake = 1.0 - prob_real
        
        # Calculate Metrics
        ece, bin_stats = calculate_ece(y_true, prob_fake)
        brier_score = np.mean((prob_fake - y_true) ** 2)
        
        mean_prob = np.mean(prob_fake)
        median_prob = np.median(prob_fake)
        
        print(f"\nTemperature T = {T:.1f}:")
        print(f"  ECE:          {ece:.4f}")
        print(f"  Brier Score:  {brier_score:.4f}")
        print(f"  Mean AI Prob: {mean_prob:.4%}")
        print(f"  Med AI Prob:  {median_prob:.4%}")
        print("  Reliability Statistics (Bins):")
        for stat in bin_stats:
            if stat["count"] > 0:
                print(f"    Bin {stat['bin']} {stat['range']}: Count={stat['count']:<4} | Avg Conf={stat['confidence']:.2%} | Actual Acc={stat['accuracy']:.2%}")
                
        results[T] = {
            "ece": ece,
            "brier": brier_score,
            "mean": mean_prob,
            "median": median_prob,
            "bin_stats": bin_stats
        }
        
        # Best temperature minimizes Expected Calibration Error (ECE)
        if ece < min_ece:
            min_ece = ece
            best_t = T

    print(f"\nOptimal Temperature Selected: T = {best_t:.1f} (ECE = {min_ece:.4f})")
    
    # Ensure config directory exists
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    # Save config
    config_data = {
        "ai_temperature": float(best_t)
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config_data, f, indent=2)
        
    print(f"Saved optimal temperature to config file: {CONFIG_PATH}")
    print(f"Content: {config_data}")

if __name__ == "__main__":
    main()
