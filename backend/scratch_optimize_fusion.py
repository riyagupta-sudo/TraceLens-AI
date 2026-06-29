import os
import sys
import json
import torch
import numpy as np
from PIL import Image, ImageOps
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

from app.dna_engine import (
    estimate_compression_artifacts, 
    AI_MODEL, 
    AI_TRANSFORM
)

VAL_PACK_DIR = os.path.join(BACKEND_DIR, "ml", "v2", "validation_pack")
VAL_MANIFEST = os.path.join(VAL_PACK_DIR, "validation_manifest.json")

def main():
    with open(VAL_MANIFEST, "r") as f:
        manifest = json.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AI_MODEL.to(device)
    AI_MODEL.eval()
    
    X = []
    y = []
    sources = []
    
    print("Collecting features for all validation images...")
    for idx, item in enumerate(manifest):
        fn = item["filename"]
        label_str = item["label"]
        source = item["source"]
        
        filepath = os.path.join(VAL_PACK_DIR, "REAL" if label_str == "REAL" else "FAKE", fn)
        if not os.path.exists(filepath):
            continue
            
        y_true = 1 if label_str == "FAKE" else 0
        
        try:
            img = Image.open(filepath)
            img_transposed = ImageOps.exif_transpose(img)
            img_rgb = img_transposed.convert("RGB")
            
            tensor = AI_TRANSFORM(img_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                logit = AI_MODEL(tensor).item()
                
            exif = img.getexif() or {}
            make = exif.get(271, "")
            model_tag = exif.get(272, "")
            software = exif.get(305, "")
            has_camera = bool(make or model_tag)
            
            img_gray = np.array(img_transposed.convert("L"))
            h, w = img_gray.shape
            if h > 500 or w > 500:
                img_gray_lap = cv2.resize(img_gray, (500, 500))
            else:
                img_gray_lap = img_gray
            laplacian = cv2.Laplacian(img_gray_lap, cv2.CV_64F)
            lap_var = float(np.var(laplacian))
            
            resized_fft = cv2.resize(img_gray, (256, 256))
            dft = np.fft.fft2(resized_fft)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1e-8)
            center = 128
            r_min, r_max = 64, 120
            y_grid, x_grid = np.ogrid[-center:256-center, -center:256-center]
            mask = (x_grid**2 + y_grid**2 >= r_min**2) & (x_grid**2 + y_grid**2 <= r_max**2)
            outer_ring = magnitude_spectrum[mask]
            mean_val = np.mean(outer_ring)
            std_val = np.std(outer_ring)
            peak_threshold = mean_val + 3.5 * std_val
            peaks = outer_ring[outer_ring > peak_threshold]
            num_peaks = len(peaks)
            
            blockiness = estimate_compression_artifacts(filepath)
            
            # Features
            feat = [
                logit,
                1.0 if has_camera else 0.0,
                lap_var,
                float(num_peaks),
                blockiness
            ]
            X.append(feat)
            y.append(y_true)
            sources.append(source)
            
        except Exception as e:
            print(f"Error processing {fn}: {e}")
            
    X = np.array(X)
    y = np.array(y)
    sources = np.array(sources)
    
    print(f"Collected {len(X)} samples.")
    
    # Grid search for the best subset of features, scaling, regularization C, and decision threshold
    best_fpr = 1.0
    best_params = None
    
    # We will try logistic regression on different feature combinations:
    # 0: logit, 1: has_camera, 2: lap_var, 3: num_peaks, 4: blockiness
    feature_combinations = [
        [0, 1, 2, 3, 4],
        [0, 2, 3, 4],
        [0, 1, 3, 4],
        [0, 3, 4],
        [0, 1, 4],
        [0, 4]
    ]
    
    real_mask = (y == 0)
    fake_mask = (y == 1)
    
    for f_indices in feature_combinations:
        for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
            # Scale features
            X_sub = X[:, f_indices]
            mean = np.mean(X_sub, axis=0)
            std = np.std(X_sub, axis=0)
            std[std == 0.0] = 1.0
            X_scaled = (X_sub - mean) / std
            
            lr = LogisticRegression(C=C, random_state=42, max_iter=1000)
            lr.fit(X_scaled, y)
            probs = lr.predict_proba(X_scaled)[:, 1]
            
            for threshold in np.linspace(0.1, 0.9, 81):
                preds = (probs >= threshold).astype(int)
                rec = recall_score(y, preds)
                
                if rec >= 0.85: # Constraint: Recall >= 85%
                    # Calculate FPR on cameras, screenshots, whatsapp
                    iphone_fpr = np.mean(preds[(sources == "IPHONE") & real_mask])
                    android_fpr = np.mean(preds[(sources == "ANDROID") & real_mask])
                    dslr_fpr = np.mean(preds[(sources == "DSLR") & real_mask])
                    ss_fpr = np.mean(preds[(sources == "SCREENSHOT") & real_mask])
                    wa_fpr = np.mean(preds[(sources == "WHATSAPP") & real_mask])
                    
                    avg_fpr = (iphone_fpr + android_fpr + dslr_fpr + ss_fpr + wa_fpr) / 5.0
                    
                    if avg_fpr < best_fpr:
                        best_fpr = avg_fpr
                        best_params = {
                            "features": f_indices,
                            "C": C,
                            "threshold": threshold,
                            "mean": mean.tolist(),
                            "std": std.tolist(),
                            "coef": lr.coef_[0].tolist(),
                            "intercept": float(lr.intercept_[0]),
                            "metrics": {
                                "recall": rec,
                                "avg_fpr": avg_fpr,
                                "iphone_fpr": iphone_fpr,
                                "android_fpr": android_fpr,
                                "dslr_fpr": dslr_fpr,
                                "screenshot_fpr": ss_fpr,
                                "whatsapp_fpr": wa_fpr
                            }
                        }
                        
    print("\nBest Parameters Found:")
    print(json.dumps(best_params, indent=4))

if __name__ == "__main__":
    main()
