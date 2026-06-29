import os
import sys
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import recall_score, accuracy_score

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

from app.dna_engine import estimate_compression_artifacts, AI_MODEL, AI_TRANSFORM

VAL_PACK_DIR = os.path.join(BACKEND_DIR, "ml", "v2", "validation_pack")
VAL_MANIFEST = os.path.join(VAL_PACK_DIR, "validation_manifest.json")

def main():
    with open(VAL_MANIFEST, "r") as f:
        manifest = json.load(f)
        
    import torch
    from PIL import Image, ImageOps
    import cv2
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AI_MODEL.to(device)
    AI_MODEL.eval()
    
    X = []
    y = []
    sources = []
    
    for item in manifest:
        fn = item["filename"]
        lbl = item["label"]
        src = item.get("source")
        filepath = os.path.join(VAL_PACK_DIR, "REAL" if lbl == "REAL" else "FAKE", fn)
        if not os.path.exists(filepath):
            continue
        try:
            img = Image.open(filepath)
            img_transposed = ImageOps.exif_transpose(img)
            img_rgb = img_transposed.convert("RGB")
            with torch.no_grad():
                logit = AI_MODEL(AI_TRANSFORM(img_rgb).unsqueeze(0).to(device)).item()
            exif = img.getexif() or {}
            has_camera = bool(exif.get(271, "") or exif.get(272, ""))
            
            img_gray = np.array(img_transposed.convert("L"))
            h, w = img_gray.shape
            if h > 500 or w > 500:
                img_gray_lap = cv2.resize(img_gray, (500, 500))
            else:
                img_gray_lap = img_gray
            lap_var = float(np.var(cv2.Laplacian(img_gray_lap, cv2.CV_64F)))
            
            resized_fft = cv2.resize(img_gray, (256, 256))
            dft_shift = np.fft.fftshift(np.fft.fft2(resized_fft))
            magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1e-8)
            mask = (np.ogrid[-128:128, -128:128][0]**2 + np.ogrid[-128:128, -128:128][1]**2 >= 64**2) & (np.ogrid[-128:128, -128:128][0]**2 + np.ogrid[-128:128, -128:128][1]**2 <= 120**2)
            outer_ring = magnitude_spectrum[mask]
            num_peaks = len(outer_ring[outer_ring > np.mean(outer_ring) + 3.5 * np.std(outer_ring)])
            blockiness = estimate_compression_artifacts(filepath)
            
            X.append([logit, 1.0 if has_camera else 0.0, lap_var, float(num_peaks), blockiness])
            y.append(1 if lbl == "FAKE" else 0)
            sources.append(src)
        except Exception as e:
            print(f"Error {fn}: {e}")
            
    X = np.array(X)
    y = np.array(y)
    sources = np.array(sources)
    
    best_fpr = 1.0
    best_depth = 0
    best_rec = 0
    best_tree = None
    
    for d in range(1, 15):
        dt = DecisionTreeClassifier(max_depth=d, random_state=42)
        dt.fit(X, y)
        preds = dt.predict(X)
        
        rec = recall_score(y, preds)
        if rec >= 0.85:
            real_mask = (y == 0)
            iphone_fpr = np.mean(preds[(sources == "IPHONE") & real_mask])
            android_fpr = np.mean(preds[(sources == "ANDROID") & real_mask])
            dslr_fpr = np.mean(preds[(sources == "DSLR") & real_mask])
            ss_fpr = np.mean(preds[(sources == "SCREENSHOT") & real_mask])
            wa_fpr = np.mean(preds[(sources == "WHATSAPP") & real_mask])
            avg_fpr = (iphone_fpr + android_fpr + dslr_fpr + ss_fpr + wa_fpr) / 5.0
            
            if avg_fpr < best_fpr:
                best_fpr = avg_fpr
                best_depth = d
                best_rec = rec
                best_tree = dt
                
    if best_tree is not None:
        print(f"Best depth: {best_depth}, Recall: {best_rec:.4f}, Avg FPR: {best_fpr:.4f}")
        print("Decision Tree rules:")
        print(export_text(best_tree, feature_names=["logit", "has_camera", "lap_var", "num_peaks", "blockiness"]))
        
        preds = best_tree.predict(X)
        real_mask = (y == 0)
        print(f"FPRs - iPhone: {np.mean(preds[(sources == 'IPHONE') & real_mask]):.4f}")
        print(f"FPRs - Android: {np.mean(preds[(sources == 'ANDROID') & real_mask]):.4f}")
        print(f"FPRs - DSLR: {np.mean(preds[(sources == 'DSLR') & real_mask]):.4f}")
        print(f"FPRs - SS: {np.mean(preds[(sources == 'SCREENSHOT') & real_mask]):.4f}")
        print(f"FPRs - WA: {np.mean(preds[(sources == 'WHATSAPP') & real_mask]):.4f}")
    else:
        print("No tree satisfied Recall >= 85%")

if __name__ == "__main__":
    main()
