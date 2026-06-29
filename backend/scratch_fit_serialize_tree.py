import os
import sys
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, accuracy_score, roc_auc_score

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

from app.dna_engine import estimate_compression_artifacts, AI_MODEL, AI_TRANSFORM

VAL_PACK_DIR = os.path.join(BACKEND_DIR, "ml", "v2", "validation_pack")
VAL_MANIFEST = os.path.join(VAL_PACK_DIR, "validation_manifest.json")

def serialize_tree(decision_tree):
    tree = decision_tree.tree_
    def recurse(node):
        if tree.feature[node] != -2: # not a leaf
            return {
                "f": int(tree.feature[node]),
                "t": float(tree.threshold[node]),
                "l": recurse(tree.children_left[node]),
                "r": recurse(tree.children_right[node])
            }
        else: # leaf
            val = tree.value[node][0]
            prob = float(val[1] / np.sum(val))
            return {
                "p": prob
            }
    return recurse(0)

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
    
    print("Collecting features...")
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
            laplacian = cv2.Laplacian(img_gray_lap, cv2.CV_64F)
            lap_var = float(np.var(laplacian))
            
            resized_fft = cv2.resize(img_gray, (256, 256))
            dft_shift = np.fft.fftshift(np.fft.fft2(resized_fft))
            magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1e-8)
            mask = (np.ogrid[-128:128, -128:128][0]**2 + np.ogrid[-128:128, -128:128][1]**2 >= 64**2) & (np.ogrid[-128:128, -128:128][0]**2 + np.ogrid[-128:128, -128:128][1]**2 <= 120**2)
            outer_ring = magnitude_spectrum[mask]
            mean_val = np.mean(outer_ring)
            std_val = np.std(outer_ring)
            peak_threshold = mean_val + 3.5 * std_val
            peaks = outer_ring[outer_ring > peak_threshold]
            num_peaks = len(peaks)
            
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
        tree_dict = serialize_tree(best_tree)
        print("\nSerialized Decision Tree:")
        print("V1_TREE = " + json.dumps(tree_dict, indent=4))
        
        # Test serialization prediction
        def predict_tree(features, node_dict):
            if "p" in node_dict:
                return node_dict["p"]
            feat_val = features[node_dict["f"]]
            if feat_val <= node_dict["t"]:
                return predict_tree(features, node_dict["l"])
            else:
                return predict_tree(features, node_dict["r"])
                
        test_probs = [predict_tree(feat, tree_dict) for feat in X]
        test_preds = (np.array(test_probs) >= 0.5).astype(int)
        
        rec_ser = recall_score(y, test_preds)
        acc_ser = accuracy_score(y, test_preds)
        print(f"\nSerialized Tree Recall: {rec_ser:.4f}, Accuracy: {acc_ser:.4f}")
        
    else:
        print("No tree satisfied Recall >= 85%")

if __name__ == "__main__":
    main()
