import os
import sys
import json
import random
import time
import numpy as np
import torch
import cv2
from PIL import Image, ImageOps
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

from app.dna_engine import (
    estimate_compression_artifacts,
    AI_MODEL,
    AI_TRANSFORM,
    V1_TEMP,
    V1_THRESHOLD,
    W_LOGIT,
    W_EXIF,
    W_NOISE,
    W_FFT,
    W_BLOCK,
    LR_INTERCEPT
)

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

VAL_PACK_DIR = os.path.join(BACKEND_DIR, "ml", "v2", "validation_pack")
VAL_MANIFEST = os.path.join(VAL_PACK_DIR, "validation_manifest.json")
DATASET_ROOT = os.path.join(os.path.dirname(BACKEND_DIR), "dataset")

def extract_features(filepath):
    img = Image.open(filepath)
    img_transposed = ImageOps.exif_transpose(img)
    img_rgb = img_transposed.convert("RGB")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = AI_TRANSFORM(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = AI_MODEL(tensor).item()
        
    exif = img.getexif() or {}
    make = exif.get(271, "")
    model_tag = exif.get(272, "")
    software = exif.get(305, "")
    has_camera = bool(make or model_tag)
    
    ai_software_tags = ["midjourney", "stable diffusion", "dall-e", "firefly", "craiyon", "wombo", "artbreeder", "bing image", "adobe firefly", "generative fill"]
    software_detected = any(t in str(software).lower() for t in ai_software_tags)
    
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
    y, x = np.ogrid[-center:256-center, -center:256-center]
    mask = (x**2 + y**2 >= r_min**2) & (x**2 + y**2 <= r_max**2)
    outer_ring = magnitude_spectrum[mask]
    mean_val = np.mean(outer_ring)
    std_val = np.std(outer_ring)
    peak_threshold = mean_val + 3.5 * std_val
    peaks = outer_ring[outer_ring > peak_threshold]
    num_peaks = len(peaks)
    
    blockiness = estimate_compression_artifacts(filepath)
    
    return {
        "logit": logit,
        "has_camera": has_camera,
        "software_detected": software_detected,
        "lap_var": lap_var,
        "num_peaks": num_peaks,
        "blockiness": blockiness
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AI_MODEL.to(device)
    AI_MODEL.eval()
    
    # 1. Gather all files for validation and generalization
    with open(VAL_MANIFEST, "r") as f:
        manifest = json.load(f)
        
    val_files = []
    for item in manifest:
        fn = item["filename"]
        lbl = item["label"]
        filepath = os.path.join(VAL_PACK_DIR, "REAL" if lbl == "REAL" else "FAKE", fn)
        if os.path.exists(filepath):
            val_files.append((filepath, 1 if lbl == "FAKE" else 0, item.get("source")))
            
    # Extract features for Validation Pack (training/fitting set)
    print("Extracting features for Validation Pack...")
    X_val = []
    y_val = []
    val_sources = []
    for filepath, y_true, src in val_files:
        feats = extract_features(filepath)
        # Features: [logit, has_camera, lap_var, num_peaks, blockiness]
        X_val.append([
            feats["logit"],
            1.0 if feats["has_camera"] else 0.0,
            feats["lap_var"],
            float(feats["num_peaks"]),
            feats["blockiness"]
        ])
        y_val.append(y_true)
        val_sources.append(src)
        
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    val_sources = np.array(val_sources)
    
    # 2. Fit the models on Validation Pack
    # (a) Baseline Logistic Regression (fixed production weights)
    # z = (0.014536 * (logit / 20.0) - 1.488286 * exif_warning + 0.0 * noise_warning - 0.64545 * fft_warning - 0.413436 * blockiness_warning + 1.558882)
    # We evaluate it directly.
    
    # (b) Decision Tree Model (Depth 5 to prevent severe overfitting)
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_model.fit(X_val, y_val)
    
    # (c) Weighted Linear Fusion Model
    # We fit a simple linear regression or logistic regression on the features
    # but constraint/regularize it, or use a LogisticRegression trained via scikit-learn.
    # Let's fit a scikit-learn LogisticRegression model on the validation features.
    lr_fitted = LogisticRegression(C=1.0, random_state=42)
    lr_fitted.fit(X_val, y_val)
    
    # Let's define the prediction functions
    def predict_baseline(X_row):
        logit, has_cam, lap_var, num_peaks, block = X_row
        exif_warning = 1.0 if has_cam == 0.0 else 0.0
        noise_warning = 1.0 if lap_var < 5.0 else 0.0
        fft_warning = 1.0 if num_peaks > 15 else 0.0
        blockiness_warning = 1.0 if block < 0.9 else 0.0
        
        z = (W_LOGIT * (logit / V1_TEMP) + 
             W_EXIF * exif_warning + 
             W_NOISE * noise_warning + 
             W_FFT * fft_warning + 
             W_BLOCK * blockiness_warning + 
             LR_INTERCEPT)
        prob = 1.0 / (1.0 + np.exp(-z))
        pred = 1 if prob >= V1_THRESHOLD else 0
        return prob, pred
        
    def predict_dt(X_row):
        prob = dt_model.predict_proba([X_row])[0, 1]
        pred = 1 if prob >= 0.5 else 0
        return prob, pred
        
    def predict_weighted_linear(X_row):
        prob = lr_fitted.predict_proba([X_row])[0, 1]
        pred = 1 if prob >= 0.5 else 0
        return prob, pred
        
    # 3. Gather generalization datasets
    generalization_datasets = {}
    
    # (a) Smartphone Photos
    iphone_dir = os.path.join(DATASET_ROOT, "ai_corpus", "real", "iphone")
    android_dir = os.path.join(DATASET_ROOT, "ai_corpus", "real", "android")
    phone_files = []
    if os.path.exists(iphone_dir) and os.path.exists(android_dir):
        iphones = [os.path.join(iphone_dir, f) for f in os.listdir(iphone_dir) if f.lower().endswith((".jpg", ".jpeg"))]
        androids = [os.path.join(android_dir, f) for f in os.listdir(android_dir) if f.lower().endswith((".jpg", ".jpeg"))]
        for p in random.sample(iphones, min(50, len(iphones))) + random.sample(androids, min(50, len(androids))):
            phone_files.append((p, 0))
    generalization_datasets["Smartphone"] = phone_files
    
    # (b) DSLR Photos
    dslr_dir = os.path.join(DATASET_ROOT, "ai_corpus", "real", "dslr")
    dslr_files = []
    if os.path.exists(dslr_dir):
        for p in random.sample([os.path.join(dslr_dir, f) for f in os.listdir(dslr_dir) if f.lower().endswith((".jpg", ".jpeg"))], 50):
            dslr_files.append((p, 0))
    generalization_datasets["DSLR"] = dslr_files
    
    # (c) WhatsApp Images
    wa_dir = os.path.join(DATASET_ROOT, "ai_corpus", "real", "whatsapp")
    wa_files = []
    if os.path.exists(wa_dir):
        for p in random.sample([os.path.join(wa_dir, f) for f in os.listdir(wa_dir) if f.lower().endswith((".jpg", ".jpeg"))], 50):
            wa_files.append((p, 0))
    generalization_datasets["WhatsApp"] = wa_files
    
    # (d) Gmail Images (simulated by stripping metadata)
    gmail_files = []
    gmail_temp_dir = os.path.join(BACKEND_DIR, "gmail_simulated_compare")
    os.makedirs(gmail_temp_dir, exist_ok=True)
    source_pool = [x[0] for x in (phone_files + dslr_files)]
    if source_pool:
        for idx, src_path in enumerate(random.sample(source_pool, 50)):
            try:
                img = Image.open(src_path)
                img = ImageOps.exif_transpose(img)
                dst_path = os.path.join(gmail_temp_dir, f"gmail_compare_{idx}.jpg")
                img.convert("RGB").save(dst_path, "JPEG", quality=90)
                gmail_files.append((dst_path, 0))
            except Exception:
                pass
    generalization_datasets["Gmail"] = gmail_files
    
    # (e) Screenshots
    ss_dir = os.path.join(DATASET_ROOT, "Screenshot", "pictures")
    ss_files = []
    if os.path.exists(ss_dir):
        for p in random.sample([os.path.join(ss_dir, f) for f in os.listdir(ss_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))], 50):
            ss_files.append((p, 0))
    generalization_datasets["Screenshots"] = ss_files
    
    # (f) AI-Edited Images (CASIA tampered)
    edited_dir = os.path.join(DATASET_ROOT, "casia_binary", "tampered")
    edited_files = []
    if os.path.exists(edited_dir):
        for p in random.sample([os.path.join(edited_dir, f) for f in os.listdir(edited_dir) if f.lower().endswith((".jpg", ".jpeg"))], 50):
            edited_files.append((p, 1))
    generalization_datasets["AI-Edited"] = edited_files
    
    # (g) Variants (Crop/Resize/Compress)
    cropped_dir = os.path.join(DATASET_ROOT, "cropped")
    resized_dir = os.path.join(DATASET_ROOT, "resized")
    compressed_dir = os.path.join(DATASET_ROOT, "compressed")
    variant_files = []
    for d in [cropped_dir, resized_dir, compressed_dir]:
        if os.path.exists(d):
            files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            for p in random.sample(files, min(20, len(files))):
                variant_files.append((p, 0))
    generalization_datasets["Variants"] = variant_files
    
    # Add validation pack as one of the benchmarks
    all_eval_sets = {"Validation Pack": val_files}
    all_eval_sets.update(generalization_datasets)
    
    # 4. Evaluate all 3 models on all benchmarks
    results = {}
    
    for ds_name, file_list in all_eval_sets.items():
        print(f"Evaluating {ds_name}...")
        y_true = []
        
        preds_base, probs_base = [], []
        preds_dt, probs_dt = [], []
        preds_lr, probs_lr = [], []
        
        # Measure runtime
        t0 = time.perf_counter()
        
        for filepath, label in [(x[0], x[1]) for x in file_list]:
            try:
                feats = extract_features(filepath)
                X_row = [
                    feats["logit"],
                    1.0 if feats["has_camera"] else 0.0,
                    feats["lap_var"],
                    float(feats["num_peaks"]),
                    feats["blockiness"]
                ]
                
                # Baseline
                pb, prb = predict_baseline(X_row)
                preds_base.append(prb)
                probs_base.append(pb)
                
                # Decision Tree
                pdt, prdt = predict_dt(X_row)
                preds_dt.append(prdt)
                probs_dt.append(pdt)
                
                # Fitted LR (Weighted Linear Fusion)
                plr, prlr = predict_weighted_linear(X_row)
                preds_lr.append(prlr)
                probs_lr.append(plr)
                
                y_true.append(label)
            except Exception as e:
                pass
                
        y_true = np.array(y_true)
        
        # Helper to compute metrics
        def get_metrics(y_t, y_p, y_pr):
            acc = accuracy_score(y_t, y_p)
            prec = precision_score(y_t, y_p, zero_division=0)
            rec = recall_score(y_t, y_p, zero_division=0)
            f1 = f1_score(y_t, y_p, zero_division=0)
            fpr = np.sum((y_t == 0) & (y_p == 1)) / np.sum(y_t == 0) if np.sum(y_t == 0) > 0 else 0.0
            ece = 0.0
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            for i in range(n_bins):
                bin_lower = bin_boundaries[i]
                bin_upper = bin_boundaries[i + 1]
                in_bin = (y_pr >= bin_lower) & (y_pr < bin_upper)
                prop_in_bin = np.mean(in_bin)
                if prop_in_bin > 0:
                    accuracy_in_bin = np.mean(y_t[in_bin])
                    avg_confidence_in_bin = np.mean(y_pr[in_bin])
                    ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1), "fpr": float(fpr), "ece": float(ece)}
            
        results[ds_name] = {
            "baseline": get_metrics(y_true, np.array(preds_base), np.array(probs_base)),
            "dt": get_metrics(y_true, np.array(preds_dt), np.array(probs_dt)),
            "lr_fitted": get_metrics(y_true, np.array(preds_lr), np.array(probs_lr))
        }
        
    # Write comparison_report.md
    comp_report_path = os.path.join(BACKEND_DIR, "comparison_report.md")
    with open(comp_report_path, "w", encoding="utf-8") as f:
        f.write("# TraceLens AI Detector V1 – Pipeline Comparison Report\n\n")
        f.write("This report presents the comparative metrics of the three candidate fusion strategies across multiple generalization datasets.\n\n")
        
        for ds_name, models in results.items():
            f.write(f"## Dataset: {ds_name}\n\n")
            f.write("| Model | Accuracy | Precision | Recall | F1 Score | False Positive Rate | ECE |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            b = models["baseline"]
            f.write(f"| Logistic Regression (Baseline) | {b['accuracy']*100:.2f}% | {b['precision']*100:.2f}% | {b['recall']*100:.2f}% | {b['f1']:.4f} | {b['fpr']*100:.2f}% | {b['ece']:.4f} |\n")
            d = models["dt"]
            f.write(f"| Decision Tree (Fitted) | {d['accuracy']*100:.2f}% | {d['precision']*100:.2f}% | {d['recall']*100:.2f}% | {d['f1']:.4f} | {d['fpr']*100:.2f}% | {d['ece']:.4f} |\n")
            lr = models["lr_fitted"]
            f.write(f"| Weighted Linear (Fitted LR) | {lr['accuracy']*100:.2f}% | {lr['precision']*100:.2f}% | {lr['recall']*100:.2f}% | {lr['f1']:.4f} | {lr['fpr']*100:.2f}% | {lr['ece']:.4f} |\n\n")
            
    print(f"Saved comparison_report.md to {comp_report_path}")
    
    # Cleanup simulated folder
    try:
        for f in os.listdir(gmail_temp_dir):
            os.remove(os.path.join(gmail_temp_dir, f))
        os.rmdir(gmail_temp_dir)
    except Exception:
        pass

if __name__ == "__main__":
    main()
