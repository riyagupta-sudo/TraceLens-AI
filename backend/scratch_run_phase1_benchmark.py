import os
import sys
import json
import random
import time
import numpy as np
import torch
import cv2
from PIL import Image, ImageOps
import sys

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

# Paths
VAL_PACK_DIR = os.path.join(BACKEND_DIR, "ml", "v2", "validation_pack")
VAL_MANIFEST = os.path.join(VAL_PACK_DIR, "validation_manifest.json")
DATASET_ROOT = os.path.join(os.path.dirname(BACKEND_DIR), "dataset")

# Helper to run the baseline model predictions
def predict_baseline_v1(logit, has_camera, lap_var, num_peaks, software_detected=False):
    # Base raw probability (T=8.0)
    ai_prob_raw = 1.0 - (1.0 / (1.0 + np.exp(-logit / 8.0)))
    
    # Heuristics
    fft_contrib = 10 if (num_peaks > 15) else 0
    lap_contrib = 10 if (lap_var < 5.0) else 0
    if software_detected:
        meta_contrib = 80
    elif has_camera:
        meta_contrib = -10
    else:
        meta_contrib = 10
        
    orig_fused = (ai_prob_raw * 100) + fft_contrib + lap_contrib + meta_contrib
    orig_fused_prob = min(98, max(2, orig_fused)) / 100.0
    pred = 1 if (orig_fused_prob * 100) >= 50 else 0
    return orig_fused_prob, pred

# Helper to run the improved V1 model predictions
def predict_improved_v1(logit, has_camera, lap_var, num_peaks, blockiness, software_detected=False):
    exif_warning = 1.0 if (not has_camera or software_detected) else 0.0
    noise_warning = 1.0 if lap_var < 5.0 else 0.0
    fft_warning = 1.0 if num_peaks > 15 else 0.0
    blockiness_warning = 1.0 if blockiness < 0.9 else 0.0
    
    z = (W_LOGIT * (logit / V1_TEMP) + 
         W_EXIF * exif_warning + 
         W_NOISE * noise_warning + 
         W_FFT * fft_warning + 
         W_BLOCK * blockiness_warning + 
         LR_INTERCEPT)
         
    fused_prob = 1.0 / (1.0 + np.exp(-z))
    pred = 1 if fused_prob >= V1_THRESHOLD else 0
    return fused_prob, pred

def extract_features(filepath):
    # Load and preprocess
    img = Image.open(filepath)
    img_transposed = ImageOps.exif_transpose(img)
    img_rgb = img_transposed.convert("RGB")
    
    # Logit
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = AI_TRANSFORM(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = AI_MODEL(tensor).item()
        
    # EXIF
    exif = img.getexif() or {}
    make = exif.get(271, "")
    model_tag = exif.get(272, "")
    software = exif.get(305, "")
    has_camera = bool(make or model_tag)
    
    ai_software_tags = ["midjourney", "stable diffusion", "dall-e", "firefly", "craiyon", "wombo", "artbreeder", "bing image", "adobe firefly", "generative fill"]
    software_detected = any(t in str(software).lower() for t in ai_software_tags)
    
    # Laplacian Variance
    img_gray = np.array(img_transposed.convert("L"))
    h, w = img_gray.shape
    if h > 500 or w > 500:
        img_gray_lap = cv2.resize(img_gray, (500, 500))
    else:
        img_gray_lap = img_gray
    laplacian = cv2.Laplacian(img_gray_lap, cv2.CV_64F)
    lap_var = float(np.var(laplacian))
    
    # FFT
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
    
    # Blockiness
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
    # Make sure AI model is loaded and set to eval
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AI_MODEL.to(device)
    AI_MODEL.eval()
    
    # 1. Define Generalization Datasets
    datasets = {}
    
    # (a) Validation Pack
    with open(VAL_MANIFEST, "r") as f:
        manifest = json.load(f)
    val_files = []
    for item in manifest:
        fn = item["filename"]
        lbl = item["label"]
        filepath = os.path.join(VAL_PACK_DIR, "REAL" if lbl == "REAL" else "FAKE", fn)
        if os.path.exists(filepath):
            val_files.append((filepath, 1 if lbl == "FAKE" else 0, item.get("source")))
    datasets["Validation Pack"] = val_files
    
    # (b) Smartphone Photos (Generalization Dataset)
    # We sample 50 from iPhone and 50 from Android corpus
    iphone_dir = os.path.join(DATASET_ROOT, "ai_corpus", "real", "iphone")
    android_dir = os.path.join(DATASET_ROOT, "ai_corpus", "real", "android")
    phone_files = []
    if os.path.exists(iphone_dir) and os.path.exists(android_dir):
        iphones = [os.path.join(iphone_dir, f) for f in os.listdir(iphone_dir) if f.lower().endswith((".jpg", ".jpeg"))]
        androids = [os.path.join(android_dir, f) for f in os.listdir(android_dir) if f.lower().endswith((".jpg", ".jpeg"))]
        sampled_iphones = random.sample(iphones, min(50, len(iphones)))
        sampled_androids = random.sample(androids, min(50, len(androids)))
        for p in sampled_iphones + sampled_androids:
            phone_files.append((p, 0, "SMARTPHONE"))
    datasets["Smartphone Photos"] = phone_files
    
    # (c) DSLR Photos (Generalization Dataset)
    dslr_dir = os.path.join(DATASET_ROOT, "ai_corpus", "real", "dslr")
    dslr_files = []
    if os.path.exists(dslr_dir):
        dslrs = [os.path.join(dslr_dir, f) for f in os.listdir(dslr_dir) if f.lower().endswith((".jpg", ".jpeg"))]
        sampled_dslrs = random.sample(dslrs, min(50, len(dslrs)))
        for p in sampled_dslrs:
            dslr_files.append((p, 0, "DSLR"))
    datasets["DSLR Photos"] = dslr_files
    
    # (d) WhatsApp-compressed images (Generalization Dataset)
    wa_dir = os.path.join(DATASET_ROOT, "ai_corpus", "real", "whatsapp")
    wa_files = []
    if os.path.exists(wa_dir):
        was = [os.path.join(wa_dir, f) for f in os.listdir(wa_dir) if f.lower().endswith((".jpg", ".jpeg"))]
        sampled_was = random.sample(was, min(50, len(was)))
        for p in sampled_was:
            wa_files.append((p, 0, "WHATSAPP"))
    datasets["WhatsApp Images"] = wa_files
    
    # (e) Gmail downloaded images (Generalization Dataset) - simulated by stripping EXIF
    # We sample 50 from general REAL (iPhone, Android, DSLR) and strip metadata
    gmail_files = []
    gmail_temp_dir = os.path.join(BACKEND_DIR, "gmail_simulated")
    os.makedirs(gmail_temp_dir, exist_ok=True)
    
    # Gather potential source files
    source_pool = []
    for p, label, src in (phone_files + dslr_files):
        source_pool.append(p)
    
    if source_pool:
        sampled_gmail_sources = random.sample(source_pool, min(50, len(source_pool)))
        for idx, src_path in enumerate(sampled_gmail_sources):
            # Save stripped version
            try:
                img = Image.open(src_path)
                img = ImageOps.exif_transpose(img)
                # Save without EXIF
                dst_path = os.path.join(gmail_temp_dir, f"gmail_stripped_{idx}.jpg")
                img.convert("RGB").save(dst_path, "JPEG", quality=90)
                gmail_files.append((dst_path, 0, "GMAIL"))
            except Exception as e:
                print(f"Error preparing simulated Gmail image {src_path}: {e}")
    datasets["Gmail Images"] = gmail_files
    
    # (f) Screenshot images (Generalization Dataset)
    ss_dir = os.path.join(DATASET_ROOT, "Screenshot", "pictures")
    ss_files = []
    if os.path.exists(ss_dir):
        sss = [os.path.join(ss_dir, f) for f in os.listdir(ss_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        sampled_sss = random.sample(sss, min(50, len(sss)))
        for p in sampled_sss:
            ss_files.append((p, 0, "SCREENSHOT"))
    datasets["Screenshot Images"] = ss_files
    
    # (g) AI-edited images (Generalization Dataset) - sampled from casia tampered
    edited_dir = os.path.join(DATASET_ROOT, "casia_binary", "tampered")
    edited_files = []
    if os.path.exists(edited_dir):
        tamps = [os.path.join(edited_dir, f) for f in os.listdir(edited_dir) if f.lower().endswith((".jpg", ".jpeg"))]
        sampled_tamps = random.sample(tamps, min(50, len(tamps)))
        for p in sampled_tamps:
            edited_files.append((p, 1, "AI_EDITED"))
    datasets["AI-Edited Images"] = edited_files
    
    # (h) Variants (cropped, resized, and compressed) (Generalization Dataset)
    cropped_dir = os.path.join(DATASET_ROOT, "cropped")
    resized_dir = os.path.join(DATASET_ROOT, "resized")
    compressed_dir = os.path.join(DATASET_ROOT, "compressed")
    variant_files = []
    
    def gather_variants(directory, src_lbl):
        if os.path.exists(directory):
            files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            sampled = random.sample(files, min(20, len(files)))
            # Since these are derived from real originals, we check if they are labeled real (label=0)
            for p in sampled:
                variant_files.append((p, 0, src_lbl))
                
    gather_variants(cropped_dir, "VARIANT_CROP")
    gather_variants(resized_dir, "VARIANT_RESIZE")
    gather_variants(compressed_dir, "VARIANT_COMPRESS")
    datasets["Variants (Crop/Resize/Compress)"] = variant_files
    
    # 2. Run Benchmarking
    all_metrics = {}
    false_positives = []
    false_negatives = []
    
    print("\nStarting Phase 1 Benchmarking across all dataset groups...")
    for ds_name, file_list in datasets.items():
        print(f"Benchmarking '{ds_name}' with {len(file_list)} images...")
        if not file_list:
            print("  Skipping (empty dataset).")
            continue
            
        y_true_all = []
        
        # Baseline V1
        y_pred_base = []
        y_prob_base = []
        
        # Improved V1
        y_pred_imp = []
        y_prob_imp = []
        
        for filepath, y_true, source in file_list:
            try:
                feats = extract_features(filepath)
                
                # Run baseline V1
                prob_base, pred_base = predict_baseline_v1(
                    feats["logit"], feats["has_camera"], feats["lap_var"], feats["num_peaks"], feats["software_detected"]
                )
                
                # Run improved V1
                prob_imp, pred_imp = predict_improved_v1(
                    feats["logit"], feats["has_camera"], feats["lap_var"], feats["num_peaks"], feats["blockiness"], feats["software_detected"]
                )
                
                y_true_all.append(y_true)
                y_pred_base.append(pred_base)
                y_prob_base.append(prob_base)
                y_pred_imp.append(pred_imp)
                y_prob_imp.append(prob_imp)
                
                filename = os.path.basename(filepath)
                # Trace false positives and negatives for Improved V1
                if y_true == 0 and pred_imp == 1:
                    false_positives.append({
                        "dataset": ds_name,
                        "filename": filename,
                        "source": source,
                        "prob": prob_imp,
                        "features": feats
                    })
                elif y_true == 1 and pred_imp == 0:
                    false_negatives.append({
                        "dataset": ds_name,
                        "filename": filename,
                        "source": source,
                        "prob": prob_imp,
                        "features": feats
                    })
                    
            except Exception as e:
                print(f"  Error processing {filepath}: {e}")
                
        y_true_all = np.array(y_true_all)
        y_pred_base = np.array(y_pred_base)
        y_prob_base = np.array(y_prob_base)
        y_pred_imp = np.array(y_pred_imp)
        y_prob_imp = np.array(y_prob_imp)
        
        # Calculate stats
        def get_group_stats(y_true, y_pred, y_prob):
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            acc = float(accuracy_score(y_true, y_pred))
            prec = float(precision_score(y_true, y_pred, zero_division=0))
            rec = float(recall_score(y_true, y_pred, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            
            # FPR
            tn_fp = np.sum(y_true == 0)
            if tn_fp > 0:
                fpr = float(np.sum((y_true == 0) & (y_pred == 1)) / tn_fp)
            else:
                fpr = 0.0
                
            # ECE & Brier
            brier = float(np.mean((y_prob - y_true) ** 2))
            ece = 0.0
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            for i in range(n_bins):
                bin_lower = bin_boundaries[i]
                bin_upper = bin_boundaries[i + 1]
                in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
                prop_in_bin = np.mean(in_bin)
                if prop_in_bin > 0:
                    accuracy_in_bin = np.mean(y_true[in_bin])
                    avg_confidence_in_bin = np.mean(y_prob[in_bin])
                    ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece = float(ece)
            
            return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "fpr": fpr, "ece": ece, "brier": brier}
            
        stats_base = get_group_stats(y_true_all, y_pred_base, y_prob_base)
        stats_imp = get_group_stats(y_true_all, y_pred_imp, y_prob_imp)
        
        all_metrics[ds_name] = {
            "baseline": stats_base,
            "improved": stats_imp
        }
        
    # 3. Save baseline_report.md
    baseline_report_path = os.path.join(BACKEND_DIR, "baseline_report.md")
    with open(baseline_report_path, "w", encoding="utf-8") as f:
        f.write("# TraceLens AI Detector V1 – Baseline Benchmarking Report\n\n")
        f.write("This report presents the performance of the **Original V1 (Legacy)** and **Improved V1 (Logistic Regression)** pipelines on multiple generalization datasets.\n\n")
        for ds_name, metrics in all_metrics.items():
            f.write(f"## Dataset: {ds_name}\n\n")
            f.write("| Pipeline | Accuracy | Precision | Recall | F1 Score | False Positive Rate | ECE | Brier |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            b = metrics["baseline"]
            f.write(f"| Original V1 (Legacy) | {b['accuracy']*100:.2f}% | {b['precision']*100:.2f}% | {b['recall']*100:.2f}% | {b['f1']:.4f} | {b['fpr']*100:.2f}% | {b['ece']:.4f} | {b['brier']:.4f} |\n")
            i = metrics["improved"]
            f.write(f"| Improved V1 (LogReg) | {i['accuracy']*100:.2f}% | {i['precision']*100:.2f}% | {i['recall']*100:.2f}% | {i['f1']:.4f} | {i['fpr']*100:.2f}% | {i['ece']:.4f} | {i['brier']:.4f} |\n\n")
            
    print(f"Saved baseline_report.md to {baseline_report_path}")
    
    # 4. Save false_positive_analysis.md
    fp_analysis_path = os.path.join(BACKEND_DIR, "false_positive_analysis.md")
    with open(fp_analysis_path, "w", encoding="utf-8") as f:
        f.write("# TraceLens AI Detector V1 – False Positive Analysis Report\n\n")
        f.write("This report details every false positive from the Improved V1 pipeline on generalization datasets, including the exact feature triggers.\n\n")
        
        f.write(f"Total False Positives: {len(false_positives)}\n\n")
        
        f.write("| # | Dataset | Filename | Source | Fused Prob | Logit | EXIF | Laplacian Var | FFT Peaks | Blockiness |\n")
        f.write("| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for idx, fp in enumerate(false_positives):
            feats = fp["features"]
            f.write(f"| {idx+1} | {fp['dataset']} | {fp['filename']} | {fp['source']} | {fp['prob']*100:.1f}% | {feats['logit']:.4f} | {feats['has_camera']} | {feats['lap_var']:.2f} | {feats['num_peaks']} | {feats['blockiness']:.4f} |\n")
            
        f.write("\n## Breakdown and Explanations by Cause:\n\n")
        f.write("1. **Negative Logit Bias**: The EfficientNet-B0 model outputs negative logits (e.g., -10 to -30) for clean camera captures due to domain/codec shifts. Since a negative logit indicates high FAKE probability, the base model is severely biased towards FAKE.\n")
        f.write("2. **FFT Peak Anomalies (num_peaks > 15)**: Sharp edge transitions, screenshots containing text, or high-contrast grids introduce periodic high-frequency patterns, creating spurious spikes in the 2D FFT magnitude spectrum and triggering AI grid indicators.\n")
        f.write("3. **Laplacian Variance Warnings (lap_var < 5.0)**: Smooth gradients, dark regions, skies, or heavy compression smoothing reduce high-frequency visual grain, triggering low Laplacian variance warnings.\n")
        f.write("4. **Missing EXIF Metadata**: WhatsApp downloads, screenshots, and stripped files lack make/model tags, causing the EXIF warning to deduct credibility and push predictions towards FAKE.\n")
        
    print(f"Saved false_positive_analysis.md to {fp_analysis_path}")
    
    # 5. Save false_negative_analysis.md
    fn_analysis_path = os.path.join(BACKEND_DIR, "false_negative_analysis.md")
    with open(fn_analysis_path, "w", encoding="utf-8") as f:
        f.write("# TraceLens AI Detector V1 – False Negative Analysis Report\n\n")
        f.write("This report details every false negative from the Improved V1 pipeline on generalization datasets, explaining why they occurred.\n\n")
        
        f.write(f"Total False Negatives: {len(false_negatives)}\n\n")
        
        f.write("| # | Dataset | Filename | Source | Fused Prob | Logit | EXIF | Laplacian Var | FFT Peaks | Blockiness |\n")
        f.write("| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for idx, fn in enumerate(false_negatives):
            feats = fn["features"]
            f.write(f"| {idx+1} | {fn['dataset']} | {fn['filename']} | {fn['source']} | {fn['prob']*100:.1f}% | {feats['logit']:.4f} | {feats['has_camera']} | {feats['lap_var']:.2f} | {feats['num_peaks']} | {feats['blockiness']:.4f} |\n")
            
        f.write("\n## Breakdown and Explanations by Cause:\n\n")
        f.write("1. **High Logits on Spliced Images**: Spliced images containing large regions of real photography preserve the original camera visual patterns, resulting in positive logits from the neural model.\n")
        f.write("2. **Metadata Presence on Spliced Images**: Spliced CASIA images often retain their original EXIF headers, giving them `has_camera = True` and lowering the warning score.\n")
        
    print(f"Saved false_negative_analysis.md to {fn_analysis_path}")
    
    # Cleanup temp directory
    try:
        for f in os.listdir(gmail_temp_dir):
            os.remove(os.path.join(gmail_temp_dir, f))
        os.rmdir(gmail_temp_dir)
    except Exception:
        pass

if __name__ == "__main__":
    main()
