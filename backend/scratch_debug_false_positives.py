import os
import sys
import json
import torch
import numpy as np
from PIL import Image, ImageOps
import cv2

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

from app.dna_engine import (
    estimate_compression_artifacts, 
    AI_MODEL, 
    AI_TRANSFORM, 
    V1_TEMP, 
    W_LOGIT, 
    W_EXIF, 
    W_NOISE, 
    W_FFT, 
    W_BLOCK, 
    LR_INTERCEPT,
    V1_THRESHOLD
)

VAL_PACK_DIR = os.path.join(BACKEND_DIR, "ml", "v2", "validation_pack")
VAL_MANIFEST = os.path.join(VAL_PACK_DIR, "validation_manifest.json")

def main():
    with open(VAL_MANIFEST, "r") as f:
        manifest = json.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if AI_MODEL is not None:
        AI_MODEL.to(device)
        AI_MODEL.eval()
        
    results = []
    
    for item in manifest:
        fn = item["filename"]
        label_str = item["label"]
        source = item["source"]
        
        filepath = os.path.join(VAL_PACK_DIR, "REAL" if label_str == "REAL" else "FAKE", fn)
        if not os.path.exists(filepath):
            continue
            
        y_true = 1 if label_str == "FAKE" else 0
        if y_true != 0: # only look at REAL images (false positives)
            continue
            
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
            
            # calculate warnings
            exif_warning = 1.0 if (not has_camera or software_detected) else 0.0
            noise_warning = 1.0 if lap_var < 5.0 else 0.0
            fft_warning = 1.0 if num_peaks > 15 else 0.0
            blockiness_warning = 1.0 if blockiness < 0.9 else 0.0
            
            # current improved formulation
            z = (W_LOGIT * (logit / V1_TEMP) + 
                 W_EXIF * exif_warning + 
                 W_NOISE * noise_warning + 
                 W_FFT * fft_warning + 
                 W_BLOCK * blockiness_warning + 
                 LR_INTERCEPT)
            fused_prob = 1.0 / (1.0 + np.exp(-z))
            
            # original formulation
            orig_raw_probs = 1.0 - (1.0 / (1.0 + np.exp(-logit / 8.0)))
            # Heuristic contributions (adjustments)
            fft_contrib = 10 if (num_peaks > 15) else 0
            lap_contrib = 10 if (lap_var < 5.0) else 0
            if software_detected:
                meta_contrib = 80
            elif has_camera:
                meta_contrib = -10
            else:
                meta_contrib = 10
            orig_fused = (orig_raw_probs * 100) + fft_contrib + lap_contrib + meta_contrib
            orig_fused_prob = min(98, max(2, orig_fused)) / 100.0
            
            results.append({
                "filename": fn,
                "source": source,
                "logit": logit,
                "exif_warning": exif_warning,
                "noise_warning": noise_warning,
                "fft_warning": fft_warning,
                "blockiness_warning": blockiness_warning,
                "lap_var": lap_var,
                "num_peaks": num_peaks,
                "blockiness": blockiness,
                "orig_raw_prob": orig_raw_probs,
                "orig_fused_prob": orig_fused_prob,
                "improved_fused_prob": fused_prob,
                "has_camera": has_camera,
                "make": make,
                "model": model_tag,
                "software": software
            })
            
        except Exception as e:
            print(f"Error processing {fn}: {e}")
            
    # Find false positives for original pipeline (threshold >= 0.5)
    orig_fps = [r for r in results if r["orig_fused_prob"] >= 0.5]
    print(f"\nOriginal Pipeline Total False Positives: {len(orig_fps)} / 175")
    
    # Find false positives for improved pipeline (threshold >= 0.42)
    imp_fps = [r for r in results if r["improved_fused_prob"] >= V1_THRESHOLD]
    print(f"Improved Pipeline Total False Positives: {len(imp_fps)} / 175")
    
    # Sort improved false positives by confidence (highest fused probability first)
    imp_fps_sorted = sorted(imp_fps, key=lambda x: x["improved_fused_prob"], reverse=True)
    
    print("\nTop 20 False Positives in Improved Pipeline:")
    for idx, fp in enumerate(imp_fps_sorted[:20]):
        print(f"{idx+1}. {fp['filename']} ({fp['source']}):")
        print(f"   Improved Fused Prob: {fp['improved_fused_prob']:.4f}, Orig Fused Prob: {fp['orig_fused_prob']:.4f}")
        print(f"   Logit: {fp['logit']:.4f}, EXIF: {fp['has_camera']} (Make: '{fp['make']}', Model: '{fp['model']}', Sw: '{fp['software']}')")
        print(f"   Laplacian Var: {fp['lap_var']:.2f} (Warning: {fp['noise_warning']})")
        print(f"   FFT Peaks: {fp['num_peaks']} (Warning: {fp['fft_warning']})")
        print(f"   Blockiness: {fp['blockiness']:.4f} (Warning: {fp['blockiness_warning']})")
        print("-" * 50)

if __name__ == "__main__":
    main()
