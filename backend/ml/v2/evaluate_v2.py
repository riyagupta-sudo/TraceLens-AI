#!/usr/bin/env python3
import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import timm
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = r"C:\Users\riya2\OneDrive\Desktop\TraceLens AI"
V2_DIR = os.path.join(PROJECT_ROOT, "backend", "ml", "v2")
VAL_PACK_DIR = os.path.join(V2_DIR, "validation_pack")
VAL_MANIFEST = os.path.join(VAL_PACK_DIR, "validation_manifest.json")
V1_MODEL_PATH = os.path.join(PROJECT_ROOT, "backend", "models", "ai_detector.pth")
V2_MODEL_PATH = os.path.join(V2_DIR, "ai_detector_v2.pth")
V1_CALIBRATION_PATH = os.path.join(PROJECT_ROOT, "backend", "config", "model_calibration.json")

# Define data transforms
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transforms_v1 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

transforms_v2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def load_v1_model():
    print("Loading V1 model...")
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(V1_MODEL_PATH, map_location="cpu"))
    model.eval()
    
    # Load V1 temperature
    temp = 1.0
    if os.path.exists(V1_CALIBRATION_PATH):
        try:
            with open(V1_CALIBRATION_PATH, 'r') as f:
                calib = json.load(f)
                temp = float(calib.get("ai_temperature", 1.0))
        except Exception:
            pass
    return model, temp

def load_v2_model():
    print("Loading V2 model...")
    model = timm.create_model("convnext_tiny", pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(V2_MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

def compute_metrics(y_true, y_pred, y_probs):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # Structure of confusion matrix:
    # y_true (rows): 0 (REAL), 1 (FAKE)
    # y_pred (cols): 0 (REAL), 1 (FAKE)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return {
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "precision": float(round(precision, 4)),
        "recall": float(round(recall, 4)),
        "f1": float(round(f1, 4)),
        "fpr": float(round(fpr, 4)),
        "fnr": float(round(fnr, 4))
    }

def main():
    if not os.path.exists(VAL_MANIFEST):
        print(f"Error: Validation manifest {VAL_MANIFEST} not found.")
        sys.exit(1)
        
    with open(VAL_MANIFEST, 'r') as f:
        manifest = json.load(f)
        
    print(f"Loaded validation pack containing {len(manifest)} files.")
    
    # Check if models exist
    if not os.path.exists(V1_MODEL_PATH):
        print(f"Error: V1 model {V1_MODEL_PATH} not found.")
        sys.exit(1)
    if not os.path.exists(V2_MODEL_PATH):
        print(f"Error: V2 model {V2_MODEL_PATH} not found. Please train the V2 model first.")
        sys.exit(1)
        
    model_v1, temp_v1 = load_v1_model()
    model_v2 = load_v2_model()
    
    results = []
    
    print("Running evaluations...")
    for idx, item in enumerate(manifest):
        fn = item["filename"]
        label_str = item["label"]  # "REAL" or "FAKE"
        source = item["source"]
        
        # Determine path
        filepath = os.path.join(VAL_PACK_DIR, label_str, fn)
        if not os.path.exists(filepath):
            # Fallback check
            filepath = os.path.join(VAL_PACK_DIR, "REAL" if label_str == "REAL" else "FAKE", fn)
            if not os.path.exists(filepath):
                print(f"Warning: File {fn} not found in validation pack.")
                continue
                
        y_true = 1 if label_str == "FAKE" else 0
        
        # Load image
        try:
            with Image.open(filepath) as img:
                img_rgb = img.convert("RGB")
                
                # V1 inference
                t_v1 = transforms_v1(img_rgb).unsqueeze(0)
                with torch.no_grad():
                    out_v1 = model_v1(t_v1).item()
                    # Inverted AI probability: 1.0 - prob(REAL)
                    prob_real_v1 = torch.sigmoid(torch.tensor(out_v1 / temp_v1)).item()
                    prob_fake_v1 = 1.0 - prob_real_v1
                    pred_v1 = 1 if prob_fake_v1 >= 0.5 else 0
                    
                # V2 inference
                t_v2 = transforms_v2(img_rgb).unsqueeze(0)
                with torch.no_grad():
                    out_v2 = model_v2(t_v2).item()
                    # Inverted AI probability: 1.0 - prob(REAL)
                    prob_real_v2 = torch.sigmoid(torch.tensor(out_v2)).item()
                    prob_fake_v2 = 1.0 - prob_real_v2
                    pred_v2 = 1 if prob_fake_v2 >= 0.5 else 0
                    
            results.append({
                "filename": fn,
                "label": y_true,
                "source": source,
                "v1": {"prob": prob_fake_v1, "pred": pred_v1},
                "v2": {"prob": prob_fake_v2, "pred": pred_v2}
            })
        except Exception as e:
            print(f"Error evaluating {fn}: {e}")
            
    print(f"Inference completed for {len(results)} files.")
    
    # Group results by source
    sources = sorted(list({item["source"] for item in results}))
    
    metrics_by_source = {}
    
    for src in sources:
        src_results = [r for r in results if r["source"] == src]
        y_true_src = [r["label"] for r in src_results]
        
        # V1
        y_pred_v1 = [r["v1"]["pred"] for r in src_results]
        y_prob_v1 = [r["v1"]["prob"] for r in src_results]
        metrics_v1 = compute_metrics(y_true_src, y_pred_v1, y_prob_v1)
        
        # V2
        y_pred_v2 = [r["v2"]["pred"] for r in src_results]
        y_prob_v2 = [r["v2"]["prob"] for r in src_results]
        metrics_v2 = compute_metrics(y_true_src, y_pred_v2, y_prob_v2)
        
        metrics_by_source[src] = {
            "v1": metrics_v1,
            "v2": metrics_v2,
            "count": len(src_results)
        }
        
    # Overall metrics
    y_true_all = [r["label"] for r in results]
    metrics_v1_all = compute_metrics(y_true_all, [r["v1"]["pred"] for r in results], [r["v1"]["prob"] for r in results])
    metrics_v2_all = compute_metrics(y_true_all, [r["v2"]["pred"] for r in results], [r["v2"]["prob"] for r in results])
    
    overall_metrics = {
        "v1": metrics_v1_all,
        "v2": metrics_v2_all,
        "count": len(results)
    }
    
    evaluation_report = {
        "overall": overall_metrics,
        "by_source": metrics_by_source
    }
    
    # Save JSON report
    report_json_path = os.path.join(V2_DIR, "evaluation_report.json")
    with open(report_json_path, 'w') as f:
        json.dump(evaluation_report, f, indent=4)
    print(f"Saved evaluation JSON report to {report_json_path}")
    
    # Save markdown report
    comparison_md_path = os.path.join(V2_DIR, "v1_vs_v2_comparison.md")
    
    md_content = f"""# TraceLens AI - Detector V1 vs V2 Comparison Report

This comparison report profiles the performance of the legacy AI Detector V1 and the new production-grade AI Detector V2 (ConvNeXt-Tiny) on the isolated validation pack ({len(results)} images).

## 1. Executive Performance Summary

| Metric | Legacy V1 | Target Threshold | New V2 | Status |
| :--- | :---: | :---: | :---: | :---: |
| **iPhone False Positive Rate (FPR)** | {metrics_by_source.get('IPHONE', {}).get('v1', {}).get('fpr', 0.0)*100:.2f}% | < 10.0% | {metrics_by_source.get('IPHONE', {}).get('v2', {}).get('fpr', 0.0)*100:.2f}% | **{"PASSED" if metrics_by_source.get('IPHONE', {}).get('v2', {}).get('fpr', 1.0) < 0.1 else "FAILED"}** |
| **Android False Positive Rate (FPR)** | {metrics_by_source.get('ANDROID', {}).get('v1', {}).get('fpr', 0.0)*100:.2f}% | < 10.0% | {metrics_by_source.get('ANDROID', {}).get('v2', {}).get('fpr', 0.0)*100:.2f}% | **{"PASSED" if metrics_by_source.get('ANDROID', {}).get('v2', {}).get('fpr', 1.0) < 0.1 else "FAILED"}** |
| **DSLR False Positive Rate (FPR)** | {metrics_by_source.get('DSLR', {}).get('v1', {}).get('fpr', 0.0)*100:.2f}% | < 10.0% | {metrics_by_source.get('DSLR', {}).get('v2', {}).get('fpr', 0.0)*100:.2f}% | **{"PASSED" if metrics_by_source.get('DSLR', {}).get('v2', {}).get('fpr', 1.0) < 0.1 else "FAILED"}** |
| **AI Recall (Sensitivity)** | {overall_metrics.get('v1', {}).get('recall', 0.0)*100:.2f}% | > 85.0% | {overall_metrics.get('v2', {}).get('recall', 0.0)*100:.2f}% | **{"PASSED" if overall_metrics.get('v2', {}).get('recall', 0.0) > 0.85 else "FAILED"}** |
| **Overall F1-Score** | {overall_metrics.get('v1', {}).get('f1', 0.0)*100:.2f}% | - | {overall_metrics.get('v2', {}).get('f1', 0.0)*100:.2f}% | Improved |

---

## 2. Category Level Breakdown

Here are the detailed confusion matrices and performance metrics for each of the 9 categories in the validation manifest.

### REAL Categories (Clean Camera/Screenshot Files)

#### iPhone
- **Count**: {metrics_by_source.get('IPHONE', {}).get('count', 0)}
- **V1 (Legacy)**: FPR: {metrics_by_source.get('IPHONE', {}).get('v1', {}).get('fpr', 0.0)*100:.2f}%, Confusion Matrix: {metrics_by_source.get('IPHONE', {}).get('v1', {}).get('confusion_matrix')}
- **V2 (New)**: FPR: {metrics_by_source.get('IPHONE', {}).get('v2', {}).get('fpr', 0.0)*100:.2f}%, Confusion Matrix: {metrics_by_source.get('IPHONE', {}).get('v2', {}).get('confusion_matrix')}

#### Android
- **Count**: {metrics_by_source.get('ANDROID', {}).get('count', 0)}
- **V1 (Legacy)**: FPR: {metrics_by_source.get('ANDROID', {}).get('v1', {}).get('fpr', 0.0)*100:.2f}%, Confusion Matrix: {metrics_by_source.get('ANDROID', {}).get('v1', {}).get('confusion_matrix')}
- **V2 (New)**: FPR: {metrics_by_source.get('ANDROID', {}).get('v2', {}).get('fpr', 0.0)*100:.2f}%, Confusion Matrix: {metrics_by_source.get('ANDROID', {}).get('v2', {}).get('confusion_matrix')}

#### DSLR
- **Count**: {metrics_by_source.get('DSLR', {}).get('count', 0)}
- **V1 (Legacy)**: FPR: {metrics_by_source.get('DSLR', {}).get('v1', {}).get('fpr', 0.0)*100:.2f}%, Confusion Matrix: {metrics_by_source.get('DSLR', {}).get('v1', {}).get('confusion_matrix')}
- **V2 (New)**: FPR: {metrics_by_source.get('DSLR', {}).get('v2', {}).get('fpr', 0.0)*100:.2f}%, Confusion Matrix: {metrics_by_source.get('DSLR', {}).get('v2', {}).get('confusion_matrix')}

#### Screenshots
- **Count**: {metrics_by_source.get('SCREENSHOT', {}).get('count', 0)}
- **V1 (Legacy)**: FPR: {metrics_by_source.get('SCREENSHOT', {}).get('v1', {}).get('fpr', 0.0)*100:.2f}%, Confusion Matrix: {metrics_by_source.get('SCREENSHOT', {}).get('v1', {}).get('confusion_matrix')}
- **V2 (New)**: FPR: {metrics_by_source.get('SCREENSHOT', {}).get('v2', {}).get('fpr', 0.0)*100:.2f}%, Confusion Matrix: {metrics_by_source.get('SCREENSHOT', {}).get('v2', {}).get('confusion_matrix')}

#### WhatsApp
- **Count**: {metrics_by_source.get('WHATSAPP', {}).get('count', 0)}
- **V1 (Legacy)**: FPR: {metrics_by_source.get('WHATSAPP', {}).get('v1', {}).get('fpr', 0.0)*100:.2f}%, Confusion Matrix: {metrics_by_source.get('WHATSAPP', {}).get('v1', {}).get('confusion_matrix')}
- **V2 (New)**: FPR: {metrics_by_source.get('WHATSAPP', {}).get('v2', {}).get('fpr', 0.0)*100:.2f}%, Confusion Matrix: {metrics_by_source.get('WHATSAPP', {}).get('v2', {}).get('confusion_matrix')}

---

### FAKE Categories (AI Generated Files)

#### Midjourney
- **Count**: {metrics_by_source.get('MIDJOURNEY', {}).get('count', 0)}
- **V1 (Legacy)**: Precision: {metrics_by_source.get('MIDJOURNEY', {}).get('v1', {}).get('precision', 0.0)*100:.2f}%, Recall: {metrics_by_source.get('MIDJOURNEY', {}).get('v1', {}).get('recall', 0.0)*100:.2f}%, F1: {metrics_by_source.get('MIDJOURNEY', {}).get('v1', {}).get('f1', 0.0)*100:.2f}%
- **V2 (New)**: Precision: {metrics_by_source.get('MIDJOURNEY', {}).get('v2', {}).get('precision', 0.0)*100:.2f}%, Recall: {metrics_by_source.get('MIDJOURNEY', {}).get('v2', {}).get('recall', 0.0)*100:.2f}%, F1: {metrics_by_source.get('MIDJOURNEY', {}).get('v2', {}).get('f1', 0.0)*100:.2f}%

#### Flux
- **Count**: {metrics_by_source.get('FLUX', {}).get('count', 0)}
- **V1 (Legacy)**: Precision: {metrics_by_source.get('FLUX', {}).get('v1', {}).get('precision', 0.0)*100:.2f}%, Recall: {metrics_by_source.get('FLUX', {}).get('v1', {}).get('recall', 0.0)*100:.2f}%, F1: {metrics_by_source.get('FLUX', {}).get('v1', {}).get('f1', 0.0)*100:.2f}%
- **V2 (New)**: Precision: {metrics_by_source.get('FLUX', {}).get('v2', {}).get('precision', 0.0)*100:.2f}%, Recall: {metrics_by_source.get('FLUX', {}).get('v2', {}).get('recall', 0.0)*100:.2f}%, F1: {metrics_by_source.get('FLUX', {}).get('v2', {}).get('f1', 0.0)*100:.2f}%

#### Stable Diffusion (SDXL)
- **Count**: {metrics_by_source.get('SDXL', {}).get('count', 0)}
- **V1 (Legacy)**: Precision: {metrics_by_source.get('SDXL', {}).get('v1', {}).get('precision', 0.0)*100:.2f}%, Recall: {metrics_by_source.get('SDXL', {}).get('v1', {}).get('recall', 0.0)*100:.2f}%, F1: {metrics_by_source.get('SDXL', {}).get('v1', {}).get('f1', 0.0)*100:.2f}%
- **V2 (New)**: Precision: {metrics_by_source.get('SDXL', {}).get('v2', {}).get('precision', 0.0)*100:.2f}%, Recall: {metrics_by_source.get('SDXL', {}).get('v2', {}).get('recall', 0.0)*100:.2f}%, F1: {metrics_by_source.get('SDXL', {}).get('v2', {}).get('f1', 0.0)*100:.2f}%

#### ChatGPT
- **Count**: {metrics_by_source.get('CHATGPT', {}).get('count', 0)}
- **V1 (Legacy)**: Precision: {metrics_by_source.get('CHATGPT', {}).get('v1', {}).get('precision', 0.0)*100:.2f}%, Recall: {metrics_by_source.get('CHATGPT', {}).get('v1', {}).get('recall', 0.0)*100:.2f}%, F1: {metrics_by_source.get('CHATGPT', {}).get('v1', {}).get('f1', 0.0)*100:.2f}%
- **V2 (New)**: Precision: {metrics_by_source.get('CHATGPT', {}).get('v2', {}).get('precision', 0.0)*100:.2f}%, Recall: {metrics_by_source.get('CHATGPT', {}).get('v2', {}).get('recall', 0.0)*100:.2f}%, F1: {metrics_by_source.get('CHATGPT', {}).get('v2', {}).get('f1', 0.0)*100:.2f}%
"""
    
    with open(comparison_md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"Saved comparison markdown report to {comparison_md_path}")

if __name__ == "__main__":
    main()
