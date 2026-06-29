import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import timm

project_root = r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI"
v2_dir = os.path.join(project_root, "backend", "ml", "v2")
val_pack_dir = os.path.join(v2_dir, "validation_pack")
val_manifest_path = os.path.join(val_pack_dir, "validation_manifest.json")
v2_model_path = os.path.join(v2_dir, "ai_detector_v2.pth")

# Transforms
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
transforms_v2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def main():
    # Load manifest
    with open(val_manifest_path, "r") as f:
        manifest = json.load(f)
        
    # Load model
    model = timm.create_model("convnext_tiny", pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(v2_model_path, map_location="cpu"))
    model.eval()
    
    samples = []
    
    print("Evaluating validation pack...")
    for idx, item in enumerate(manifest):
        fn = item["filename"]
        label_str = item["label"]
        source = item["source"]
        orig_path = item["original_filepath"]
        
        filepath = os.path.join(val_pack_dir, label_str, fn)
        if not os.path.exists(filepath):
            filepath = os.path.join(val_pack_dir, "REAL" if label_str == "REAL" else "FAKE", fn)
            if not os.path.exists(filepath):
                continue
                
        y_true = 1 if label_str == "FAKE" else 0
        
        try:
            with Image.open(filepath) as img:
                img_rgb = img.convert("RGB")
                t_img = transforms_v2(img_rgb).unsqueeze(0)
                with torch.no_grad():
                    logit = model(t_img).item()
                    prob_real = torch.sigmoid(torch.tensor(logit)).item()
                    prob_fake = 1.0 - prob_real
                    pred = 1 if prob_fake >= 0.5 else 0
            
            samples.append({
                "filename": fn,
                "label": y_true,
                "source": source,
                "original_filepath": orig_path,
                "prob_fake": prob_fake,
                "pred": pred
            })
        except Exception as e:
            print(f"Error reading {fn}: {e}")
            
    print(f"Completed evaluation for {len(samples)} images.")
    
    # 1. Confusion Matrix
    # y_true: rows, y_pred: columns
    # 0 = REAL, 1 = FAKE
    tn, fp, fn, tp = 0, 0, 0, 0
    for s in samples:
        if s["label"] == 0 and s["pred"] == 0:
            tn += 1
        elif s["label"] == 0 and s["pred"] == 1:
            fp += 1
        elif s["label"] == 1 and s["pred"] == 0:
            fn += 1
        elif s["label"] == 1 and s["pred"] == 1:
            tp += 1
            
    print("\n=== CONFUSION MATRIX ===")
    print(f"TN (Actual REAL, Pred REAL): {tn}")
    print(f"FP (Actual REAL, Pred FAKE): {fp} (False Positives)")
    print(f"FN (Actual FAKE, Pred REAL): {fn} (False Negatives)")
    print(f"TP (Actual FAKE, Pred FAKE): {tp}")
    
    # 2. Per-category accuracy
    # Accuracy = correct / total
    categories = sorted(list({s["source"] for s in samples}))
    print("\n=== PER-CATEGORY ACCURACY ===")
    category_stats = {}
    for cat in categories:
        cat_samples = [s for s in samples if s["source"] == cat]
        correct = sum(1 for s in cat_samples if s["label"] == s["pred"])
        total = len(cat_samples)
        acc = correct / total if total > 0 else 0.0
        
        # Calculate FPR or FNR depending on category
        cat_label = cat_samples[0]["label"]
        rate_str = ""
        if cat_label == 0: # REAL category
            cat_fp = sum(1 for s in cat_samples if s["pred"] == 1)
            cat_fpr = cat_fp / total if total > 0 else 0.0
            rate_str = f"FPR: {cat_fpr*100:.2f}%"
        else: # FAKE category
            cat_fn = sum(1 for s in cat_samples if s["pred"] == 0)
            cat_fnr = cat_fn / total if total > 0 else 0.0
            rate_str = f"FNR: {cat_fnr*100:.2f}%"
            
        print(f"{cat:<12} | Accuracy: {acc*100:6.2f}% ({correct}/{total}) | {rate_str}")
        category_stats[cat] = {
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "rate_str": rate_str
        }
        
    # 3. Top 50 false positives (labeled 0, predicted 1)
    false_positives = [s for s in samples if s["label"] == 0 and s["pred"] == 1]
    # Sort by probability of FAKE descending
    false_positives.sort(key=lambda x: x["prob_fake"], reverse=True)
    
    # 4. Top 50 false negatives (labeled 1, predicted 0)
    false_negatives = [s for s in samples if s["label"] == 1 and s["pred"] == 0]
    # Sort by probability of FAKE ascending
    false_negatives.sort(key=lambda x: x["prob_fake"])
    
    # Write to a JSON file for easy parsing in the report
    analysis_results = {
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "per_category_accuracy": {k: {"accuracy": v["accuracy"], "correct": v["correct"], "total": v["total"]} for k, v in category_stats.items()},
        "top_50_false_positives": [
            {
                "filename": s["filename"],
                "prob_fake": s["prob_fake"],
                "source": s["source"],
                "original_filepath": s["original_filepath"]
            } for s in false_positives[:50]
        ],
        "top_50_false_negatives": [
            {
                "filename": s["filename"],
                "prob_fake": s["prob_fake"],
                "source": s["source"],
                "original_filepath": s["original_filepath"]
            } for s in false_negatives[:50]
        ]
    }
    
    with open(os.path.join(v2_dir, "forensic_analysis_results.json"), "w") as out_f:
        json.dump(analysis_results, out_f, indent=4)
        
    print(f"\nWritten forensic analysis results to {os.path.join(v2_dir, 'forensic_analysis_results.json')}")
    print(f"Found {len(false_positives)} False Positives total. Top 50 extracted.")
    print(f"Found {len(false_negatives)} False Negatives total. Top 50 extracted.")

    # 5. Verify mislabeling (e.g. CASIA tampered in AI-generated categories)
    print("\n=== VERIFYING MISLABELING ===")
    ai_categories = ["MIDJOURNEY", "FLUX", "SDXL", "CHATGPT"]
    casia_count = 0
    non_casia_count = 0
    for s in samples:
        if s["source"] in ai_categories:
            if "casia_binary" in s["original_filepath"] or "tampered" in s["original_filepath"]:
                casia_count += 1
            else:
                non_casia_count += 1
                
    print(f"Total images labeled as AI-generated: {casia_count + non_casia_count}")
    print(f"  - Sourced from CASIA (Splicing/Copy-Move): {casia_count} ({casia_count / (casia_count + non_casia_count) * 100:.2f}%)")
    print(f"  - Sourced from actual AI-generated sets: {non_casia_count} ({non_casia_count / (casia_count + non_casia_count) * 100:.2f}%)")

if __name__ == "__main__":
    main()
