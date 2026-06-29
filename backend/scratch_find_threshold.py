import os
import json
import numpy as np

project_root = r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI"
report_path = os.path.join(project_root, "backend", "ml", "v2", "evaluation_report.json")

# Since the detailed file-level probabilities are not fully in evaluation_report.json,
# let's reload them from a run of inference on the validation pack,
# or write a script that does the evaluation with varying thresholds.
# Wait, let's write a script that parses the results and finds the best threshold.

import torch
from PIL import Image
from torchvision import transforms
import timm

V2_DIR = os.path.join(project_root, "backend", "ml", "v2")
VAL_PACK_DIR = os.path.join(V2_DIR, "validation_pack")
VAL_MANIFEST = os.path.join(VAL_PACK_DIR, "validation_manifest.json")
V2_MODEL_PATH = os.path.join(V2_DIR, "ai_detector_v2.pth")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
transforms_v2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def main():
    with open(VAL_MANIFEST, 'r') as f:
        manifest = json.load(f)
        
    model_v2 = timm.create_model("convnext_tiny", pretrained=False, num_classes=1)
    model_v2.load_state_dict(torch.load(V2_MODEL_PATH, map_location="cpu"))
    model_v2.eval()
    
    probs = []
    labels = []
    sources = []
    
    print("Running inference to collect raw probabilities...")
    for item in manifest:
        fn = item["filename"]
        label_str = item["label"]
        source = item["source"]
        
        filepath = os.path.join(VAL_PACK_DIR, label_str, fn)
        if not os.path.exists(filepath):
            filepath = os.path.join(VAL_PACK_DIR, "REAL" if label_str == "REAL" else "FAKE", fn)
            if not os.path.exists(filepath):
                continue
                
        y_true = 1 if label_str == "FAKE" else 0
        
        try:
            with Image.open(filepath) as img:
                img_rgb = img.convert("RGB")
                t_v2 = transforms_v2(img_rgb).unsqueeze(0)
                with torch.no_grad():
                    out_v2 = model_v2(t_v2).item()
                    prob_real = torch.sigmoid(torch.tensor(out_v2)).item()
                    prob_fake = 1.0 - prob_real
            probs.append(prob_fake)
            labels.append(y_true)
            sources.append(source)
        except Exception as e:
            pass

    probs = np.array(probs)
    labels = np.array(labels)
    sources = np.array(sources)
    
    print(f"Collected {len(probs)} sample outputs.")
    
    # Try various thresholds
    best_t = None
    best_score = -1
    
    print("\nAnalyzing thresholds:")
    print(f"{'Threshold':<10} | {'iPhone FPR':<12} | {'Android FPR':<12} | {'DSLR FPR':<12} | {'AI Recall':<12}")
    print("-" * 65)
    
    for t in np.linspace(0.01, 0.99, 99):
        preds = (probs >= t).astype(int)
        
        # Calculate FPRs
        iphone_mask = (sources == "IPHONE") & (labels == 0)
        iphone_fpr = np.mean(preds[iphone_mask]) if np.sum(iphone_mask) > 0 else 0.0
        
        android_mask = (sources == "ANDROID") & (labels == 0)
        android_fpr = np.mean(preds[android_mask]) if np.sum(android_mask) > 0 else 0.0
        
        dslr_mask = (sources == "DSLR") & (labels == 0)
        dslr_fpr = np.mean(preds[dslr_mask]) if np.sum(dslr_mask) > 0 else 0.0
        
        # Calculate Recall
        fake_mask = (labels == 1)
        recall = np.mean(preds[fake_mask]) if np.sum(fake_mask) > 0 else 0.0
        
        if t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] or (iphone_fpr < 0.1 and android_fpr < 0.1 and dslr_fpr < 0.1):
            print(f"{t:.2f}       | {iphone_fpr*100:9.2f}% | {android_fpr*100:9.2f}% | {dslr_fpr*100:9.2f}% | {recall*100:9.2f}%")
            
        # Check if constraints met
        if iphone_fpr < 0.10 and android_fpr < 0.10 and dslr_fpr < 0.10 and recall >= 0.85:
            score = recall - (iphone_fpr + android_fpr + dslr_fpr)
            if score > best_score:
                best_score = score
                best_t = t
                
    if best_t is not None:
        print(f"\nFound a threshold that satisfies ALL conditions! Threshold: {best_t:.4f}")
    else:
        print("\nNo single threshold satisfies all target conditions simultaneously.")
        
        # Let's find the threshold that minimizes FPRs while keeping Recall as high as possible
        print("\nLet's check the trade-offs:")
        for target_recall in [0.85, 0.80, 0.75, 0.70, 0.65]:
            best_t_for_recall = None
            min_fpr = 999
            for t in np.linspace(0.01, 0.99, 99):
                preds = (probs >= t).astype(int)
                iphone_mask = (sources == "IPHONE") & (labels == 0)
                iphone_fpr = np.mean(preds[iphone_mask]) if np.sum(iphone_mask) > 0 else 0.0
                
                android_mask = (sources == "ANDROID") & (labels == 0)
                android_fpr = np.mean(preds[android_mask]) if np.sum(android_mask) > 0 else 0.0
                
                dslr_mask = (sources == "DSLR") & (labels == 0)
                dslr_fpr = np.mean(preds[dslr_mask]) if np.sum(dslr_mask) > 0 else 0.0
                
                fake_mask = (labels == 1)
                recall = np.mean(preds[fake_mask]) if np.sum(fake_mask) > 0 else 0.0
                
                avg_fpr = (iphone_fpr + android_fpr + dslr_fpr) / 3.0
                if recall >= target_recall and avg_fpr < min_fpr:
                    min_fpr = avg_fpr
                    best_t_for_recall = t
            if best_t_for_recall is not None:
                preds = (probs >= best_t_for_recall).astype(int)
                iphone_fpr = np.mean(preds[(sources == "IPHONE") & (labels == 0)])
                android_fpr = np.mean(preds[(sources == "ANDROID") & (labels == 0)])
                dslr_fpr = np.mean(preds[(sources == "DSLR") & (labels == 0)])
                recall = np.mean(preds[labels == 1])
                print(f"For Recall >= {target_recall*100:.1f}%, best threshold {best_t_for_recall:.2f} yields:")
                print(f"  iPhone FPR: {iphone_fpr*100:.1f}%, Android FPR: {android_fpr*100:.1f}%, DSLR FPR: {dslr_fpr*100:.1f}%")

if __name__ == '__main__':
    main()
