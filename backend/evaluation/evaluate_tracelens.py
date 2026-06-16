import os
import sys
import argparse
import time
import json
import csv
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set paths to ensure backend import works
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.dna_engine import (
    compute_image_hashes,
    extract_metadata_signature,
    calculate_integrity_and_risk,
    get_clip_embedding
)

def parse_args():
    parser = argparse.ArgumentParser(description="TraceLens forensic validation against CASIA 2.0")
    parser.add_argument(
        "--dataset-dir", 
        type=str, 
        default=None,
        help="Path to CASIA2 dataset containing Au and Tp folders"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=100,
        help="Max images to evaluate per category (100: debug, 1000: validation, 0: full benchmark)"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=4,
        help="Number of threads for parallel image analysis"
    )
    return parser.parse_args()

def resolve_paths(dataset_dir):
    # Search common folders for CASIA2
    options = [
        "../dataset/CASIA2",
        "dataset/CASIA2",
        "../../dataset/CASIA2"
    ]
    if dataset_dir:
        options.insert(0, dataset_dir)
        
    for opt in options:
        abs_opt = os.path.abspath(opt)
        au_dir = os.path.join(abs_opt, "Au")
        tp_dir = os.path.join(abs_opt, "Tp")
        if os.path.exists(au_dir) and os.path.exists(tp_dir):
            print(f"Found CASIA 2.0 dataset at: {abs_opt}")
            return abs_opt, au_dir, tp_dir
            
    print("ERROR: CASIA2 dataset directories ('Au' and 'Tp') not found.")
    print("Please download/extract CASIA 2.0 and point to it with --dataset-dir <path>")
    sys.exit(1)

def scan_images(folder):
    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    files = []
    for f in os.listdir(folder):
        ext = os.path.splitext(f)[1].lower()
        if ext in valid_exts:
            files.append(os.path.join(folder, f))
    files.sort()
    return files

def process_single_image(filepath, label):
    try:
        # Run standard TraceLens pipeline steps
        phash, dhash, ahash = compute_image_hashes(filepath)
        metadata = extract_metadata_signature(filepath)
        
        # Inject embeddings and hashes for diagnostics consistency
        metadata["embedding"] = get_clip_embedding(filepath)
        metadata["sha256"] = "dummy_sha"
        metadata["dhash"] = dhash
        metadata["ahash"] = ahash
        
        integrity, risk, forensics = calculate_integrity_and_risk(
            filepath, metadata, "image/jpeg", phash
        )
        
        # Fetch detailed forensics metrics
        summary = forensics.get("investigation_summary", {})
        screenshot_prob = summary.get("screenshot_probability", 0)
        ai_prob = summary.get("ai_generation_probability", 0)
        stego_susp = summary.get("steganography_suspicion", 0)
        
        meta_intel = forensics.get("metadata_intelligence", {})
        meta_trust = meta_intel.get("metadata_trust_score", 100)
        
        blockiness = metadata.get("blockiness", 1.0)
        crop_detected = forensics.get("cropping_detected", False)
        resize_detected = forensics.get("resizing_detected", False)
        watermark_detected = forensics.get("watermark_detected", False)
        comp_status = forensics.get("compression_status", "CLEAN")
        
        # Convert categorical / boolean fields to numerical features
        comp_map = {"CLEAN": 0, "LOW": 1, "HEAVY": 2}
        comp_val = comp_map.get(comp_status, 0)
        
        return {
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "label": label,
            "manipulation_risk": risk,
            "screenshot_probability": screenshot_prob,
            "metadata_trust_score": meta_trust,
            "blockiness": blockiness,
            "crop_detected": 1 if crop_detected else 0,
            "resize_detected": 1 if resize_detected else 0,
            "watermark_detected": 1 if watermark_detected else 0,
            "compression_status": comp_status,
            "compression_status_val": comp_val,
            "ai_generation_probability": ai_prob,
            "stego_suspicion": stego_susp,
            "error": None
        }
    except Exception as e:
        return {
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "label": label,
            "error": str(e)
        }

def compute_pearson_correlation(x_list, y_list):
    n = len(x_list)
    if n == 0:
        return 0.0
    mean_x = sum(x_list) / n
    mean_y = sum(y_list) / n
    
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for x, y in zip(x_list, y_list):
        dx = x - mean_x
        dy = y - mean_y
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
        
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return num / math.sqrt(den_x * den_y)

def compute_auc(scores, labels):
    paired = sorted(zip(scores, labels), key=lambda x: x[0])
    n = len(paired)
    if n == 0:
        return 0.0
    
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    
    # Calculate rank with tie handling
    ranks = [0] * n
    i = 0
    while i < n:
        j = i
        while j < n and paired[j][0] == paired[i][0]:
            j += 1
        avg_rank = 1.0 + (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
        
    pos_rank_sum = sum(ranks[k] for k in range(n) if paired[k][1] == 1)
    u_score = pos_rank_sum - (n_pos * (n_pos + 1)) / 2.0
    return u_score / (n_pos * n_neg)

def get_roc_curve(scores, labels):
    unique_scores = sorted(list(set(scores)))
    # Add boundary scores
    thresholds = [unique_scores[0] - 1] + unique_scores + [unique_scores[-1] + 1]
    
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    
    curve = []
    # Evaluate TPR and FPR at each threshold
    for t in sorted(thresholds):
        tp = 0
        fp = 0
        for s, l in zip(scores, labels):
            if s > t:
                if l == 1:
                    tp += 1
                else:
                    fp += 1
        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        curve.append({"threshold": float(t), "fpr": fpr, "tpr": tpr})
    return curve

def main():
    args = parse_args()
    
    # Resolve Casia dataset paths
    dataset_base, au_dir, tp_dir = resolve_paths(args.dataset_dir)
    
    # Scan files
    au_files = scan_images(au_dir)
    tp_files = scan_images(tp_dir)
    
    limit = args.limit
    if limit == 0:
        limit = None
        print("Running full benchmark on all available CASIA 2.0 images.")
    else:
        print(f"Running in sampling mode. Limit per category: {limit}")
        
    if limit:
        au_files = au_files[:limit]
        tp_files = tp_files[:limit]
        
    total_images = len(au_files) + len(tp_files)
    print(f"Total images to evaluate: {total_images} (Au: {len(au_files)}, Tp: {len(tp_files)})")
    
    # We combine lists with corresponding ground truth labels: 0 for Au, 1 for Tp
    task_list = [(path, 0) for path in au_files] + [(path, 1) for path in tp_files]
    
    # Run pipeline in parallel using a thread pool
    results = []
    completed = 0
    start_time = time.time()
    
    print(f"Starting execution with {args.workers} worker threads...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_image, path, label): (path, label) for path, label in task_list}
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            if res.get("error"):
                print(f"[{completed}/{total_images}] ERROR on {res['filename']}: {res['error']}")
            else:
                results.append(res)
                # Print progress updates occasionally
                if completed % max(1, total_images // 10) == 0 or completed == total_images:
                    print(f"Progress: {completed}/{total_images} files analyzed ({completed/total_images*100:.1f}%)")
                    
    total_elapsed = time.time() - start_time
    print(f"\nDone! Successfully analyzed {len(results)}/{total_images} images in {total_elapsed:.2f} seconds.")
    
    if not results:
        print("ERROR: No valid results were computed. Check image formats and files.")
        return
        
    # Generate CSV of extracted feature vectors
    feature_fields = [
        "filepath", "filename", "label", "manipulation_risk", 
        "screenshot_probability", "metadata_trust_score", "blockiness",
        "crop_detected", "resize_detected", "watermark_detected", 
        "compression_status", "ai_generation_probability", "stego_suspicion"
    ]
    with open("feature_vectors.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=feature_fields)
        writer.writeheader()
        for r in results:
            # Save only valid columns
            row = {k: r[k] for k in feature_fields}
            writer.writerow(row)
    print("Saved feature vectors to: feature_vectors.csv")
    
    # Calculate performance metrics at standard threshold (risk > 35)
    decision_threshold = 35
    tp, tn, fp, fn = 0, 0, 0, 0
    
    false_positives = []
    false_negatives = []
    
    scores = []
    labels = []
    
    for r in results:
        score = r["manipulation_risk"]
        label = r["label"]
        scores.append(score)
        labels.append(label)
        
        pred = 1 if score > decision_threshold else 0
        
        if label == 1 and pred == 1:
            tp += 1
        elif label == 0 and pred == 0:
            tn += 1
        elif label == 0 and pred == 1:
            fp += 1
            false_positives.append(r)
        elif label == 1 and pred == 0:
            fn += 1
            false_negatives.append(r)
            
    # Standard classification metrics
    accuracy = (tp + tn) / len(results) if results else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Save False Positives and False Negatives lists
    for name, items in [("false_positives.csv", false_positives), ("false_negatives.csv", false_negatives)]:
        with open(name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=feature_fields)
            writer.writeheader()
            for r in items:
                row = {k: r[k] for k in feature_fields}
                writer.writerow(row)
    print("Saved False Positives report to: false_positives.csv")
    print("Saved False Negatives report to: false_negatives.csv")
    
    # Save confusion matrix CSV
    with open("confusion_matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", "Predicted Clean", "Predicted Tampered"])
        writer.writerow(["Actual Clean (Au)", tn, fp])
        writer.writerow(["Actual Tampered (Tp)", fn, tp])
    print("Saved Confusion Matrix to: confusion_matrix.csv")
    
    # Calculate ROC-AUC
    auc_score = compute_auc(scores, labels)
    roc_curve = get_roc_curve(scores, labels)
    print(f"ROC-AUC Score: {auc_score:.4f}")
    
    # Threshold Scan Analysis
    threshold_metrics = []
    best_f1 = -1.0
    best_thresh = decision_threshold
    
    for t in range(5, 96, 5):
        t_tp, t_tn, t_fp, t_fn = 0, 0, 0, 0
        for s, l in zip(scores, labels):
            pred = 1 if s > t else 0
            if l == 1 and pred == 1:
                t_tp += 1
            elif l == 0 and pred == 0:
                t_tn += 1
            elif l == 0 and pred == 1:
                t_fp += 1
            elif l == 1 and pred == 0:
                t_fn += 1
        t_prec = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0.0
        t_rec = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0.0
        t_f1 = 2 * (t_prec * t_rec) / (t_prec + t_rec) if (t_prec + t_rec) > 0 else 0.0
        t_acc = (t_tp + t_tn) / len(results)
        
        if t_f1 > best_f1:
            best_f1 = t_f1
            best_thresh = t
            
        threshold_metrics.append({
            "threshold": t,
            "tp": t_tp,
            "tn": t_tn,
            "fp": t_fp,
            "fn": t_fn,
            "accuracy": t_acc,
            "precision": t_prec,
            "recall": t_rec,
            "f1": t_f1
        })
        
    # Feature Correlation calculation (correlation with Label)
    corr_results = []
    features_to_corr = [
        ("manipulation_risk", "manipulation_risk"),
        ("screenshot_probability", "screenshot_probability"),
        ("metadata_trust_score", "metadata_trust_score"),
        ("blockiness", "blockiness"),
        ("crop_detected", "crop_detected"),
        ("resize_detected", "resize_detected"),
        ("watermark_detected", "watermark_detected"),
        ("compression_status_val", "compression_status_val"),
        ("ai_generation_probability", "ai_generation_probability"),
        ("stego_suspicion", "stego_suspicion")
    ]
    
    for feat_name, col_key in features_to_corr:
        x_vals = [r[col_key] for r in results]
        r_coeff = compute_pearson_correlation(x_vals, labels)
        corr_results.append({
            "feature": feat_name,
            "correlation": r_coeff,
            "abs_correlation": abs(r_coeff)
        })
    corr_results.sort(key=lambda x: x["abs_correlation"], reverse=True)
    
    # Save JSON results
    json_output = {
        "dataset_summary": {
            "dataset_base": dataset_base,
            "total_images": len(results),
            "au_images": len(au_files),
            "tp_images": len(tp_files),
            "elapsed_seconds": total_elapsed
        },
        "default_metrics": {
            "decision_threshold": decision_threshold,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "auc_score": auc_score
        },
        "correlations": corr_results,
        "threshold_scan": threshold_metrics,
        "roc_curve": roc_curve[:100]  # limit points in json to keep size reasonable
    }
    with open("evaluation_results.json", "w") as f:
        json.dump(json_output, f, indent=2)
    print("Saved validation metrics in JSON format: evaluation_results.json")
    
    # Write Forensic Markdown report
    write_forensic_markdown_report(
        results, json_output["default_metrics"], best_thresh, best_f1,
        corr_results, threshold_metrics, false_positives, false_negatives, roc_curve
    )
    print("Saved Markdown validation report to: forensic_validation_report.md")

def write_forensic_markdown_report(results, metrics, best_thresh, best_f1, corr_results, threshold_metrics, false_positives, false_negatives, roc_curve):
    report_path = "forensic_validation_report.md"
    
    # Format ROC curve samples (decimated to ~15 steps for presentation)
    roc_points = roc_curve
    if len(roc_points) > 15:
        step = len(roc_points) // 12
        roc_points = roc_points[::step]
        if roc_curve[-1] not in roc_points:
            roc_points.append(roc_curve[-1])
            
    with open(report_path, "w") as f:
        f.write("# TraceLens Forensic Validation Report\n")
        f.write("### Benchmark Dataset: CASIA 2.0 (Au vs Tp)\n\n")
        
        f.write("> [!NOTE]\n")
        f.write(f"This validation report was automatically generated on the CASIA 2.0 forensic dataset. It evaluates the detection accuracy, threshold boundaries, and correlation coefficients of the TraceLens image integrity pipeline.\n\n")
        
        # 1. Presentation Summary Table
        f.write("## 1. Executive Summary: Benchmark Metrics\n")
        f.write("The table below presents the validation metrics suitable for academic and internship presentations:\n\n")
        f.write("| Metric | Value |\n")
        f.write("| :--- | :--- |\n")
        f.write(f"| **Dataset Size** | {len(results)} images (Au/Clean: {len(results) - sum(r['label'] for r in results)}, Tp/Tampered: {sum(r['label'] for r in results)}) |\n")
        f.write(f"| **True Positives (TP)** | {metrics['tp']} |\n")
        f.write(f"| **True Negatives (TN)** | {metrics['tn']} |\n")
        f.write(f"| **False Positives (FP)** | {metrics['fp']} |\n")
        f.write(f"| **False Negatives (FN)** | {metrics['fn']} |\n")
        f.write(f"| **Accuracy** | {metrics['accuracy']*100:.2f}% |\n")
        f.write(f"| **Precision** | {metrics['precision']*100:.2f}% |\n")
        f.write(f"| **Recall (Sensitivity)** | {metrics['recall']*100:.2f}% |\n")
        f.write(f"| **F1 Score** | {metrics['f1_score']*100:.2f}% |\n")
        f.write(f"| **ROC-AUC Score** | {metrics['auc_score']:.4f} |\n\n")
        
        # 2. Current Classification Method
        f.write("## 2. Current Classification Method\n")
        f.write("This benchmark evaluates the **existing rule-based cumulative heuristic engine** of TraceLens and **not a machine-learning classifier**. Predictions are derived via the following explicit heuristics:\n\n")
        f.write("### Manipulation Risk Score Heuristic\n")
        f.write("The manipulation risk score starts at `0` (clean base) and cumulatively increments when specific forensic indicators are flagged:\n")
        f.write("* **Crop Detected**: `+15` points\n")
        f.write("* **Resize Detected**: `+10` points\n")
        f.write("* **Watermark Detected**: `+20` points\n")
        f.write("* **Recompression/Quantization Detected**: `+25` points\n")
        f.write("* **Screenshot Properties Detected**: `+15` points\n")
        f.write("* **Metadata Stripped Detected**: `+10` points\n\n")
        f.write("The accumulated score is capped within bounds of `0` to `95` points:\n")
        f.write("$$\\text{Risk Score} = \\max\\left(0, \\min\\left(95, \\sum \\text{Weights of Active Triggers}\\right)\\right)$$\n\n")
        f.write("### Decision Boundary\n")
        f.write("An image is classified as **Manipulated (Label 1)** if its risk score is greater than **35** (which triggers a verdict of `Manipulated` or `Highly Suspicious` in the Media Profile):\n")
        f.write("$$\\text{Prediction} = \\begin{cases} 1 & \\text{if } \\text{manipulation\\_risk} > 35 \\\\ 0 & \\text{otherwise} \\end{cases}$$\n\n")
        
        # 3. Feature Correlation Analysis
        f.write("## 3. Forensic Indicator Contributions (Feature Importance)\n")
        f.write("To determine which forensic indicators contribute most strongly to the classification of manipulated assets, we compute the Pearson correlation coefficient ($r$) between each extracted indicator and the binary ground truth label. A higher absolute correlation indicates a stronger contribution to positive manipulation detection:\n\n")
        f.write("| Rank | Forensic Indicator | Pearson Correlation ($r$) | Contribution Strength |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for idx, corr in enumerate(corr_results):
            r_val = corr["correlation"]
            strength = "Strong Positive" if r_val > 0.5 else "Moderate Positive" if r_val > 0.2 else "Weak Positive" if r_val > 0.05 else "Weak Negative" if r_val < -0.05 else "Moderate Negative" if r_val < -0.2 else "Negligible"
            f.write(f"| {idx+1} | `{corr['feature']}` | {r_val:+.4f} | {strength} |\n")
        f.write("\n")
        
        # 4. Threshold Boundary Analysis
        f.write("## 4. Decision Boundary Threshold Scan\n")
        f.write("The table below scans the threshold values for the `manipulation_risk` score from 5 to 95 to locate the mathematically optimal decision boundary for CASIA 2.0:\n\n")
        f.write("| Threshold | TP | TN | FP | FN | Accuracy | Precision | Recall | F1 Score |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for m in threshold_metrics:
            is_best = " **(Optimum)**" if m["threshold"] == best_thresh else ""
            f.write(f"| {m['threshold']}{is_best} | {m['tp']} | {m['tn']} | {m['fp']} | {m['fn']} | {m['accuracy']*100:.1f}% | {m['precision']*100:.1f}% | {m['recall']*100:.1f}% | {m['f1']*100:.1f}% |\n")
        f.write("\n")
        f.write(f"The mathematically optimal threshold is **{best_thresh}**, which achieves an F1 score of **{best_f1*100:.2f}%** (compared to the current hardcoded threshold of 35 which yields an F1 score of **{metrics['f1_score']*100:.2f}%**).\n\n")
        
        # 5. ROC Curve Points
        f.write("## 5. ROC Curve Points\n")
        f.write("The Receiver Operating Characteristic (ROC) curve evaluates TPR (Sensitivity) against FPR (1 - Specificity) across decision thresholds:\n\n")
        f.write("| Threshold Scan | False Positive Rate (FPR) | True Positive Rate (TPR) |\n")
        f.write("| :--- | :--- | :--- |\n")
        for pt in roc_points:
            f.write(f"| {pt['threshold']:.1f} | {pt['fpr']:.4f} | {pt['tpr']:.4f} |\n")
        f.write(f"\n**Area Under the ROC Curve (ROC-AUC)**: `{metrics['auc_score']:.4f}`\n\n")
        
        # 6. False Positive Analysis
        f.write("## 6. False Positive Analysis\n")
        f.write("A **False Positive (FP)** occurs when an authentic (unmodified) image from the `Au` folder is classified as manipulated because its risk score exceeded 35.\n\n")
        if not false_positives:
            f.write("No False Positives were detected in this evaluation run.\n\n")
        else:
            f.write(f"Total False Positives: {len(false_positives)} images.\n")
            f.write("### Key Root Causes Identified:\n")
            # Analyze metadata stripping
            fp_missing_meta = sum(1 for fp_item in false_positives if fp_item["metadata_trust_score"] < 50)
            fp_high_compression = sum(1 for fp_item in false_positives if fp_item["compression_status"] in ["LOW", "HEAVY"])
            fp_blockiness = [fp_item["blockiness"] for fp_item in false_positives]
            avg_fp_blockiness = sum(fp_blockiness)/len(fp_blockiness) if fp_blockiness else 1.0
            
            f.write(f"1. **Metadata Absence**: {fp_missing_meta}/{len(false_positives)} False Positives had stripped EXIF data, which automatically penalizes the risk score (`+10` points) and lowers the metadata trust score.\n")
            f.write(f"2. **Natural Compression / Re-saving**: {fp_high_compression}/{len(false_positives)} False Positives had blockiness indices exceeding normal thresholds, triggering compression penalties (`+25` points) despite having no copy-move edit.\n")
            f.write(f"3. **Average Blockiness Index**: The average blockiness index for False Positives was `{avg_fp_blockiness:.2f}`.\n\n")
            f.write("#### Sample False Positive Files:\n")
            f.write("| Filename | Risk Score | Metadata Trust | Blockiness | Active Penalties |\n")
            f.write("| :--- | :---: | :---: | :---: | :--- |\n")
            for fp_item in false_positives[:8]:
                penalties = []
                if fp_item["crop_detected"]: penalties.append("Crop")
                if fp_item["resize_detected"]: penalties.append("Resize")
                if fp_item["watermark_detected"]: penalties.append("Watermark")
                if fp_item["metadata_trust_score"] < 50: penalties.append("Metadata Stripped")
                if fp_item["blockiness"] > 1.3: penalties.append("Compression")
                penalties_str = ", ".join(penalties) if penalties else "None"
                f.write(f"| `{fp_item['filename']}` | {fp_item['manipulation_risk']} | {fp_item['metadata_trust_score']} | {fp_item['blockiness']:.2f} | {penalties_str} |\n")
            f.write("\n")
            
        # 7. False Negative Analysis
        f.write("## 7. False Negative Analysis\n")
        f.write("A **False Negative (FN)** occurs when a tampered (manipulated) image from the `Tp` folder is classified as authentic because its risk score was 35 or lower.\n\n")
        if not false_negatives:
            f.write("No False Negatives were detected in this evaluation run.\n\n")
        else:
            f.write(f"Total False Negatives: {len(false_negatives)} images.\n")
            f.write("### Key Root Causes Identified:\n")
            fn_intact_meta = sum(1 for fn_item in false_negatives if fn_item["metadata_trust_score"] >= 80)
            fn_low_blockiness = sum(1 for fn_item in false_negatives if fn_item["blockiness"] <= 1.2)
            
            f.write(f"1. **Intact Provenance Signatures**: {fn_intact_meta}/{len(false_negatives)} False Negatives retained valid EXIF camera/capture signatures from their source files, meaning no metadata stripped penalty was triggered.\n")
            f.write(f"2. **Undetected Edge Blur / Resampling**: {fn_low_blockiness}/{len(false_negatives)} False Negatives had a blockiness index under `1.2` (average JPEG), which failed to trigger compression/quantization penalties.\n")
            f.write("3. **Missing Copy-Move Ground Truth**: The rule-based engine evaluates local compression blockiness inconsistencies. If a tampered crop was perfectly resampled and recompressed, visual blockiness metrics remain uniform, leaving crop/splice artifacts undetected without semantic reference comparison.\n\n")
            f.write("#### Sample False Negative Files:\n")
            f.write("| Filename | Risk Score | Metadata Trust | Blockiness | Stego Suspicion |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: |\n")
            for fn_item in false_negatives[:8]:
                f.write(f"| `{fn_item['filename']}` | {fn_item['manipulation_risk']} | {fn_item['metadata_trust_score']} | {fn_item['blockiness']:.2f} | {fn_item['stego_suspicion']} |\n")
            f.write("\n")

if __name__ == "__main__":
    main()
