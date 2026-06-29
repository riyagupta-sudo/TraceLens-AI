import os
import io
import time
import cv2
import numpy as np
from PIL import Image
import logging

def detect_ai_editing(filepath: str) -> dict:
    """
    Localized AI Editing Forensics Engine (independent backend module).
    Uses a Two-Stage sliding-window pipeline to locate localized inpainting/erasure anomalies.
    Returns:
      {
        "editing_probability": int (2-98),
        "confidence": str ("High" | "Medium" | "Low"),
        "suspicious_regions": list,
        "signals": dict
      }
    """
    default_res = {
        "editing_probability": 2,
        "confidence": "Low",
        "suspicious_regions": [],
        "signals": {
            "ela_inconsistency": 0.0,
            "noise_residual_var": 0.0,
            "jpeg_block_inconsistency": 0.0,
            "laplacian_variance_avg": 0.0,
            "local_entropy_avg": 0.0
        }
    }
    
    if not os.path.exists(filepath):
        logging.warning(f"[AI Editing Detector] File not found: {filepath}")
        return default_res

    try:
        t_start = time.perf_counter()
        
        # Load Image
        img = Image.open(filepath)
        orig_w, orig_h = img.size
        
        # Downscale if max dimension > 1024 to optimize CPU
        max_dim = 1024
        if max(orig_w, orig_h) > max_dim:
            if orig_w > orig_h:
                new_w = max_dim
                new_h = int(orig_h * (max_dim / orig_w))
            else:
                new_h = max_dim
                new_w = int(orig_w * (max_dim / orig_h))
            img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            img_resized = img
            new_w, new_h = orig_w, orig_h

        img_rgb = img_resized.convert("RGB")
        img_gray = np.array(img_resized.convert("L"))
        img_rgb_arr = np.array(img_rgb)
        
        # Define patch and step parameters
        patch_size = 128
        p_w = min(patch_size, new_w)
        p_h = min(patch_size, new_h)
        step = 64

        # ------------------ IMAGE-WIDE BASELINES ------------------
        # 1. Image-wide ELA Baseline
        out = io.BytesIO()
        img_rgb.save(out, format="JPEG", quality=90)
        out.seek(0)
        img_jpeg = Image.open(out)
        img_jpeg_arr = np.array(img_jpeg)
        img_ela_full = np.abs(img_rgb_arr.astype(float) - img_jpeg_arr.astype(float))
        img_ela_gray = np.mean(img_ela_full, axis=2)
        avg_ela = np.mean(img_ela_gray)
        
        # 2. Image-wide Noise Baseline
        full_median = cv2.medianBlur(img_gray, 3)
        full_noise_residual = img_gray.astype(float) - full_median.astype(float)
        avg_noise_var = np.var(full_noise_residual)
        
        # 3. Image-wide Blockiness Baseline
        h_diff = np.abs(img_gray[:, :-1].astype(float) - img_gray[:, 1:].astype(float))
        v_diff = np.abs(img_gray[:-1, :].astype(float) - img_gray[1:, :].astype(float))
        
        boundary_cols = [i for i in range(7, new_w - 1, 8)]
        boundary_rows = [i for i in range(7, new_h - 1, 8)]
        internal_cols = [i for i in range(new_w - 1) if i not in boundary_cols]
        internal_rows = [i for i in range(new_h - 1) if i not in boundary_rows]
        
        if boundary_cols and internal_cols:
            b_h_val = np.mean(h_diff[:, boundary_cols])
            i_h_val = np.mean(h_diff[:, internal_cols])
        else:
            b_h_val, i_h_val = 0.0, 0.0
            
        if boundary_rows and internal_rows:
            b_v_val = np.mean(v_diff[boundary_rows, :])
            i_v_val = np.mean(v_diff[internal_rows, :])
        else:
            b_v_val, i_v_val = 0.0, 0.0
            
        avg_blockiness_ratio = (b_h_val + b_v_val) / (i_h_val + i_v_val + 1e-5)

        # 4. Image-wide Laplacian & Entropy
        avg_lap_var_full = float(np.var(cv2.Laplacian(img_gray, cv2.CV_64F)))
        hist_full, _ = np.histogram(img_gray, bins=16, range=(0, 256), density=True)
        hist_full = hist_full[hist_full > 0]
        entropy_full = float(-np.sum(hist_full * np.log2(hist_full))) if hist_full.size > 0 else 0.0

        # ------------------ SLIDING WINDOW PIPELINE ------------------
        candidate_boxes = []
        lap_var_all = []
        entropy_all = []

        # Sliding loop
        for y in range(0, new_h - p_h + 1, step):
            for x in range(0, new_w - p_w + 1, step):
                patch_gray = img_gray[y:y+p_h, x:x+p_w]
                
                # 1. Laplacian Variance (texture detail)
                lap_var = float(np.var(cv2.Laplacian(patch_gray, cv2.CV_64F)))
                lap_var_all.append(lap_var)
                
                # 2. Local Entropy (information content)
                hist, _ = np.histogram(patch_gray, bins=16, range=(0, 256), density=True)
                hist = hist[hist > 0]
                entropy = float(-np.sum(hist * np.log2(hist))) if hist.size > 0 else 0.0
                entropy_all.append(entropy)
                
                # 3. Local Noise Residual
                local_noise_var = np.var(full_noise_residual[y:y+p_h, x:x+p_w])
                
                # 4. Local ELA
                local_ela = np.mean(img_ela_gray[y:y+p_h, x:x+p_w])
                
                # 5. Canny Edges
                canny = cv2.Canny(patch_gray, 50, 150)
                edge_density = float(np.mean(canny > 0))

                # --- FLAT REGION SUPPRESSION GATE ---
                # Avoid false positives on naturally flat regions (skies, graphic backgrounds)
                # by suppressing anomalies in patches lacking texture, entropy, and noise.
                is_flat_patch = (lap_var < 15.0) and (entropy < 1.8) and (local_noise_var < 5.0)
                
                if is_flat_patch:
                    s1_ela = 0.0
                    s1_noise = 0.0
                    s1_block = 0.0
                    s2_lap = 0.0
                    s2_entropy = 0.0
                    s2_fft = 0.0
                    s2_edge = 0.0
                else:
                    # --- STAGE 1: FAST SCREENING ---
                    # Check ELA Anomaly
                    if avg_ela < 1.5:
                        s1_ela = 0.0
                    else:
                        ela_ratio = local_ela / (avg_ela + 1e-5)
                        if ela_ratio > 1.8:
                            s1_ela = min(1.0, (ela_ratio - 1.8) / 2.0)
                        elif ela_ratio < 0.35:
                            s1_ela = 1.0 - (ela_ratio / 0.35)
                        else:
                            s1_ela = 0.0
                    
                    # Check Noise Anomaly
                    if avg_noise_var < 5.0:
                        s1_noise = 0.0
                    else:
                        noise_ratio = local_noise_var / (avg_noise_var + 1e-5)
                        if noise_ratio < 0.3:
                            s1_noise = 1.0 - (noise_ratio / 0.3)
                        else:
                            s1_noise = 0.0
                    
                    # Check JPEG Block Boundary Disruption
                    if avg_blockiness_ratio < 1.05 or avg_blockiness_ratio > 2.0:
                        s1_block = 0.0
                    else:
                        p_h_diff = h_diff[y:y+p_h, max(0, x-1):min(new_w-1, x+p_w)]
                        p_v_diff = v_diff[max(0, y-1):min(new_h-1, y+p_h), x:x+p_w]
                        
                        p_boundary_cols = [col - x for col in boundary_cols if col >= x and col < x+p_w-1]
                        p_boundary_rows = [row - y for row in boundary_rows if row >= y and row < y+p_h-1]
                        p_internal_cols = [col - x for col in internal_cols if col >= x and col < x+p_w-1]
                        p_internal_rows = [row - y for row in internal_rows if row >= y and row < y+p_h-1]
                        
                        b_h_p = np.mean(p_h_diff[:, p_boundary_cols]) if p_boundary_cols and p_h_diff.size > 0 else 0.0
                        i_h_p = np.mean(p_h_diff[:, p_internal_cols]) if p_internal_cols and p_h_diff.size > 0 else 0.0
                        b_v_p = np.mean(p_v_diff[p_boundary_rows, :]) if p_boundary_rows and p_v_diff.size > 0 else 0.0
                        i_v_p = np.mean(p_v_diff[p_internal_rows, :]) if p_internal_rows and p_v_diff.size > 0 else 0.0
                        
                        p_blockiness = (b_h_p + b_v_p) / (i_h_p + i_v_p + 1e-5)
                        if p_blockiness < 0.45 * avg_blockiness_ratio:
                            s1_block = 1.0 - (p_blockiness / (0.45 * avg_blockiness_ratio + 1e-5))
                        else:
                            s1_block = 0.0
                    
                    # --- STAGE 2: DETAILED ANALYSIS ---
                    # Check Texture smoothing relative to overall image texture
                    if avg_lap_var_full < 100.0:
                        s2_lap = 0.0
                    else:
                        if lap_var < 0.15 * avg_lap_var_full and lap_var < 25.0:
                            s2_lap = 1.0 - (lap_var / (0.15 * avg_lap_var_full))
                        else:
                            s2_lap = 0.0
                    
                    # Check Local Entropy relative to image baseline
                    if entropy_full < 3.0:
                        s2_entropy = 0.0
                    else:
                        if entropy < 0.35 * entropy_full and entropy < 2.2:
                            s2_entropy = 1.0 - (entropy / (0.35 * entropy_full))
                        else:
                            s2_entropy = 0.0
                    
                    # Check FFT Magnitude Spikes (periodic grid anomalies)
                    dft = np.fft.fft2(patch_gray)
                    dft_shift = np.fft.fftshift(dft)
                    magnitude = 20 * np.log(np.abs(dft_shift) + 1e-8)
                    cy, cx = p_h // 2, p_w // 2
                    Y, X = np.ogrid[-cy:p_h-cy, -cx:p_w-cx]
                    r2 = X**2 + Y**2
                    r_min, r_max = min(p_w, p_h) // 4, min(p_w, p_h) // 2 - 4
                    mask = (r2 >= r_min**2) & (r2 <= r_max**2)
                    ring = magnitude[mask] if np.any(mask) else np.array([])
                    
                    if ring.size > 0:
                        r_mean = np.mean(ring)
                        r_std = np.std(ring)
                        peaks = ring[ring > (r_mean + 3.5 * r_std)]
                        fft_peaks = len(peaks)
                    else:
                        fft_peaks = 0
                    s2_fft = min(1.0, fft_peaks / 12.0)
                    
                    # Check Edge Density continuity
                    s2_edge = 1.0 - (edge_density / 0.005) if edge_density < 0.005 else 0.0

                # Compute weighted patch anomaly score
                patch_score = (
                    0.20 * s1_ela +
                    0.20 * s1_noise +
                    0.15 * s2_lap +
                    0.15 * s2_entropy +
                    0.15 * s1_block +
                    0.075 * s2_fft +
                    0.075 * s2_edge
                )
                
                # Boost if multiple independent forensic indicators are strong
                active_signals_count = sum(1 for s in [s1_ela, s1_noise, s2_lap, s2_entropy, s1_block] if s > 0.3)
                if active_signals_count >= 2:
                    patch_score = max(patch_score, 0.75 * s1_ela, 0.75 * s2_lap, 0.75 * s1_noise, 0.75 * s1_block)
                
                patch_score = max(0.02, min(0.98, patch_score))
                
                if patch_score >= 0.25:
                    candidate_boxes.append((
                        x, y, x + p_w, y + p_h, patch_score,
                        s1_ela, s1_noise, s2_lap, s2_entropy, s1_block, s2_fft, s2_edge
                    ))

        # ------------------ MERGING ALGORITHM ------------------
        def check_closeness(boxA, boxB, margin=64):
            x1_A, y1_A, x2_A, y2_A = boxA[0], boxA[1], boxA[2], boxA[3]
            x1_B, y1_B, x2_B, y2_B = boxB[0], boxB[1], boxB[2], boxB[3]
            x_close = not (x2_A + margin < x1_B or x2_B + margin < x1_A)
            y_close = not (y2_A + margin < y1_B or y2_B + margin < y1_A)
            return x_close and y_close

        def merge_overlapping_candidates(boxes, margin=64):
            merged_any = True
            while merged_any:
                merged_any = False
                i = 0
                while i < len(boxes):
                    j = i + 1
                    while j < len(boxes):
                        if check_closeness(boxes[i], boxes[j], margin):
                            b1 = boxes[i]
                            b2 = boxes[j]
                            x1 = min(b1[0], b2[0])
                            y1 = min(b1[1], b2[1])
                            x2 = max(b1[2], b2[2])
                            y2 = max(b1[3], b2[3])
                            
                            score = max(b1[4], b2[4])
                            s_ela = max(b1[5], b2[5])
                            s_noise = max(b1[6], b2[6])
                            s_lap = max(b1[7], b2[7])
                            s_ent = max(b1[8], b2[8])
                            s_blk = max(b1[9], b2[9])
                            s_fft = max(b1[10], b2[10])
                            s_edg = max(b1[11], b2[11])
                            
                            boxes[i] = (x1, y1, x2, y2, score, s_ela, s_noise, s_lap, s_ent, s_blk, s_fft, s_edg)
                            boxes.pop(j)
                            merged_any = True
                        else:
                            j += 1
                    i += 1
            return boxes

        # Run candidate merging
        merged_boxes = merge_overlapping_candidates(candidate_boxes, margin=64)
        
        suspicious_regions = []
        for box in merged_boxes:
            x1, y1, x2, y2, score, s_ela, s_noise, s_lap, s_ent, s_blk, s_fft, s_edg = box
            
            x_pct = (x1 / new_w) * 100
            y_pct = (y1 / new_h) * 100
            width_pct = ((x2 - x1) / new_w) * 100
            height_pct = ((y2 - y1) / new_h) * 100
            
            active_signals = []
            if s_lap > 0.4: active_signals.append("Texture smoothing")
            if s_ela > 0.4: active_signals.append("ELA anomaly")
            if s_noise > 0.4: active_signals.append("Noise residual mismatch")
            if s_blk > 0.4: active_signals.append("JPEG block grid disruption")
            if s_ent > 0.4: active_signals.append("Entropy anomaly")
            if s_fft > 0.4: active_signals.append("Deconvolution spikes")
            if s_edg > 0.4: active_signals.append("Edge discontinuity")
            
            if not active_signals:
                active_signals.append("Localized compression mismatch")
                
            reason = " & ".join(active_signals)
            confidence_level = "High" if score > 0.7 else "Medium" if score > 0.4 else "Low"
            
            suspicious_regions.append({
                "x_pct": float(round(x_pct, 2)),
                "y_pct": float(round(y_pct, 2)),
                "width_pct": float(round(width_pct, 2)),
                "height_pct": float(round(height_pct, 2)),
                "confidence": confidence_level,
                "reason": reason,
                "score": int(score * 100),
                "signals": {
                    "ela_score": float(round(s_ela, 2)),
                    "noise_consistency": float(round(s_noise, 2)),
                    "laplacian_variance": float(round(s_lap, 2)),
                    "local_entropy": float(round(s_ent, 2)),
                    "block_disruption": float(round(s_blk, 2)),
                    "fft_anomaly": float(round(s_fft, 2)),
                    "edge_discontinuity": float(round(s_edg, 2))
                }
            })
            
        if suspicious_regions:
            sorted_regions = sorted(suspicious_regions, key=lambda r: r["score"], reverse=True)
            top_scores = [r["score"] for r in sorted_regions[:3]]
            editing_probability = int(np.mean(top_scores))
        else:
            editing_probability = 2
            
        editing_probability = max(2, min(98, editing_probability))
        
        if editing_probability >= 70:
            confidence = "High"
        elif editing_probability >= 35:
            confidence = "Medium"
        else:
            confidence = "Low"
            
        t_duration_ms = (time.perf_counter() - t_start) * 1000
        logging.info(f"[AI Editing Detector] Completed analysis in {t_duration_ms:.2f} ms. Probability: {editing_probability}%")
        
        signals = {
            "ela_inconsistency": float(round(avg_ela, 4)),
            "noise_residual_var": float(round(avg_noise_var, 4)),
            "jpeg_block_inconsistency": float(round(avg_blockiness_ratio, 4)),
            "laplacian_variance_avg": float(round(np.mean(lap_var_all), 2)) if lap_var_all else 0.0,
            "local_entropy_avg": float(round(np.mean(entropy_all), 2)) if entropy_all else 0.0
        }
        
        return {
            "editing_probability": editing_probability,
            "confidence": confidence,
            "suspicious_regions": suspicious_regions,
            "signals": signals
        }

    except Exception as e:
        logging.error(f"[AI Editing Detector] Error analyzing {filepath}: {e}", exc_info=True)
        return default_res
