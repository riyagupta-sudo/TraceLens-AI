import os
import sys
import json
import time

backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
app_dir = os.path.join(backend_dir, "app")
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from app.dna_engine import (
    compute_sha256,
    compute_image_hashes,
    extract_metadata_signature,
    get_clip_embedding,
    calculate_integrity_and_risk,
    predict_casia_tampering,
    analyze_steganography_and_forensics,
    estimate_compression_artifacts,
    detect_screenshot_properties,
    detect_ai_generation
)
from PIL import Image
import numpy as np

filepath = r"C:\Users\riya2\Downloads\1000111612.jpg"
print("File path:", filepath)
print("Exists:", os.path.exists(filepath))

if os.path.exists(filepath):
    # Ingestion steps
    start_total_time = time.perf_counter()
    sha = compute_sha256(filepath)
    phash, dhash, ahash = compute_image_hashes(filepath)
    meta = extract_metadata_signature(filepath)
    metadata_time_ms = (time.perf_counter() - start_total_time) * 1000

    emb = get_clip_embedding(filepath)
    meta["embedding"] = emb
    meta["sha256"] = sha
    meta["dhash"] = dhash
    meta["ahash"] = ahash

    mime_type = "image/jpeg"

    # Open PIL Image once and convert to RGB and L for reuse
    img_pil = Image.open(filepath)
    img_rgb = img_pil.convert("RGB")
    img_l = img_pil.convert("L")

    # Run detectors individually to get internal details
    print("\n--- Running sub-detectors individually ---")
    ai_res = detect_ai_generation(filepath, meta, emb, img_rgb, img_l)
    casia_prob, casia_class = predict_casia_tampering(filepath, img_rgb)
    stego_res = analyze_steganography_and_forensics(filepath, metadata=meta)
    blockiness = estimate_compression_artifacts(filepath, img_l)
    
    # Run screenshot detector
    ss_status, ss_score, ss_lvl, ss_matrix = detect_screenshot_properties(
        filepath, meta, is_derived=False, img_rgb=img_rgb
    )
    
    # Compute FFT peak count manually to be sure we get the exact value
    import cv2
    num_peaks = 0
    img_gray = np.array(img_l)
    resized = cv2.resize(img_gray, (256, 256))
    dft = np.fft.fft2(resized)
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

    # Let's run calculate_integrity_and_risk
    print("\n--- Running calculate_integrity_and_risk standalone ---")
    integrity, risk, forensics = calculate_integrity_and_risk(
        filepath,
        meta,
        mime_type,
        phash,
        parent_metadata=None,
        metadata_time_ms=metadata_time_ms,
        start_total_time=start_total_time
    )

    print("\n--- DETECTOR OUTPUTS ---")
    print(f"CASIA probability: {casia_prob}%")
    rf_prob_pct = forensics.get("ml_tampering_probability", 0.0) * 100
    print(f"RF probability: {rf_prob_pct}%")
    print(f"Screenshot probability: {forensics.get('screenshot_indicators', {}).get('confidence', 0)}%")
    print(f"Stego suspicion: {forensics.get('forensic_investigation', {}).get('suspicion_score', 0)}%")
    print(f"Metadata trust: {forensics.get('metadata_intelligence', {}).get('metadata_trust_score', 100)}")
    print(f"ELA blockiness: {blockiness}")
    print(f"FFT anomaly score (peaks): {num_peaks}")
    print(f"Metadata stripped check: {forensics.get('metadata_stripped')}")
    print(f"Metadata stripped possible check: {forensics.get('metadata_stripped_possible')}")

    print("\n--- SCORE AUDIT TRACE ---")
    # Let's rebuild the score trace exactly from the formula in calculate_integrity_and_risk
    exif = meta.get("exif", {})
    metadata_present = bool(exif)
    camera_information = bool(exif.get("Make") or exif.get("Model")) if exif else False
    capture_timestamp = bool(exif.get("DateTimeOriginal") or exif.get("DateTime")) if exif else False
    editing_software = forensics.get("re_encoded", False)
    gps_information = bool(exif.get("GPSInfo")) if exif else False
    
    if not metadata_present:
        meta_trust = 15
    else:
        meta_trust = 100
        if not camera_information:
            meta_trust -= 30
        if not capture_timestamp:
            meta_trust -= 30
        if editing_software:
            meta_trust -= 30
        if not gps_information:
            meta_trust -= 10
        meta_trust = max(10, meta_trust)

    is_crop = forensics.get("cropping_detected", False)
    is_resize = forensics.get("resizing_detected", False)
    is_watermark = forensics.get("watermark_detected", False)
    is_compressed = forensics.get("heavy_compression", False)
    ss_status_f = forensics.get("screenshot_indicators", {}).get("status")
    is_screenshot = (ss_status_f in ["Likely Screenshot", "Possible Screenshot"])
    metadata_stripped = forensics.get("metadata_stripped", False)

    # Base Integrity deductions:
    base_integrity_deductions = []
    if is_crop: base_integrity_deductions.append(("cropping", 15))
    if is_resize: base_integrity_deductions.append(("resizing", 10))
    if is_watermark: base_integrity_deductions.append(("watermarking", 15))
    if is_compressed: base_integrity_deductions.append(("heavy_compression", 20))
    if is_screenshot: base_integrity_deductions.append(("screenshot", 25))
    
    if meta_trust < 30:
        base_integrity_deductions.append(("metadata_trust", 20))
    elif meta_trust < 60:
        base_integrity_deductions.append(("metadata_trust", 10))
    elif meta_trust < 90:
        base_integrity_deductions.append(("metadata_trust", 5))

    # Base Risk additions:
    base_risk_additions = []
    if is_crop: base_risk_additions.append(("cropping", 15))
    if is_resize: base_risk_additions.append(("resizing", 10))
    if is_watermark: base_risk_additions.append(("watermarking", 20))
    if is_compressed: base_risk_additions.append(("heavy_compression", 25))
    if is_screenshot: base_risk_additions.append(("screenshot", 15))
    if metadata_stripped: base_risk_additions.append(("metadata_stripped", 10))

    if meta_trust < 30:
        base_risk_additions.append(("metadata_trust", 20))
    elif meta_trust < 60:
        base_risk_additions.append(("metadata_trust", 10))
    elif meta_trust < 90:
        base_risk_additions.append(("metadata_trust", 5))

    # Forensic additions/deductions:
    rf_deduction = 0
    rf_addition = 0
    if rf_prob_pct >= 75:
        rf_deduction = 30
        rf_addition = 30
    elif rf_prob_pct >= 50:
        rf_deduction = 20
        rf_addition = 20
    elif rf_prob_pct >= 30:
        rf_deduction = 10
        rf_addition = 10

    casia_deduction = 0
    casia_addition = 0
    if casia_prob >= 90:
        casia_deduction = 25
        casia_addition = 25
    elif casia_prob >= 75:
        casia_deduction = 15
        casia_addition = 15
    elif casia_prob >= 50:
        casia_deduction = 10
        casia_addition = 10

    if casia_prob >= 75 and meta_trust >= 90 and rf_prob_pct < 20:
        casia_deduction = min(casia_deduction, 10)
        casia_addition = min(casia_addition, 10)

    stego_deduction = 0
    stego_addition = 0
    stego_susp = forensics.get("forensic_investigation", {}).get("suspicion_score", 0)
    if stego_susp >= 50:
        stego_deduction = 15
        stego_addition = 15
    elif stego_susp >= 30:
        stego_deduction = 10
        stego_addition = 10

    print("Risk Additions detail:")
    print("  Base crop addition:", 15 if is_crop else 0)
    print("  Base resize addition:", 10 if is_resize else 0)
    print("  Base watermark addition:", 20 if is_watermark else 0)
    print("  Base compressed addition:", 25 if is_compressed else 0)
    print("  Base screenshot addition:", 15 if is_screenshot else 0)
    print("  Base metadata stripped addition:", 10 if metadata_stripped else 0)
    meta_trust_add = 0
    if meta_trust < 30: meta_trust_add = 20
    elif meta_trust < 60: meta_trust_add = 10
    elif meta_trust < 90: meta_trust_add = 5
    print("  Metadata trust score addition:", meta_trust_add)
    print("  RF addition:", rf_addition)
    print("  CASIA addition:", casia_addition)
    print("  Stego addition:", stego_addition)

    print("\nIntegrity Deductions detail:")
    print("  Base crop deduction:", 15 if is_crop else 0)
    print("  Base resize deduction:", 10 if is_resize else 0)
    print("  Base watermark deduction:", 15 if is_watermark else 0)
    print("  Base compressed deduction:", 20 if is_compressed else 0)
    print("  Base screenshot deduction:", 25 if is_screenshot else 0)
    meta_trust_ded = 0
    if meta_trust < 30: meta_trust_ded = 20
    elif meta_trust < 60: meta_trust_ded = 10
    elif meta_trust < 90: meta_trust_ded = 5
    print("  Metadata trust score deduction:", meta_trust_ded)
    print("  RF deduction:", rf_deduction)
    print("  CASIA deduction:", casia_deduction)
    print("  Stego deduction:", stego_deduction)

    print("\nIntegrity Score:", integrity)
    print("Manipulation Risk Score:", risk)

else:
    print("Target image file not found.")
