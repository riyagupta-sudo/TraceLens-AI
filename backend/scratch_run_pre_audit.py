import os
import sys
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

filepath = os.path.join(backend_dir, "app", "uploads", "upload_1782113177.447184.jpg")
print("Target file path:", filepath)
print("Exists:", os.path.exists(filepath))

if os.path.exists(filepath):
    # Standard ingestion steps
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

    # We run the sub-detectors individually to capture raw details like FFT peaks
    img_pil = Image.open(filepath)
    img_rgb = img_pil.convert("RGB")
    img_l = img_pil.convert("L")

    print("\n--- Running detectors individually to get internal details ---")
    ai_res = detect_ai_generation(filepath, meta, emb, img_rgb, img_l)
    print("AI detector response:", ai_res)

    casia_prob, casia_class = predict_casia_tampering(filepath, img_rgb)
    print(f"CASIA: prob={casia_prob}, class={casia_class}")

    stego_res = analyze_steganography_and_forensics(filepath, metadata=meta)
    print("Stego response:", stego_res)

    blockiness = estimate_compression_artifacts(filepath, img_l)
    print("Blockiness:", blockiness)

    ss_status, ss_score, ss_lvl, ss_matrix = detect_screenshot_properties(filepath, meta, is_derived=False, img_rgb=img_rgb)
    print(f"Screenshot: status={ss_status}, score={ss_score}, lvl={ss_lvl}, matrix={ss_matrix}")

    # Now run calculate_integrity_and_risk (standalone, parent_metadata=None)
    print("\n--- Running calculate_integrity_and_risk standalone (parent_metadata=None) ---")
    integrity, risk, forensics = calculate_integrity_and_risk(
        filepath,
        meta,
        mime_type,
        phash,
        parent_metadata=None,
        metadata_time_ms=metadata_time_ms,
        start_total_time=start_total_time
    )

    print("\n--- STANDALONE SCORES ---")
    print("Integrity:", integrity)
    print("Risk:", risk)
    print("Forensics keys:", list(forensics.keys()))
    print("Forensic explanations:")
    import json
    print(json.dumps(forensics.get("forensic_score_explanations", []), indent=2))
else:
    print("File not found")
