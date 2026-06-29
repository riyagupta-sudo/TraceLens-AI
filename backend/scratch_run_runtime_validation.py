import os
import sys
import time

# Ensure backend folder is in PATH
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
    calculate_integrity_and_risk
)

# Define target test files
test_files = {
    "Real Image": os.path.join(backend_dir, "../dataset/originals/building_001.jpg"),
    "AI-Generated Image": os.path.join(backend_dir, "../dataset/ai_detection/test/FAKE/0.jpg"),
    "Screenshot": os.path.join(backend_dir, "../dataset/Screenshot/screenshot/Discord/1.jpg")
}

for label, rel_path in test_files.items():
    abs_path = os.path.abspath(rel_path)
    print(f"\n========================================\n[UPLOAD TEST] {label}\nFile: {os.path.basename(abs_path)}")
    if not os.path.exists(abs_path):
        print(f"Error: file not found at {abs_path}")
        continue

    # Simulate ingestion preprocessing
    start_total_time = time.perf_counter()
    
    t_meta_start = time.perf_counter()
    sha = compute_sha256(abs_path)
    phash, dhash, ahash = compute_image_hashes(abs_path)
    meta = extract_metadata_signature(abs_path)
    metadata_time_ms = (time.perf_counter() - t_meta_start) * 1000

    emb = get_clip_embedding(abs_path)
    
    meta["embedding"] = emb
    meta["sha256"] = sha
    meta["dhash"] = dhash
    meta["ahash"] = ahash

    mime_type = "image/png" if abs_path.lower().endswith(".png") else "image/jpeg"

    # Call calculate_integrity_and_risk
    integrity, risk, forensics = calculate_integrity_and_risk(
        abs_path,
        meta,
        mime_type,
        phash,
        parent_metadata=None,
        metadata_time_ms=metadata_time_ms,
        start_total_time=start_total_time
    )

    # Print requested Phase 3 outputs
    ai_prob = forensics.get("ai_detection", {}).get("raw_model_probability", 0)
    casia_prob = forensics.get("casia_detection", {}).get("probability", 0)
    rf_prob = int(forensics.get("ml_tampering_probability", 0.0) * 100)
    
    print("\n--- PHASE 3 REQUIRED VALUES ---")
    print(f"AI Model Probability: {ai_prob}%")
    print(f"CASIA Probability: {casia_prob}%")
    print(f"Random Forest Probability: {rf_prob}%")
    print(f"Integrity Score: {integrity}")
    print(f"Risk Score: {risk}")
    
    # Timing is already printed inside calculate_integrity_and_risk,
    # but let's print the requested line here as well for Phase 3 checklist.
    total_time = (time.perf_counter() - start_total_time) * 1000
    print(f"Total Analysis Time: {int(round(total_time))} ms")
    print("========================================\n")
