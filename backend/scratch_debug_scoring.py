import os
import sys

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

# Test target files
test_files = {
    "Real Image": os.path.join(backend_dir, "../dataset/originals/building_001.jpg"),
    "AI-Generated Image": os.path.join(backend_dir, "../dataset/ai_detection/test/FAKE/0.jpg"),
}

for label, path in test_files.items():
    abs_path = os.path.abspath(path)
    print(f"\n========================================\n[DIAGNOSTIC] {label}\nFile: {os.path.basename(abs_path)}")
    
    sha = compute_sha256(abs_path)
    phash, dhash, ahash = compute_image_hashes(abs_path)
    meta = extract_metadata_signature(abs_path)
    
    # Print raw EXIF keys
    print("EXIF Metadata keys present:", list(meta.get("exif", {}).keys()))
    
    emb = [0.0] * 512
    meta["embedding"] = emb
    meta["sha256"] = sha
    meta["dhash"] = dhash
    meta["ahash"] = ahash

    mime_type = "image/jpeg"

    integrity, risk, forensics = calculate_integrity_and_risk(
        abs_path,
        meta,
        mime_type,
        phash,
        parent_metadata=None
    )

    # Let's inspect the flags inside forensics:
    print("Forensics flags:")
    print("  metadata_stripped:", forensics.get("metadata_stripped"))
    print("  heavy_compression:", forensics.get("heavy_compression"))
    print("  low_resolution:", forensics.get("low_resolution"))
    print("  manipulation_indicator:", forensics.get("manipulation_indicator"))
    print("  re_encoded:", forensics.get("re_encoded"))
    print("  cropping_detected:", forensics.get("cropping_detected"))
    print("  resizing_detected:", forensics.get("resizing_detected"))
    print("  watermark_detected:", forensics.get("watermark_detected"))
    print("  screenshot_indicators.status:", forensics.get("screenshot_indicators", {}).get("status"))
    print("  screenshot_indicators.confidence:", forensics.get("screenshot_indicators", {}).get("confidence"))
    print("  metadata_intelligence.metadata_trust_score:", forensics.get("metadata_intelligence", {}).get("metadata_trust_score"))
    print("  asset_classification:", forensics.get("asset_classification"))
    print("----------------------------------------")
    print("Calculated Integrity Score:", integrity)
    print("Calculated Risk Score:", risk)
    print("========================================\n")
