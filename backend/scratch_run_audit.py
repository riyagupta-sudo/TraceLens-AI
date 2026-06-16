import os
import sys
import json
import numpy as np

# Add backend to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.dna_engine import (
    compute_sha256,
    compute_image_hashes,
    extract_metadata_signature,
    get_clip_embedding,
    calculate_integrity_and_risk
)

files_to_test = {
    "birds.jpg": "../dataset/case_intel_leak/birds.jpg",
    "spider.jpg": "../dataset/case_intel_leak/spider.jpg",
    "random.jpg": "../dataset/case_crowd_event/random.jpg",
    "Screenshot 2026-06-09.png": "../dataset/Screenshot 2026-06-09.png"
}

results = {}

for name, rel_path in files_to_test.items():
    abs_path = os.path.abspath(rel_path)
    if not os.path.exists(abs_path):
        print(f"File missing: {abs_path}")
        continue
    
    sha = compute_sha256(abs_path)
    phash, dhash, ahash = compute_image_hashes(abs_path)
    meta = extract_metadata_signature(abs_path)
    emb = get_clip_embedding(abs_path)
    
    meta["embedding"] = emb
    meta["sha256"] = sha
    meta["dhash"] = dhash
    meta["ahash"] = ahash
    
    # Run the dna engine
    integrity, risk, forensics = calculate_integrity_and_risk(
        abs_path, meta, "image/png" if name.endswith(".png") else "image/jpeg", phash, parent_metadata=None
    )
    
    # Extract needed scores
    results[name] = {
        "Manipulation Risk": risk,
        "Screenshot Probability": forensics.get("screenshot_indicators", {}).get("confidence", 0),
        "AI Generation Probability": forensics.get("ai_detection", {}).get("probability", 0),
        "Metadata Trust Score": forensics.get("metadata_intelligence", {}).get("metadata_trust_score", 100),
        "Stego Suspicion": forensics.get("forensic_investigation", {}).get("suspicion_score", 0),
        "Investigation Confidence": forensics.get("overall_investigation_confidence", {}).get("score", 0)
    }

print("\n--- RESULTS JSON ---")
print(json.dumps(results, indent=2))
print("--- END RESULTS ---")
