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
    calculate_integrity_and_risk
)

targets = [
    r"C:\Users\riya2\Downloads\IMG_2252.jpeg",
    r"C:\Users\riya2\Downloads\IMG_2253.jpeg"
]

print("=== START AI DETECTOR V1 VERIFICATION ===")

for t in targets:
    print(f"\nEvaluating target file: {t}")
    if not os.path.exists(t):
        print(f"ERROR: File not found at {t}")
        continue

    # Run full ingestion simulation
    start_total_time = time.perf_counter()
    
    t_meta_start = time.perf_counter()
    sha = compute_sha256(t)
    phash, dhash, ahash = compute_image_hashes(t)
    meta = extract_metadata_signature(t)
    metadata_time_ms = (time.perf_counter() - t_meta_start) * 1000

    meta["sha256"] = sha
    meta["dhash"] = dhash
    meta["ahash"] = ahash
    meta["embedding"] = None # None or default

    mime_type = "image/jpeg"

    # Call calculate_integrity_and_risk
    integrity, risk, forensics = calculate_integrity_and_risk(
        t,
        meta,
        mime_type,
        phash,
        parent_metadata=None,
        metadata_time_ms=metadata_time_ms,
        start_total_time=start_total_time
    )

    print("\n--- RESULTS ---")
    print(f"File: {os.path.basename(t)}")
    print(f"Integrity Score: {integrity} (Expected: 100)")
    print(f"Risk Score: {risk} (Expected: 0)")
    
    ai_prob = forensics.get("ai_detection", {}).get("probability", 0)
    print(f"AI Detector Signal: {ai_prob}%")
    
    audit_note = forensics.get("ai_audit_note", {})
    print(f"Audit Note Triggered: {audit_note.get('triggered')}")
    if audit_note.get("triggered"):
        print(f"Audit Note Message: {audit_note.get('message')}")
        print(f"Signal Breakdown: {audit_note.get('signal_breakdown')}")

    # Assertions
    assert integrity == 100, f"Integrity was {integrity}, expected 100"
    assert risk == 0, f"Risk was {risk}, expected 0"
    expected_triggered = ai_prob > 70
    assert audit_note.get("triggered") == expected_triggered, f"Audit note triggered was {audit_note.get('triggered')}, expected {expected_triggered}"
    print("SUCCESS: Target verified correctly!")

print("\n=== VERIFICATION COMPLETED ===")
