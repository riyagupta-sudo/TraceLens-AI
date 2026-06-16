import os
import shutil
import json
import sys
from PIL import Image

# Add backend to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.dna_engine import (
    calculate_integrity_and_risk,
    extract_metadata_signature,
    compute_image_hashes
)

def test_blind_investigation_standalone():
    print("Running Standalone Blind Investigation Test...")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_dir = os.path.join(base_dir, "dataset", "case_intel_leak")
    orig_path = os.path.join(dataset_dir, "moon_original.jpg")
    
    if not os.path.exists(orig_path):
        # Fallback to copy moon.jpg if it exists
        moon_path = os.path.join(dataset_dir, "moon.jpg")
        if os.path.exists(moon_path):
            shutil.copy(moon_path, orig_path)
            
    if not os.path.exists(orig_path):
        print("Error: moon_original.jpg missing and no fallback.")
        sys.exit(1)
        
    ph, _, _ = compute_image_hashes(orig_path)
    meta = extract_metadata_signature(orig_path)
    
    integrity, risk, forensics = calculate_integrity_and_risk(
        orig_path, meta, "image/jpeg", ph, parent_metadata=None
    )
    
    # Assertions for 5 main sections
    assert "executive_summary" in forensics, "Missing executive_summary"
    assert "technical_profile" in forensics, "Missing technical_profile"
    assert "forensic_findings" in forensics, "Missing forensic_findings"
    assert "relationship_analysis" in forensics, "Missing relationship_analysis"
    assert "investigation_insights" in forensics, "Missing investigation_insights"
    
    # Assertions for Standalone/Inconclusive checks
    findings = {f["finding"]: f for f in forensics["forensic_findings"]}
    
    assert findings["Metadata stripped"]["status"] in ("Inconclusive", "Not Detected"), "Standalone metadata stripped status error"
    assert findings["Resized"]["status"] == "Inconclusive", "Standalone resizing check must be inconclusive"
    assert findings["Cropped"]["status"] == "Inconclusive", "Standalone cropping check must be inconclusive"
    assert findings["Watermark detected"]["status"] == "Inconclusive", "Standalone watermark check must be inconclusive"
    
    # Technical profile check
    assert "compression_indicators" in forensics["technical_profile"], "Missing compression_indicators in tech profile"
    assert "blockiness" in forensics["technical_profile"]["compression_indicators"], "Missing blockiness in tech profile"
    
    # Executive summary confidence breakdown
    exec_sum = forensics["executive_summary"]
    assert "confidence_score" in exec_sum, "Missing confidence_score in executive summary"
    assert "confidence_factors" in exec_sum, "Missing confidence_factors in executive summary"
    
    factors = exec_sum["confidence_factors"]
    required_factors = [
        "phash_similarity", "dhash_similarity", "ahash_similarity",
        "clip_semantic_similarity", "metadata_consistency",
        "dimension_consistency", "compression_consistency"
    ]
    for key in required_factors:
        assert key in factors, f"Missing confidence factor: {key}"
        assert isinstance(factors[key], int), f"Confidence factor {key} is not an integer"
        
    print("Standalone Blind Investigation Test: PASSED")

def test_blind_investigation_comparative():
    print("\nRunning Comparative Blind Investigation Test...")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_dir = os.path.join(base_dir, "dataset", "case_intel_leak")
    orig_path = os.path.join(dataset_dir, "moon_original.jpg")
    resize_path = os.path.join(dataset_dir, "moon_resized.jpg")
    
    if not os.path.exists(orig_path) or not os.path.exists(resize_path):
        print("Error: moon test images missing.")
        sys.exit(1)
        
    # Set parent metadata
    parent_ph, _, _ = compute_image_hashes(orig_path)
    parent_meta = extract_metadata_signature(orig_path)
    parent_meta["phash"] = parent_ph
    
    # Target (resized variant)
    ph, _, _ = compute_image_hashes(resize_path)
    meta = extract_metadata_signature(resize_path)
    
    integrity, risk, forensics = calculate_integrity_and_risk(
        resize_path, meta, "image/jpeg", ph, parent_metadata=parent_meta
    )
    
    findings = {f["finding"]: f for f in forensics["forensic_findings"]}
    
    # Resized variant should have resizing detected with high confidence
    assert findings["Resized"]["status"] == "Detected", "Resized variant resizing not detected"
    assert findings["Resized"]["confidence"] > 80, "Resized variant resizing detection confidence too low"
    assert len(findings["Resized"]["evidence"]) > 0, "Missing evidence for resized variant"
    
    # Check relationship type classification in relationship analysis
    assert forensics["relationship_analysis"]["relationship_type"] == "Resized Variant", "Incorrect relationship classification"
    
    print("Comparative Blind Investigation Test: PASSED")

if __name__ == "__main__":
    test_blind_investigation_standalone()
    test_blind_investigation_comparative()
    print("\nAll Blind Investigation Tests passed successfully!")