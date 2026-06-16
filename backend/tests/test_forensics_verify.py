import os
import shutil
import json
from PIL import Image

# Add backend to python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.dna_engine import (
    calculate_integrity_and_risk,
    extract_metadata_signature,
    compute_image_hashes
)

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_dir = os.path.join(base_dir, "dataset", "case_intel_leak")
    
    # 1. Ensure moon_original.jpg and moon_compressed.jpg exist in the dataset folder
    moon_path = os.path.join(dataset_dir, "moon.jpg")
    moon_com_path = os.path.join(dataset_dir, "moon_com.jpg")
    
    moon_orig_path = os.path.join(dataset_dir, "moon_original.jpg")
    moon_comp_path = os.path.join(dataset_dir, "moon_compressed.jpg")
    
    if os.path.exists(moon_path) and not os.path.exists(moon_orig_path):
        shutil.copy(moon_path, moon_orig_path)
        print(f"Copied moon.jpg to moon_original.jpg")
        
    if os.path.exists(moon_com_path) and not os.path.exists(moon_comp_path):
        shutil.copy(moon_com_path, moon_comp_path)
        print(f"Copied moon_com.jpg to moon_compressed.jpg")
        
    if not os.path.exists(moon_orig_path) or not os.path.exists(moon_comp_path):
        print("Error: moon images not found in dataset folder!")
        sys.exit(1)
        
    # 2. Generate moon_cropped.jpg programmatically
    moon_crop_path = os.path.join(dataset_dir, "moon_cropped.jpg")
    with Image.open(moon_orig_path) as img:
        w, h = img.size
        # Crop center 30%
        left = int(w * 0.15)
        top = int(h * 0.15)
        right = int(w * 0.85)
        bottom = int(h * 0.85)
        cropped_img = img.crop((left, top, right, bottom))
        cropped_img.save(moon_crop_path, "JPEG", quality=95)
    print(f"Generated moon_cropped.jpg programmatically")
        
    # 3. Generate moon_resized.jpg programmatically
    moon_resize_path = os.path.join(dataset_dir, "moon_resized.jpg")
    with Image.open(moon_orig_path) as img:
        # Resize down by 50%
        resized_img = img.resize((w // 2, h // 2), Image.Resampling.BILINEAR)
        resized_img.save(moon_resize_path, "JPEG", quality=95)
    print(f"Generated moon_resized.jpg programmatically")
    
    # 4. Extract signatures and compute scores
    # Original parent metadata
    orig_ph, _, _ = compute_image_hashes(moon_orig_path)
    orig_meta = extract_metadata_signature(moon_orig_path)
    orig_parent_meta = {
        "width": orig_meta.get("width", 0),
        "height": orig_meta.get("height", 0),
        "phash": orig_ph,
        "exif": orig_meta.get("exif", {})
    }
    
    # Target files to test
    targets = [
        ("Original (moon_original.jpg)", moon_orig_path, None, "image/jpeg"),
        ("Compressed (moon_compressed.jpg)", moon_comp_path, orig_parent_meta, "image/jpeg"),
        ("Cropped (moon_cropped.jpg)", moon_crop_path, orig_parent_meta, "image/jpeg"),
        ("Resized (moon_resized.jpg)", moon_resize_path, orig_parent_meta, "image/jpeg"),
    ]
    
    results = {}
    
    print("\n" + "="*80)
    print("FORENSIC SCORES ANALYSIS")
    print("="*80)
    
    for name, filepath, parent_meta, mime in targets:
        ph, _, _ = compute_image_hashes(filepath)
        meta = extract_metadata_signature(filepath)
        # We also override filename in metadata for the keyword check
        meta["filename"] = os.path.basename(filepath)
        
        integrity, risk, forensics = calculate_integrity_and_risk(
            filepath, meta, mime, ph, parent_metadata=parent_meta
        )
        results[name] = (integrity, risk, forensics)
        
        print(f"\nAsset: {name}")
        print(f"  Dimensions: {meta.get('width')}x{meta.get('height')} | Size: {meta.get('file_size')} bytes")
        print(f"  Integrity Score: {integrity}/100")
        print(f"  Risk Score:      {risk}/100")
        print(f"  Indicators:      {forensics}")
        
    print("\n" + "="*80)
    print("VERIFICATION CHECKS")
    print("="*80)
    
    # Verify each receives different scores
    scores = [(integrity, risk) for integrity, risk, _ in results.values()]
    unique_scores = set(scores)
    
    print(f"Total test assets: {len(scores)}")
    print(f"Unique forensic score pairs: {len(unique_scores)}")
    
    if len(unique_scores) == len(scores):
        print("\nSUCCESS: All test assets produce distinct forensic scores!")
    else:
        print("\nWARNING: Some assets have identical forensic scores! Check implementation logic.")
        sys.exit(1)
        
if __name__ == "__main__":
    main()
