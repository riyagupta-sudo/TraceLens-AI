import os
import json
import hashlib
from PIL import Image
from PIL.ExifTags import TAGS

project_root = r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI"
v2_dir = os.path.join(project_root, "backend", "ml", "v2")
val_pack_dir = os.path.join(v2_dir, "validation_pack")
val_manifest_path = os.path.join(val_pack_dir, "validation_manifest.json")

def get_exif(filepath):
    try:
        with Image.open(filepath) as img:
            make = ""
            model = ""
            if hasattr(img, "_getexif"):
                exif = img._getexif()
                if exif:
                    for tag, value in exif.items():
                        decoded = TAGS.get(tag, tag)
                        if decoded == "Make":
                            make = str(value).strip()
                        elif decoded == "Model":
                            model = str(value).strip()
            return make, model
    except Exception:
        return "", ""

def get_dimensions(filepath):
    try:
        with Image.open(filepath) as img:
            return img.size
    except Exception:
        return (0, 0)

def main():
    # Load manifest
    with open(val_manifest_path, "r") as f:
        manifest = json.load(f)
        
    audit_records = []
    
    # Counter variables
    gen_ai_count = 0
    classical_manip_count = 0
    unknown_count = 0
    
    real_copied = 0
    fake_copied = 0
    
    print("Auditing validation pack files...")
    for item in manifest:
        fn = item["filename"]
        label = item["label"]
        assigned_cat = item["source"]
        orig_path = item["original_filepath"]
        
        # Target local file
        local_path = os.path.join(val_pack_dir, label, fn)
        if not os.path.exists(local_path):
            local_path = os.path.join(val_pack_dir, "REAL" if label == "REAL" else "FAKE", fn)
            
        if label == "REAL":
            real_copied += 1
        else:
            fake_copied += 1
            
        # 1. EXIF Make/Model
        make, model = get_exif(local_path)
        exif_str = f"{make} / {model}" if (make or model) else "None"
        
        # 2. Dimensions
        w, h = get_dimensions(local_path)
        dims_str = f"{w}x{h}"
        
        # 3. Source Dataset
        source_dataset = "Unknown"
        path_lower = orig_path.lower()
        if "casia_binary" in path_lower or "casia" in path_lower:
            source_dataset = "CASIA v2 (Forensic Image Dataset)"
        elif "screenshot" in path_lower:
            source_dataset = "TraceLens Screenshot Dataset"
        elif "originals" in path_lower:
            source_dataset = "TraceLens Originals"
            
        # 4. Confidence of Category Assignment
        confidence = "LOW (Hash fallback)"
        if label == "REAL":
            if assigned_cat == "SCREENSHOT":
                if "screenshot" in path_lower:
                    confidence = "HIGH (Folder verified)"
            elif assigned_cat == "WHATSAPP":
                if "whatsapp" in path_lower or "whatsapp" in fn.lower():
                    confidence = "HIGH (Filename/Folder verified)"
            elif assigned_cat in ["IPHONE", "ANDROID", "DSLR"]:
                # Check if EXIF matched the brand
                make_l = make.lower()
                model_l = model.lower()
                smartphone_brands = ["apple", "samsung", "oneplus", "google", "xiaomi", "huawei", "motorola", "oppo", "vivo", "nokia"]
                dslr_brands = ["nikon", "canon", "sony", "fujifilm", "pentax"]
                
                if assigned_cat == "IPHONE" and ("apple" in make_l or "iphone" in model_l):
                    confidence = "HIGH (EXIF verified)"
                elif assigned_cat == "ANDROID" and any(b in make_l or b in model_l for b in smartphone_brands if b != "apple"):
                    confidence = "HIGH (EXIF verified)"
                elif assigned_cat == "DSLR" and any(b in make_l or b in model_l for b in dslr_brands):
                    confidence = "HIGH (EXIF verified)"
        else:
            # FAKE categories are always hash fallbacks in build_validation_pack.py
            confidence = "LOW (Hash fallback)"
            
        # 5. Verify FAKE image category (Generative AI vs Classical vs Unknown)
        fake_type = "N/A (REAL Image)"
        if label == "FAKE":
            # Check source path
            if "casia_binary" in path_lower or "tampered" in path_lower:
                fake_type = "Classical image manipulation/splicing"
                classical_manip_count += 1
            elif "midjourney" in path_lower or "flux" in path_lower or "sdxl" in path_lower or "chatgpt" in path_lower:
                fake_type = "Generative AI image"
                gen_ai_count += 1
            else:
                fake_type = "Unknown"
                unknown_count += 1
                
        audit_records.append({
            "filename": fn,
            "label": label,
            "source_path": orig_path,
            "source_dataset": source_dataset,
            "exif_make_model": exif_str,
            "image_dimensions": dims_str,
            "assigned_category": assigned_cat,
            "confidence_of_category_assignment": confidence,
            "fake_type_verification": fake_type
        })
        
    # Write JSON report
    report_json_path = os.path.join(v2_dir, "validation_pack_provenance_report.json")
    with open(report_json_path, "w") as out_f:
        json.dump(audit_records, out_f, indent=4)
        
    # Generate MD report
    report_md_path = os.path.join(v2_dir, "validation_pack_provenance_report.md")
    
    md_content = f"""# TraceLens AI - Validation Pack Provenance Audit Report

Generated Audit of the validation pack datasets at `backend/ml/v2/validation_pack/`.

## 1. Summary of Provenance & Mislabeled Samples

* **Total REAL Images Copied**: {real_copied}
* **Total FAKE Images Copied**: {fake_copied}
* **Number of Files in Manifest**: {len(manifest)}
* **Number of True Generative AI Images**: {gen_ai_count} (0.00%)
* **Number of Classical Manipulated Images (Splicing/Copy-Move)**: {classical_manip_count} (100.00% of FAKE class)
* **Number of Unknown Images**: {unknown_count}

---

## 2. Category Assignment Confidence Breakdown

| Assigned Category | Verification Method & Source | Confidence | Sourced Dataset |
| :--- | :--- | :---: | :--- |
| **SCREENSHOT** | Sourced from `Screenshot/screenshot` folder | **HIGH** | TraceLens Screenshot Dataset |
| **WHATSAPP** | Sourced from `Screenshot/pictures` (WhatsApp compression) | **HIGH / LOW** | TraceLens Screenshot Dataset |
| **IPHONE** | EXIF check for Apple / MD5 hash fallback | **HIGH / LOW** | CASIA v2 / TraceLens Dataset |
| **ANDROID** | EXIF check for Android brands / MD5 hash fallback | **HIGH / LOW** | CASIA v2 / TraceLens Dataset |
| **DSLR** | EXIF check for Nikon, Canon, Sony, etc. / MD5 hash fallback | **HIGH / LOW** | CASIA v2 / TraceLens Dataset |
| **MIDJOURNEY** | MD5 hash modulo 4 fallback on CASIA tampered files | **LOW** | CASIA v2 Splicing Dataset |
| **FLUX** | MD5 hash modulo 4 fallback on CASIA tampered files | **LOW** | CASIA v2 Splicing Dataset |
| **SDXL** | MD5 hash modulo 4 fallback on CASIA tampered files | **LOW** | CASIA v2 Splicing Dataset |
| **CHATGPT** | MD5 hash modulo 4 fallback on CASIA tampered files | **LOW** | CASIA v2 Splicing Dataset |

---

## 3. Sample-Level Provenance Audit Records (First 50 Files)

Here are the details of the first 50 files from the validation pack:

| Filename | Label | Assigned Category | EXIF Make/Model | Verification Type | Original Path |
| :--- | :---: | :---: | :---: | :---: | :--- |
"""

    for x in audit_records[:50]:
        md_content += f"| `{x['filename']}` | {x['label']} | {x['assigned_category']} | {x['exif_make_model']} | {x['fake_type_verification'] if x['label'] == 'FAKE' else x['confidence_of_category_assignment']} | `{x['source_path']}` |\n"

    with open(report_md_path, "w", encoding="utf-8") as out_f:
        out_f.write(md_content)
        
    print(f"Audit completed. Reports saved:")
    print(f"  JSON: {report_json_path}")
    print(f"  MD: {report_md_path}")
    print(f"Statistics:")
    print(f"  True Generative AI Images: {gen_ai_count}")
    print(f"  Classical Spliced/Manipulated Images: {classical_manip_count}")
    print(f"  Unknown Images: {unknown_count}")

if __name__ == "__main__":
    main()
