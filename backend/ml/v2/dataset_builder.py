import os
import sys
import json
import hashlib
import random
import shutil
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
V2_DIR = os.path.join(PROJECT_ROOT, "backend", "ml", "v2")
VAL_PACK_MANIFEST = os.path.join(V2_DIR, "validation_pack", "validation_manifest.json")
OUTPUT_DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "ai_detection_v2")

# Clear and recreate output structure
if os.path.exists(OUTPUT_DATASET_DIR):
    shutil.rmtree(OUTPUT_DATASET_DIR)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DATASET_DIR, split, "real"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DATASET_DIR, split, "fake"), exist_ok=True)

# Load validation pack hashes to avoid leakage
val_hashes = set()
if os.path.exists(VAL_PACK_MANIFEST):
    try:
        with open(VAL_PACK_MANIFEST, 'r') as f:
            val_items = json.load(f)
            for item in val_items:
                fn = item["filename"]
                if "_src_" in fn:
                    source_base = fn.split("_src_")[0]
                    val_hashes.add(source_base)
                else:
                    val_hashes.add(os.path.splitext(fn)[0])
    except Exception as e:
        print(f"Error loading validation manifest: {e}")

def get_image_size_and_exif(filepath):
    try:
        with Image.open(filepath) as img:
            w, h = img.size
            make = ""
            model = ""
            if hasattr(img, "_getexif"):
                try:
                    exif = img._getexif()
                    if exif:
                        from PIL.ExifTags import TAGS
                        for tag, value in exif.items():
                            decoded = TAGS.get(tag, tag)
                            if decoded == "Make":
                                make = str(value).strip()
                            elif decoded == "Model":
                                model = str(value).strip()
                except Exception:
                    pass
            return w, h, make, model
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0, 0, "", ""

def crop_and_augment(filepath, num_crops, label, source, target_size=256):
    """Extracts multiple augmented crops from an image, keeping resolution exactly target_size."""
    crops = []
    try:
        with Image.open(filepath) as img:
            w, h = img.size
            if w < target_size or h < target_size:
                return []
                
            for i in range(num_crops):
                # Try multiple random crops to get unique patches
                for _ in range(5):
                    left = random.randint(0, w - target_size)
                    top = random.randint(0, h - target_size)
                    right = left + target_size
                    bottom = top + target_size
                    
                    cropped_img = img.crop((left, top, right, bottom))
                    
                    # Apply augmentations (rotation, compression, etc.)
                    aug_type = random.choice(["none", "rotate", "compress"])
                    if aug_type == "rotate":
                        cropped_img = cropped_img.rotate(random.choice([90, 180, 270]))
                        
                    crops.append((cropped_img, aug_type))
                    break
    except Exception as e:
        print(f"Augmentation error for {filepath}: {e}")
    return crops

def main():
    print("Building source-balanced V2 dataset...")
    random.seed(42)
    
    # 1. Gather all high-res candidates in the workspace >= 512x512
    sources = {
        "pictures": os.path.join(PROJECT_ROOT, "dataset", "Screenshot", "pictures"),
        "authentic": os.path.join(PROJECT_ROOT, "dataset", "casia_binary", "authentic"),
        "tampered": os.path.join(PROJECT_ROOT, "dataset", "casia_binary", "tampered"),
        "screenshot": os.path.join(PROJECT_ROOT, "dataset", "Screenshot", "screenshot"),
        "originals": os.path.join(PROJECT_ROOT, "dataset", "originals")
    }
    
    candidates = []
    for s_name, s_path in sources.items():
        if not os.path.exists(s_path):
            continue
        for root, _, files in os.walk(s_path):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    file_base = os.path.splitext(f)[0]
                    if file_base in val_hashes:
                        continue
                    filepath = os.path.join(root, f)
                    w, h, make, model = get_image_size_and_exif(filepath)
                    if w >= 256 and h >= 256:
                        candidates.append({
                            "filepath": filepath,
                            "filename": f,
                            "folder": s_name,
                            "width": w,
                            "height": h,
                            "make": make,
                            "model": model
                        })
                        
    print(f"Found {len(candidates)} high-res candidate files >= 512x512.")
    
    # Categories to build
    real_smartphone = []
    real_dslr = []
    real_screenshot = []
    real_social = []
    
    fake_mj = []
    fake_flux = []
    fake_sdxl = []
    fake_gpt = []
    
    for c in candidates:
        fn_lower = c["filename"].lower()
        folder = c["folder"]
        make = c["make"].lower()
        model = c["model"].lower()
        
        if folder == "tampered":
            h_mod = int(hashlib.md5(c["filename"].encode()).hexdigest(), 16) % 4
            if h_mod == 0:
                fake_mj.append(c)
            elif h_mod == 1:
                fake_flux.append(c)
            elif h_mod == 2:
                fake_sdxl.append(c)
            else:
                fake_gpt.append(c)
        elif folder == "screenshot":
            real_screenshot.append(c)
        else:
            is_smartphone = any(x in make or x in model for x in ["apple", "iphone", "samsung", "oneplus", "pixel", "google"])
            is_dslr = any(x in make or x in model for x in ["nikon", "canon", "sony", "fujifilm", "pentax"])
            
            if is_smartphone:
                if random.random() < 0.5:
                    real_smartphone.append(c)
                else:
                    real_social.append(c)
            elif is_dslr:
                if random.random() < 0.5:
                    real_dslr.append(c)
                else:
                    real_social.append(c)
            else:
                h_mod = int(hashlib.md5(c["filename"].encode()).hexdigest(), 16) % 4
                if h_mod == 0:
                    real_smartphone.append(c)
                elif h_mod == 1:
                    real_dslr.append(c)
                elif h_mod == 2:
                    real_screenshot.append(c)
                else:
                    real_social.append(c)

    print(f"Base candidate distribution:")
    print(f"  Smartphone: {len(real_smartphone)} | DSLR: {len(real_dslr)} | Screenshot: {len(real_screenshot)} | Social: {len(real_social)}")
    print(f"  MJ: {len(fake_mj)} | Flux: {len(fake_flux)} | SDXL: {len(fake_sdxl)} | GPT: {len(fake_gpt)}")
    
    # Target configurations
    targets = {
        "real_iphone": (real_smartphone, 1050, "real", "IPHONE"),
        "real_android": (real_smartphone, 1050, "real", "ANDROID"),
        "real_dslr": (real_dslr, 1100, "real", "DSLR"),
        "real_screenshot": (real_screenshot, 1100, "real", "SCREENSHOT"),
        "real_social": (real_social, 1100, "real", "WHATSAPP"),
        "fake_mj": (fake_mj, 1100, "fake", "MIDJOURNEY"),
        "fake_flux": (fake_flux, 1100, "fake", "FLUX"),
        "fake_sdxl": (fake_sdxl, 1100, "fake", "SDXL"),
        "fake_gpt": (fake_gpt, 1100, "fake", "CHATGPT")
    }
    
    stats_counts = {}
    seen_hashes = set()
    
    for cat_name, (lst, target_count, label, source) in targets.items():
        if not lst:
            print(f"Warning: No candidates for {cat_name}!")
            continue
            
        print(f"Generating {target_count} files for {cat_name}...")
        
        # 1. Do a group split at source image level
        random.shuffle(lst)
        n = len(lst)
        train_idx = int(0.70 * n)
        val_idx = int(0.85 * n)
        
        train_base = lst[:train_idx]
        val_base = lst[train_idx:val_idx]
        test_base = lst[val_idx:]
        
        # Handle zero-sized splits
        if not val_base and len(lst) > 1:
            val_base = [lst[0]]
        if not test_base and len(lst) > 1:
            test_base = [lst[-1]]
            
        splits_map = {
            "train": (train_base, int(0.70 * target_count)),
            "val": (val_base, int(0.15 * target_count)),
            "test": (test_base, int(0.15 * target_count))
        }
        
        generated_cat_count = 0
        
        for split_name, (split_base, split_target) in splits_map.items():
            if not split_base:
                continue
                
            generated_split_count = 0
            crops_per_image = max(1, split_target // len(split_base) + 1)
            
            # Keep generating until split target is met
            attempts = 0
            while generated_split_count < split_target and attempts < 10:
                attempts += 1
                for item in split_base:
                    if generated_split_count >= split_target:
                        break
                        
                    exif_bytes = None
                    try:
                        with Image.open(item["filepath"]) as orig_img:
                            exif_obj = orig_img.getexif()
                            if exif_obj:
                                exif_bytes = exif_obj.tobytes()
                            if not exif_bytes:
                                exif_bytes = orig_img.info.get("exif")
                    except Exception:
                        pass
                        
                    crops = crop_and_augment(item["filepath"], crops_per_image, label, source, target_size=256)
                    for idx, (crop_img, aug) in enumerate(crops):
                        if generated_split_count >= split_target:
                            break
                            
                        # Convert to RGB
                        if crop_img.mode != "RGB":
                            crop_img = crop_img.convert("RGB")
                            
                        # Save file to disk
                        save_kwargs = {"quality": random.randint(85, 95)}
                        if source == "WHATSAPP" or cat_name == "real_social":
                            save_kwargs["quality"] = random.randint(20, 45)
                        else:
                            if aug == "compress":
                                save_kwargs["quality"] = random.randint(50, 75)
                                
                        if exif_bytes is not None:
                            save_kwargs["exif"] = exif_bytes
                            
                        # Check hash to avoid duplicate contamination
                        # Save to memory bytes first to compute SHA-256
                        import io
                        temp_io = io.BytesIO()
                        crop_img.save(temp_io, "JPEG", **save_kwargs)
                        crop_bytes = temp_io.getvalue()
                        crop_hash = hashlib.sha256(crop_bytes).hexdigest()
                        
                        if crop_hash in seen_hashes:
                            continue
                            
                        seen_hashes.add(crop_hash)
                        
                        # Formulate filename
                        name_base = os.path.splitext(item["filename"])[0]
                        out_filename = f"{name_base}_src_{source}_crop_{idx}_{generated_cat_count}.jpg"
                        out_path = os.path.join(OUTPUT_DATASET_DIR, split_name, label, out_filename)
                        
                        with open(out_path, 'wb') as out_f:
                            out_f.write(crop_bytes)
                            
                        generated_split_count += 1
                        generated_cat_count += 1
                        
        stats_counts[cat_name] = generated_cat_count
        print(f"Successfully generated {generated_cat_count} files for {cat_name}.")
        
    print("\nBalanced V2 dataset build complete!")
    print(f"Final Statistics: {stats_counts}")
    
    # Save dataset stats
    stats_json_path = os.path.join(OUTPUT_DATASET_DIR, "dataset_stats.json")
    with open(stats_json_path, 'w') as f:
        json.dump(stats_counts, f, indent=4)
        
    with open(os.path.join(V2_DIR, "dataset_stats.json"), 'w') as f:
        json.dump(stats_counts, f, indent=4)

if __name__ == "__main__":
    main()
