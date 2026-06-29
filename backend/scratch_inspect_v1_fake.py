import os
from PIL import Image
from PIL.ExifTags import TAGS

project_root = r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI"
v1_fake_dir = os.path.join(project_root, "dataset", "ai_detection", "train", "fake")

def get_image_info(filepath):
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
            return make, model, img.size, os.path.getsize(filepath)
    except Exception:
        return "", "", (0, 0), 0

def main():
    if not os.path.exists(v1_fake_dir):
        print(f"V1 fake directory {v1_fake_dir} does not exist.")
        return
        
    files = [f for f in os.listdir(v1_fake_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    print(f"Total images in V1 train/fake: {len(files)}")
    
    if not files:
        return
        
    exif_count = 0
    total_w = 0
    total_h = 0
    makes = {}
    models = {}
    
    for f in files[:200]: # Sample 200 files
        filepath = os.path.join(v1_fake_dir, f)
        make, model, size, size_bytes = get_image_info(filepath)
        w, h = size
        total_w += w
        total_h += h
        if make or model:
            exif_count += 1
            makes[make] = makes.get(make, 0) + 1
            models[model] = models.get(model, 0) + 1
            
    avg_res = f"{int(total_w/min(len(files), 200))}x{int(total_h/min(len(files), 200))}"
    exif_rate = exif_count / min(len(files), 200)
    
    print(f"Sample-based Statistics (on 200 files):")
    print(f"  EXIF Availability: {exif_rate*100:.2f}%")
    print(f"  Average Resolution: {avg_res}")
    if makes:
        print("  Top Camera Makes:", sorted(makes.items(), key=lambda x: x[1], reverse=True)[:5])
        print("  Top Camera Models:", sorted(models.items(), key=lambda x: x[1], reverse=True)[:5])
    else:
        print("  No EXIF Make/Model found.")

if __name__ == "__main__":
    main()
