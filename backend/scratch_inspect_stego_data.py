import os
from PIL import Image
from PIL.ExifTags import TAGS

project_root = r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI"
stego_dir = os.path.join(project_root, "dataset", "Steganography")

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
    if not os.path.exists(stego_dir):
        print(f"Directory {stego_dir} does not exist.")
        return
        
    all_files = []
    for root, dirs, files in os.walk(stego_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                all_files.append(os.path.join(root, f))
                
    print(f"Total files in Steganography: {len(all_files)}")
    
    if not all_files:
        return
        
    # Analyze a sample of 200 files
    exif_count = 0
    total_w = 0
    total_h = 0
    makes = {}
    models = {}
    
    import random
    random.seed(42)
    sample = random.sample(all_files, min(len(all_files), 200))
    
    for filepath in sample:
        make, model, size, bytes_size = get_image_info(filepath)
        w, h = size
        total_w += w
        total_h += h
        if make or model:
            exif_count += 1
            makes[make] = makes.get(make, 0) + 1
            models[model] = models.get(model, 0) + 1
            
    avg_res = f"{int(total_w/len(sample))}x{int(total_h/len(sample))}"
    exif_rate = exif_count / len(sample)
    
    print(f"Sample-based Statistics (on 200 files):")
    print(f"  EXIF Availability: {exif_rate*100:.2f}%")
    print(f"  Average Resolution: {avg_res}")
    print(f"  Sample paths: {[os.path.basename(p) for p in sample[:5]]}")
    if makes:
        print("  Top Camera Makes:", sorted(makes.items(), key=lambda x: x[1], reverse=True)[:5])
        print("  Top Camera Models:", sorted(models.items(), key=lambda x: x[1], reverse=True)[:5])
    else:
        print("  No EXIF Make/Model found.")

if __name__ == "__main__":
    main()
