import os
from PIL import Image
from PIL.ExifTags import TAGS

project_root = r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI"
dataset_dir = os.path.join(project_root, "dataset")

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
            return make, model, img.size
    except Exception:
        return "", "", (0, 0)

def main():
    folders_to_scan = [
        "Screenshot/pictures",
        "Screenshot/screenshot",
        "casia_binary/authentic",
        "casia_binary/tampered",
        "originals",
        "compressed",
        "cropped",
        "resized",
        "watermarked"
    ]
    
    print("Auditing raw candidate datasets in workspace...")
    
    for folder in folders_to_scan:
        folder_path = os.path.join(dataset_dir, folder)
        if not os.path.exists(folder_path):
            print(f"Directory {folder} does not exist.")
            continue
            
        img_count = 0
        exif_count = 0
        makes = {}
        models = {}
        total_w = 0
        total_h = 0
        
        for root, _, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    filepath = os.path.join(root, f)
                    img_count += 1
                    make, model, size = get_exif(filepath)
                    w, h = size
                    total_w += w
                    total_h += h
                    if make or model:
                        exif_count += 1
                        makes[make] = makes.get(make, 0) + 1
                        models[model] = models.get(model, 0) + 1
                        
        avg_res = f"{int(total_w/img_count)}x{int(total_h/img_count)}" if img_count > 0 else "N/A"
        exif_rate = exif_count / img_count if img_count > 0 else 0.0
        
        print(f"\nFolder: dataset/{folder}")
        print(f"  Total Images: {img_count}")
        print(f"  EXIF Availability: {exif_rate*100:.2f}% ({exif_count} files)")
        print(f"  Average Resolution: {avg_res}")
        if makes:
            print("  Top Camera Makes:", sorted(makes.items(), key=lambda x: x[1], reverse=True)[:5])
            print("  Top Camera Models:", sorted(models.items(), key=lambda x: x[1], reverse=True)[:5])
        else:
            print("  No EXIF Make/Model found.")

if __name__ == "__main__":
    main()
