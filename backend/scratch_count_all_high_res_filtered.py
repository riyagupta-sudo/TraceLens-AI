import os
from PIL import Image

dataset_dir = r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset"

for d in os.listdir(dataset_dir):
    d_path = os.path.join(dataset_dir, d)
    if not os.path.isdir(d_path):
        continue
    if d in ["ai_detection", "Steganography"]:
        continue
        
    count_gte_256 = 0
    total = 0
    for root, _, files in os.walk(d_path):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                total += 1
                filepath = os.path.join(root, f)
                try:
                    with Image.open(filepath) as img:
                        w, h = img.size
                        if w >= 256 and h >= 256:
                            count_gte_256 += 1
                except Exception:
                    pass
    print(f"Folder: {d} | Total: {total} | GTE 256x256: {count_gte_256}")
