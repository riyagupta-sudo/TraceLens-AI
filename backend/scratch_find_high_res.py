import os
from PIL import Image

dataset_dir = r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset"

folder_counts = {}

for root, dirs, files in os.walk(dataset_dir):
    # filter out hidden
    dirs[:] = [d for d in dirs if not d.startswith('.')]
    count_gte_256 = 0
    sample_files = []
    
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            filepath = os.path.join(root, f)
            try:
                # Use PIL to read size without loading pixels (very fast)
                with Image.open(filepath) as img:
                    w, h = img.size
                    if w >= 256 and h >= 256:
                        count_gte_256 += 1
                        if len(sample_files) < 3:
                            sample_files.append((f, f"{w}x{h}"))
            except Exception:
                pass
                
    if count_gte_256 > 0:
        folder_counts[root] = {
            "count": count_gte_256,
            "samples": sample_files
        }

print("\n--- Folders with images >= 256x256 ---")
for folder, info in sorted(folder_counts.items(), key=lambda x: x[1]["count"], reverse=True):
    print(f"Folder: {folder}")
    print(f"  Count: {info['count']} images")
    print(f"  Samples: {info['samples']}")
