import os
from PIL import Image

dirs = [
    r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures",
    r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\casia_binary\authentic",
    r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\casia_binary\tampered",
    r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\originals",
    r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\screenshot",
]

for d in dirs:
    if not os.path.exists(d):
        print(f"Directory {d} does not exist.")
        continue
    
    count_gte_256 = 0
    total = 0
    for root, _, files in os.walk(d):
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
    print(f"Directory: {d} | Total: {total} | GTE 256x256: {count_gte_256}")
