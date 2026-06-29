import os

dataset_dir = r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset"

for root, dirs, files in os.walk(dataset_dir):
    # filter out hidden/git dirs
    dirs[:] = [d for d in dirs if not d.startswith('.')]
    if files:
        # print path and count of files
        images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        if images:
            print(f"Directory: {root} -> Count: {len(images)} images")
            print(f"  Samples: {images[:5]}")
