import os

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

image_paths = []
for root, dirs, files in os.walk(PROJECT_ROOT):
    # Skip virtual environment
    if ".venv" in root or ".git" in root or "node_modules" in root:
        continue
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            full_path = os.path.join(root, f)
            image_paths.append(full_path)

print(f"Total image files found: {len(image_paths)}")
# Group by directory and print counts
from collections import Counter
dirs = [os.path.dirname(p) for p in image_paths]
for d, count in Counter(dirs).items():
    print(f"  {d}: {count}")
