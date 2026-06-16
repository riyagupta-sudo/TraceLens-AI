import os
from PIL import Image

dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "originals"))
print(f"Dataset dir: {dataset_dir}")
if os.path.exists(dataset_dir):
    files = os.listdir(dataset_dir)
    for f in files:
        path = os.path.join(dataset_dir, f)
        if os.path.isfile(path):
            try:
                with Image.open(path) as img:
                    print(f"File: {f} | Size: {os.path.getsize(path)} | Dim: {img.width}x{img.height}")
            except Exception as e:
                print(f"File: {f} | Size: {os.path.getsize(path)} | Error: {e}")
else:
    print("Dataset directory does not exist")
