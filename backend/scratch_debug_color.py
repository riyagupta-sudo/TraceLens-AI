import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "app")))
from app.similarity_engine import compute_color_histogram_similarity

dataset_dir = r"C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset"
f1 = os.path.join(dataset_dir, "originals", "human_006.jpg")
f2 = os.path.join(dataset_dir, "resized", "human_006_resize.jpg")

print("f1 path:", f1)
print("f2 path:", f2)

# Test compute_color_histogram_similarity without crop bounds
sim_no_crop = compute_color_histogram_similarity(f1, f2, None)
print("Similarity without crop bounds:", sim_no_crop)

# Let's inspect what happens inside with crop bounds
crop_bounds = {"left": 0, "right": 500, "top": 0, "bottom": 500} # dummy crop bounds
sim_with_crop = compute_color_histogram_similarity(f1, f2, crop_bounds)
print("Similarity with crop bounds:", sim_with_crop)

# Read images directly
img1 = cv2.imread(f1)
img2 = cv2.imread(f2)
print("img1 is None:", img1 is None)
print("img2 is None:", img2 is None)
if img1 is not None and img2 is not None:
    print("img1 shape:", img1.shape)
    print("img2 shape:", img2.shape)
