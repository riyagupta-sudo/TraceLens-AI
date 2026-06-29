import sys
import os
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "app")))

from app.similarity_engine import analyze_matches, save_heavy_features
from app.dna_engine import compute_sha256, compute_image_hashes

dataset_dir = r"C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset"
f1 = os.path.join(dataset_dir, "originals", "human_006.jpg")
f2 = os.path.join(dataset_dir, "resized", "human_006_resize.jpg")

print("f1 exists:", os.path.exists(f1))
print("f2 exists:", os.path.exists(f2))

import cv2
img1 = cv2.imread(f1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(f2, cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

p1 = save_heavy_features(os.path.basename(f1), kp1, des1)
p2 = save_heavy_features(os.path.basename(f2), kp2, des2)

# Generate DNA profiles
h1 = compute_image_hashes(f1)
h2 = compute_image_hashes(f2)

import os
from PIL import Image

def get_dna(filepath, ph, dh, ah, cache_path):
    img = Image.open(filepath)
    return {
        "filepath": filepath,
        "filename": os.path.basename(filepath),
        "sha256": "dummy",
        "phash": ph,
        "dhash": dh,
        "ahash": ah,
        "width": img.width,
        "height": img.height,
        "file_size": os.path.getsize(filepath),
        "feature_cache_path": cache_path,
        "embedding": None
    }

dna1 = get_dna(f1, h1[0], h1[1], h1[2], p1)
dna2 = get_dna(f2, h2[0], h2[1], h2[2], p2)

score, match_lvl, forensics = analyze_matches(dna1, dna2)
print("Score:", score)
print("Match Level:", match_lvl)
print("Forensics:")
print(json.dumps(forensics, indent=2))
