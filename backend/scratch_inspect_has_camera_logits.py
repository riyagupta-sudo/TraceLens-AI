import os
import json
import numpy as np
import torch
from PIL import Image, ImageOps
import sys

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

from app.dna_engine import AI_MODEL, AI_TRANSFORM

VAL_PACK_DIR = os.path.join(BACKEND_DIR, "ml", "v2", "validation_pack")
VAL_MANIFEST = os.path.join(VAL_PACK_DIR, "validation_manifest.json")

def main():
    with open(VAL_MANIFEST, "r") as f:
        manifest = json.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AI_MODEL.to(device)
    AI_MODEL.eval()
    
    count_fake_with_cam_high_logit = 0
    
    for item in manifest:
        fn = item["filename"]
        lbl = item["label"]
        filepath = os.path.join(VAL_PACK_DIR, "REAL" if lbl == "REAL" else "FAKE", fn)
        if not os.path.exists(filepath):
            continue
        try:
            img = Image.open(filepath)
            exif = img.getexif() or {}
            has_camera = bool(exif.get(271, "") or exif.get(272, ""))
            if lbl == "FAKE" and has_camera:
                img_transposed = ImageOps.exif_transpose(img)
                img_rgb = img_transposed.convert("RGB")
                with torch.no_grad():
                    logit = AI_MODEL(AI_TRANSFORM(img_rgb).unsqueeze(0).to(device)).item()
                if logit >= 2.0:
                    print(f"FAKE with camera: {fn} logit={logit:.4f}")
                    count_fake_with_cam_high_logit += 1
        except Exception as e:
            print(f"Error {fn}: {e}")
            
    print(f"Total FAKE with has_camera=True and logit >= 2.0: {count_fake_with_cam_high_logit}")

if __name__ == "__main__":
    main()
