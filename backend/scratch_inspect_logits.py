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
    
    real_logits = []
    fake_logits = []
    
    for item in manifest:
        fn = item["filename"]
        lbl = item["label"]
        filepath = os.path.join(VAL_PACK_DIR, "REAL" if lbl == "REAL" else "FAKE", fn)
        if not os.path.exists(filepath):
            continue
        try:
            img = Image.open(filepath)
            img_transposed = ImageOps.exif_transpose(img)
            img_rgb = img_transposed.convert("RGB")
            with torch.no_grad():
                logit = AI_MODEL(AI_TRANSFORM(img_rgb).unsqueeze(0).to(device)).item()
            if lbl == "REAL":
                real_logits.append(logit)
            else:
                fake_logits.append(logit)
        except Exception as e:
            print(f"Error {fn}: {e}")
            
    print(f"REAL: count={len(real_logits)}, min={np.min(real_logits):.4f}, max={np.max(real_logits):.4f}, mean={np.mean(real_logits):.4f}")
    print(f"FAKE: count={len(fake_logits)}, min={np.min(fake_logits):.4f}, max={np.max(fake_logits):.4f}, mean={np.mean(fake_logits):.4f}")

if __name__ == "__main__":
    main()
