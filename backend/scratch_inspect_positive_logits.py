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
    
    reals = []
    fakes = []
    
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
                reals.append((fn, logit))
            else:
                fakes.append((fn, logit))
        except Exception as e:
            print(f"Error {fn}: {e}")
            
    reals_pos = [x for x in reals if x[1] >= 0.0]
    fakes_pos = [x for x in fakes if x[1] >= 0.0]
    
    print(f"REAL >= 0.0 count: {len(reals_pos)} out of {len(reals)}")
    for fn, l in reals_pos:
        print(f"  REAL: {fn} logit={l:.4f}")
        
    print(f"FAKE >= 0.0 count: {len(fakes_pos)} out of {len(fakes)}")
    for fn, l in fakes_pos[:10]:
        print(f"  FAKE: {fn} logit={l:.4f}")
    if len(fakes_pos) > 10:
        print("  ...")

if __name__ == "__main__":
    main()
