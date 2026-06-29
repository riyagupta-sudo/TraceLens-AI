import os
import json

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
VAL_MANIFEST = os.path.join(BACKEND_DIR, "ml", "v2", "validation_pack", "validation_manifest.json")

with open(VAL_MANIFEST, "r") as f:
    manifest = json.load(f)

keywords = ["gmail", "eraser", "magic", "removal", "edit", "casia"]
found = {kw: [] for kw in keywords}

for item in manifest:
    fn = item["filename"].lower()
    src = item.get("source", "").lower()
    for kw in keywords:
        if kw in fn or kw in src:
            found[kw].append(item)

for kw, items in found.items():
    print(f"Keyword '{kw}': found {len(items)} items")
    if items:
        print(f"  Sample: {items[:3]}")
