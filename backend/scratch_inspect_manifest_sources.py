import os
import json
from collections import Counter

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
VAL_MANIFEST = os.path.join(BACKEND_DIR, "ml", "v2", "validation_pack", "validation_manifest.json")

with open(VAL_MANIFEST, "r") as f:
    manifest = json.load(f)

print(f"Total items in manifest: {len(manifest)}")
sources = [item.get("source") for item in manifest]
labels = [item.get("label") for item in manifest]
combined = [(item.get("source"), item.get("label")) for item in manifest]

print("Source counts:")
for s, count in Counter(sources).items():
    print(f"  {s}: {count}")

print("Label counts:")
for l, count in Counter(labels).items():
    print(f"  {l}: {count}")

print("Source & Label counts:")
for c, count in Counter(combined).items():
    print(f"  {c}: {count}")
