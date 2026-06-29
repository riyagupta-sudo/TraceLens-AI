import os
import json

project_root = r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI"
manifest_path = os.path.join(project_root, "backend", "ml", "v2", "validation_pack", "validation_manifest.json")
dataset_dir = os.path.join(project_root, "dataset", "ai_detection_v2")

with open(manifest_path, "r") as f:
    manifest = json.load(f)

print(f"Total validation items: {len(manifest)}")
sources = {}
labels = {}
for x in manifest:
    sources[x["source"]] = sources.get(x["source"], 0) + 1
    labels[x["label"]] = labels.get(x["label"], 0) + 1

print("Sources count:")
for k, v in sorted(sources.items()):
    print(f"  {k}: {v}")
print("Labels count:")
for k, v in sorted(labels.items()):
    print(f"  {k}: {v}")

# Verify leakage
val_bases = set()
for x in manifest:
    fn = x["filename"]
    if "_src_" in fn:
        base = fn.split("_src_")[0]
    else:
        base = os.path.splitext(fn)[0]
    val_bases.add(base)

leaked = []
for root, dirs, files in os.walk(dataset_dir):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            if "_src_" in f:
                base = f.split("_src_")[0]
            else:
                base = os.path.splitext(f)[0]
            if base in val_bases:
                leaked.append(os.path.join(root, f))

print(f"Leaked files found: {len(leaked)}")
if leaked:
    print("Sample leaked paths:", leaked[:10])
else:
    print("Leakage check PASSED: 0 leaked files between validation pack and dataset splits!")
