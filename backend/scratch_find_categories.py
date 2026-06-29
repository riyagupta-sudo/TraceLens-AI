import os
import json

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
VAL_PACK_DIR = os.path.join(BACKEND_DIR, "ml", "v2", "validation_pack")

# Let's list files in REAL and FAKE
real_files = os.listdir(os.path.join(VAL_PACK_DIR, "REAL"))
fake_files = os.listdir(os.path.join(VAL_PACK_DIR, "FAKE"))

print(f"REAL folder file count: {len(real_files)}")
print(f"FAKE folder file count: {len(fake_files)}")

# Check filenames
print("\nSample REAL files:")
for f in real_files[:10]:
    print("  ", f)

print("\nSample FAKE files:")
for f in fake_files[:10]:
    print("  ", f)

# Let's search if there are other directories in backend/dataset or elsewhere
print("\nSearching other directories under backend/dataset:")
for root, dirs, files in os.walk(os.path.join(BACKEND_DIR, "dataset")):
    if len(files) > 0:
        print(f"  {root}: {len(files)} files")
        # print sample
        print(f"    Sample: {files[:3]}")
