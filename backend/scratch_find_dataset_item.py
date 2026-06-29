import os

dataset_dir = r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset"
print(f"Searching {dataset_dir} recursively...")
found = []
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if '1612' in file:
            found.append(os.path.join(root, file))

print(f"Found {len(found)} files:")
for f in found:
    print(f)
