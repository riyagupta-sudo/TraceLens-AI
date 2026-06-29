import os

search_dir = r"C:\Users\riya2"
target = "1000111612"

print(f"Searching {search_dir} recursively for files containing '{target}'...")
found = []
# Avoid walking AppData if it is too huge, but let's check Downloads, Desktop, Documents, OneDrive
sub_dirs = ["Downloads", "Documents", "OneDrive", "Desktop", ".gemini"]
for sub in sub_dirs:
    p = os.path.join(search_dir, sub)
    if os.path.exists(p):
        print(f"Walking {p}...")
        for root, dirs, files in os.walk(p):
            for file in files:
                if target in file:
                    path = os.path.join(root, file)
                    print(f"FOUND: {path}")
                    found.append(path)

print(f"Done. Found {len(found)} files.")
