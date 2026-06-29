f1 = r"C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\originals\human_006.jpg"
f2 = r"C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\resized\human_006_resize.jpg"

with open(f1, "rb") as f:
    header1 = f.read(32)
with open(f2, "rb") as f:
    header2 = f.read(32)

print("human_006.jpg header:", header1)
print("human_006_resize.jpg header:", header2)
