import os

for root, dirs, files in os.walk("dataset"):
    image_count = sum(
        1 for f in files
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    )

    if image_count > 0:
        print(f"{root}: {image_count}")