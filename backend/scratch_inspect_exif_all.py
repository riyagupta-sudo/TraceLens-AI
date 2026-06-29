import os
from PIL import Image
from PIL.ExifTags import TAGS

dataset_dir = r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset"

makes_models = {}

for root, dirs, files in os.walk(dataset_dir):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            filepath = os.path.join(root, f)
            try:
                with Image.open(filepath) as img:
                    if hasattr(img, "_getexif"):
                        exif = img._getexif()
                        if exif:
                            make = None
                            model = None
                            for tag, value in exif.items():
                                decoded = TAGS.get(tag, tag)
                                if decoded == "Make":
                                    make = value
                                elif decoded == "Model":
                                    model = value
                            if make or model:
                                key = (str(make).strip(), str(model).strip())
                                if key not in makes_models:
                                    makes_models[key] = []
                                if len(makes_models[key]) < 3:
                                    makes_models[key].append(filepath)
            except Exception as e:
                pass

print("Unique Makes and Models found:")
for (make, model), sample_files in makes_models.items():
    print(f"Make: {make} | Model: {model} | Sample: {sample_files[0]}")
