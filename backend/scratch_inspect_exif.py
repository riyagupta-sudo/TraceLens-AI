import os
from PIL import Image
from PIL.ExifTags import TAGS

dirs_to_check = {
    "originals": r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\originals",
    "ai_detection_train_real": r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\ai_detection\train\REAL",
    "ai_detection_test_real": r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\ai_detection\test\REAL",
    "screenshot_pictures": r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures",
}

for name, path in dirs_to_check.items():
    print(f"\n--- Checking EXIF for {name} ({path}) ---")
    files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        print("No files found.")
        continue
    
    # Read EXIF of first 10 files
    checked = 0
    for f in files:
        if checked >= 10:
            break
        filepath = os.path.join(path, f)
        try:
            with Image.open(filepath) as img:
                exif = img._getexif()
                if exif:
                    info = {}
                    for tag, value in exif.items():
                        decoded = TAGS.get(tag, tag)
                        if decoded in ["Make", "Model", "Software", "DateTimeOriginal"]:
                            info[decoded] = value
                    if info:
                        print(f"File {f}: {info}")
                        checked += 1
                else:
                    # Check if there is any exif at all
                    pass
        except Exception as e:
            print(f"Error {f}: {e}")
