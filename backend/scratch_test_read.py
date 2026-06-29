import os
import cv2
from PIL import Image

f1 = r"C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\originals\human_006.jpg"
print("File exists:", os.path.exists(f1))
print("File size:", os.path.getsize(f1))

try:
    with Image.open(f1) as img:
        print("PIL open: Success! Size:", img.size, "Mode:", img.mode)
except Exception as e:
    print("PIL open: Failed!", e)

try:
    img_cv = cv2.imread(f1)
    if img_cv is None:
        print("cv2.imread: Failed! Returned None")
    else:
        print("cv2.imread: Success! Shape:", img_cv.shape)
except Exception as e:
    print("cv2.imread: Exception!", e)
