import os
import cv2
import numpy as np

f1 = r"C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\originals\human_006.jpg"

try:
    with open(f1, "rb") as f:
        file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_cv is None:
        print("cv2.imdecode: Failed! Returned None")
    else:
        print("cv2.imdecode: Success! Shape:", img_cv.shape)
except Exception as e:
    print("cv2.imdecode: Exception!", e)
