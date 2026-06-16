import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.database import SessionLocal
from app.models import MediaItem

db = SessionLocal()
try:
    i46 = db.query(MediaItem).filter(MediaItem.id == 46).first()
    i47 = db.query(MediaItem).filter(MediaItem.id == 47).first()
    
    path1 = os.path.join("app", "uploads", os.path.basename(i46.filepath))
    path2 = os.path.join("app", "uploads", os.path.basename(i47.filepath))
    
    print(f"Path 1 (source): {path1}")
    print(f"Path 2 (target): {path2}")
    
    from PIL import Image
    img1_pil = Image.open(path1).convert("L")
    img2_pil = Image.open(path2).convert("L")
    img1 = np.array(img1_pil)
    img2 = np.array(img2_pil)
    
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    print(f"Source size: {w1}x{h1}")
    print(f"Target size: {w2}x{h2}")
    
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    
    inliers = int(np.sum(mask))
    print(f"Inliers: {inliers}")
    
    pts = np.float32([[0, 0], [0, h2 - 1], [w2 - 1, h2 - 1], [w2 - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)
    
    x_coords = dst[:, 0, 0]
    y_coords = dst[:, 0, 1]
    
    left = int(np.min(x_coords))
    right = int(np.max(x_coords))
    top = int(np.min(y_coords))
    bottom = int(np.max(y_coords))
    
    print(f"Transformed coordinates: left={left}, right={right}, top={top}, bottom={bottom}")
    
    contained_standard = (
        left >= -10 and right <= w1 + 10 and
        top >= -10 and bottom <= h1 + 10 and
        (right - left) > 0 and (bottom - top) > 0
    )
    print(f"contained_standard check results:")
    print(f"  left >= -10: {left >= -10} ({left})")
    print(f"  right <= w1 + 10: {right <= w1 + 10} ({right} <= {w1 + 10})")
    print(f"  top >= -10: {top >= -10} ({top})")
    print(f"  bottom <= h1 + 10: {bottom <= h1 + 10} ({bottom} <= {h1 + 10})")
    print(f"  contained_standard: {contained_standard}")

finally:
    db.close()
