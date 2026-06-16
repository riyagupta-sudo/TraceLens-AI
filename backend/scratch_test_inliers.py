import os
import cv2
import numpy as np

def test_containment(img1_path, img2_path):
    default_res = (False, 0, 0)
    img1_path = os.path.normpath(img1_path)
    img2_path = os.path.normpath(img2_path)
    
    print(f"\nChecking: {img1_path} exists={os.path.exists(img1_path)}")
    print(f"Checking: {img2_path} exists={os.path.exists(img2_path)}")
    
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        return default_res
        
    try:
        from PIL import Image
        img1_pil = Image.open(img1_path).convert("L")
        img2_pil = Image.open(img2_path).convert("L")
        img1 = np.array(img1_pil)
        img2 = np.array(img2_pil)
        if img1 is None or img2 is None:
            print(f"  FAILED to convert PIL to numpy: img1 is None={img1 is None}, img2 is None={img2 is None}")
            return default_res
            
        h1, w1 = img1.shape
        h2, w2 = img2.shape
        orb = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        print(f"  Image 1 size: {w1}x{h1}, keypoints: {len(kp1) if kp1 else 0}, descriptors: {des1.shape if des1 is not None else None}")
        print(f"  Image 2 size: {w2}x{h2}, keypoints: {len(kp2) if kp2 else 0}, descriptors: {des2.shape if des2 is not None else None}")
        
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return default_res
            
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if len(matches) < 6:
            return default_res
            
        matches = sorted(matches, key=lambda x: x.distance)
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            return default_res
            
        inliers = int(np.sum(mask))
        
        # Calculate corners and containment
        pts = np.float32([[0, 0], [0, h2 - 1], [w2 - 1, h2 - 1], [w2 - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)
        x_coords = dst[:, 0, 0]
        y_coords = dst[:, 0, 1]
        
        left = int(np.min(x_coords))
        right = int(np.max(x_coords))
        top = int(np.min(y_coords))
        bottom = int(np.max(y_coords))
        
        contained = (
            left >= -10 and right <= w1 + 10 and
            top >= -10 and bottom <= h1 + 10 and
            (right - left) > 0 and (bottom - top) > 0
        )
        
        return contained, inliers, len(matches)
    except Exception as e:
        print(f"Error: {e}")
        return default_res

dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))

tests = [
    # True positives (actual crops)
    ("originals/human_006.jpg", "cropped/human_006_crop.jpg", "Human Crop (True Positive)"),
    ("originals/building_001.jpg", "cropped/building_001_crop.jpg", "Building Crop (True Positive)"),
    # False positives (unrelated)
    ("originals/human_006.jpg", "case_intel_leak/drone_orignal.jpg", "Human vs Drone (False Positive)"),
    ("resized/human_006_resize.jpg", "case_intel_leak/drone_orignal.jpg", "Human Resize vs Drone (False Positive)"),
    ("originals/building_001.jpg", "originals/human_006.jpg", "Building vs Human (False Positive)")
]

for p1, p2, label in tests:
    path1 = os.path.join(dataset_dir, p1)
    path2 = os.path.join(dataset_dir, p2)
    contained, inliers, total_matches = test_containment(path1, path2)
    print(f"{label}:")
    print(f"  Contained: {contained} | Inliers: {inliers} | Total Matches: {total_matches}")
