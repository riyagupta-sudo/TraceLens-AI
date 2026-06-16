import cv2
import numpy as np
import os

test_files = {
    "REAL (birds.jpg)": r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\case_intel_leak\birds.jpg",
    "REAL (test dataset 0000.jpg)": r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\ai_detection\test\REAL\0000.jpg",
    "FAKE (test dataset 0.jpg)": r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\ai_detection\test\FAKE\0.jpg",
}

for name, filepath in test_files.items():
    if not os.path.exists(filepath):
        print(f"{name} file not found at {filepath}")
        continue
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        resized = cv2.resize(img, (256, 256))
        dft = np.fft.fft2(resized)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1e-8)
        
        center = 128
        r_min, r_max = 64, 120
        y, x = np.ogrid[-center:256-center, -center:256-center]
        mask = (x**2 + y**2 >= r_min**2) & (x**2 + y**2 <= r_max**2)
        
        outer_ring = magnitude_spectrum[mask]
        mean_val = np.mean(outer_ring)
        max_val = np.max(outer_ring)
        std_val = np.std(outer_ring)
        
        peak_threshold = mean_val + 3.5 * std_val
        peaks = outer_ring[outer_ring > peak_threshold]
        
        print(f"{name}:")
        print(f"  Outer ring size: {len(outer_ring)}")
        print(f"  Mean: {mean_val:.4f} | Std: {std_val:.4f} | Max: {max_val:.4f}")
        print(f"  Threshold (mean + 3.5*std): {peak_threshold:.4f}")
        print(f"  Peak count: {len(peaks)}")
        print(f"  Heuristic trigger (peaks > 15): {len(peaks) > 15}")
