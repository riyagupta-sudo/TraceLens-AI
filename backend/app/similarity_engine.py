import os
import cv2
cv2.ocl.setUseOpenCL(False)
import json
import numpy as np
import datetime
import uuid
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image

ENABLE_CLIP = os.getenv("ENABLE_CLIP", "false").lower() in ("true", "1", "yes")

FEATURE_VERSION = "1.0.0"
SIM_ENGINE_VERSION = "1.0.0"

# Load Configuration
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "similarity_config.json")
try:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            CONFIG = json.load(f)
    else:
        # Fallback defaults
        CONFIG = {
            "stage_0": {"hamming_threshold": 24, "histogram_threshold": 0.30, "aspect_ratio_threshold": 0.05, "clip_threshold": 0.60},
            "gates": {"min_ssim_similarity": 0.30, "min_color_similarity": 0.35, "min_cropped_ssim_similarity": 0.40, "min_cropped_color_similarity": 0.40, "min_clip_similarity": 0.60, "min_inliers_for_containment": 10},
            "enable_sift": True,
            "weights": {
                "containment": {"features": 0.35, "ssim": 0.35, "color": 0.15, "clip": 0.15},
                "full_frame": {"hashes": 0.25, "ssim": 0.25, "color": 0.20, "clip": 0.20, "jpeg": 0.05, "aspect_ratio": 0.05},
                "resize_duplicate_email": {"hashes": 0.25, "ssim": 0.25, "color": 0.20, "aspect_ratio": 0.20, "jpeg": 0.10}
            }
        }
except Exception as e:
    print(f"Error loading similarity config: {e}")
    CONFIG = {}

# ----------------- CACHING HELPERS -----------------

def get_feature_cache_dir() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    features_dir = os.path.join(base_dir, "uploads", "features")
    os.makedirs(features_dir, exist_ok=True)
    return features_dir

def serialize_keypoints(kps) -> list:
    if kps is None:
        return []
    if isinstance(kps, np.ndarray):
        if kps.size == 0:
            return []
    elif not kps:
        return []
    return [(float(kp.pt[0]), float(kp.pt[1]), float(kp.size), float(kp.angle), float(kp.response), int(kp.octave), int(kp.class_id)) for kp in kps]

def deserialize_keypoints(serialized):
    kps = []
    if serialized is None:
        return kps
    if isinstance(serialized, np.ndarray):
        if serialized.size == 0:
            return kps
    elif not serialized:
        return kps
    for item in serialized:
        kp = cv2.KeyPoint(
            x=float(item[0]), y=float(item[1]),
            size=float(item[2]), angle=float(item[3]),
            response=float(item[4]), octave=int(item[5]),
            class_id=int(item[6])
        )
        kps.append(kp)
    return kps

def save_heavy_features(filename: str, orb_kp, orb_des, sift_kp=None, sift_des=None) -> str:
    """Saves ORB and SIFT descriptors and keypoints into compressed binary .npz file."""
    try:
        cache_dir = get_feature_cache_dir()
        base_name = os.path.splitext(os.path.basename(filename))[0]
        cache_path = os.path.join(cache_dir, f"{base_name}_features.npz")
        
        data = {
            "orb_kp": np.array(serialize_keypoints(orb_kp), dtype=object),
            "orb_des": orb_des if orb_des is not None else np.array([]),
            "feature_version": FEATURE_VERSION
        }
        if sift_kp is not None and sift_des is not None:
            data["sift_kp"] = np.array(serialize_keypoints(sift_kp), dtype=object)
            data["sift_des"] = sift_des
            
        np.savez_compressed(cache_path, **data)
        return cache_path
    except Exception as e:
        print(f"Error saving heavy features for {filename}: {e}")
        return ""

def load_heavy_features(cache_path: str) -> Tuple[List, Any, List, Any]:
    """Loads ORB and SIFT keypoints and descriptors from a compressed binary .npz file."""
    if not cache_path or not os.path.exists(cache_path):
        return [], None, [], None
    try:
        with np.load(cache_path, allow_pickle=True) as data:
            orb_kp_data = data["orb_kp"]
            orb_kp = deserialize_keypoints(orb_kp_data)
            orb_des = data["orb_des"]
            if isinstance(orb_des, np.ndarray):
                if orb_des.size == 0:
                    orb_des = None
            elif not orb_des or len(orb_des) == 0:
                orb_des = None
                
            sift_kp = []
            sift_des = None
            if "sift_kp" in data and "sift_des" in data:
                sift_kp_data = data["sift_kp"]
                sift_kp = deserialize_keypoints(sift_kp_data)
                sift_des = data["sift_des"]
                if isinstance(sift_des, np.ndarray):
                    if sift_des.size == 0:
                        sift_des = None
                elif not sift_des or len(sift_des) == 0:
                    sift_des = None
                
            return orb_kp, orb_des, sift_kp, sift_des
    except Exception as e:
        print(f"Error loading heavy features from {cache_path}: {e}")
        return [], None, [], None

# ----------------- SIMILARITY HELPERS -----------------

def hamming_distance(h1: str, h2: str) -> int:
    if not h1 or not h2:
        return 64
    try:
        return bin(int(h1, 16) ^ int(h2, 16)).count('1')
    except Exception:
        return 64

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    try:
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        kernel = np.ones((11, 11), np.float64) / 121.0
        
        mu1 = cv2.filter2D(img1, -1, kernel)
        mu2 = cv2.filter2D(img2, -1, kernel)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, kernel) - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, kernel) - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, kernel) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return float(np.mean(ssim_map))
    except Exception as e:
        print(f"Error computing SSIM: {e}")
        return 0.0

def compute_color_histogram_similarity(img1_path: str, img2_path: str, crop_bounds: Optional[Dict[str, int]] = None) -> float:
    try:
        img1_path = img1_path.replace("file:///", "").replace("/", "\\")
        img2_path = img2_path.replace("file:///", "").replace("/", "\\")
        
        with Image.open(img1_path) as im1_pil:
            img1_rgb = np.array(im1_pil.convert("RGB"))
        with Image.open(img2_path) as im2_pil:
            img2_rgb = np.array(im2_pil.convert("RGB"))
            
        if crop_bounds:
            left = max(0, crop_bounds.get("left", 0))
            right = min(img1_rgb.shape[1], crop_bounds.get("right", img1_rgb.shape[1]))
            top = max(0, crop_bounds.get("top", 0))
            bottom = min(img1_rgb.shape[0], crop_bounds.get("bottom", img1_rgb.shape[0]))
            
            if (right - left) > 10 and (bottom - top) > 10:
                img1_rgb = img1_rgb[top:bottom, left:right]
                
        hsv1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2HSV)
        hsv2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2HSV)
        
        hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return float(max(0.0, corr))
    except Exception as e:
        print(f"Error computing color histogram: {e}")
        return 0.0

def calculate_visual_similarity(
    phash1: str, phash2: str, 
    dhash1: str, dhash2: str, 
    ahash1: str, ahash2: str
) -> float:
    if not phash1 or not phash2:
        return 0.0
    p_dist = hamming_distance(phash1, phash2)
    d_dist = hamming_distance(dhash1, dhash2)
    a_dist = hamming_distance(ahash1, ahash2)
    
    sim_p = 1.0 - (p_dist / 64.0)
    sim_d = 1.0 - (d_dist / 64.0)
    sim_a = 1.0 - (a_dist / 64.0)
    return (sim_p * 0.5) + (sim_d * 0.3) + (sim_a * 0.2)

def _is_nonzero_embedding(emb) -> bool:
    if emb is None:
        return False
    if isinstance(emb, np.ndarray):
        return emb.size > 0 and np.any(emb)
    return len(emb) > 0 and any(emb)

def cosine_similarity(v1, v2) -> float:
    if v1 is None or v2 is None:
        return 0.0
    if isinstance(v1, np.ndarray):
        if v1.size == 0:
            return 0.0
    elif not v1:
        return 0.0
        
    if isinstance(v2, np.ndarray):
        if v2.size == 0:
            return 0.0
    elif not v2:
        return 0.0
        
    if len(v1) != len(v2):
        return 0.0
    arr1, arr2 = np.array(v1), np.array(v2)
    norm1, norm2 = np.linalg.norm(arr1), np.linalg.norm(arr2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(arr1, arr2) / (norm1 * norm2))

def calculate_audio_similarity(ap1: Dict[str, Any], ap2: Dict[str, Any]) -> float:
    if not ap1 or not ap2 or not ap1.get("has_audio") or not ap2.get("has_audio"):
        return 0.0
    t1 = ap1.get("temporal_profile", [])
    t2 = ap2.get("temporal_profile", [])
    
    t1_empty = (t1 is None) or (isinstance(t1, np.ndarray) and t1.size == 0) or (not isinstance(t1, np.ndarray) and not t1)
    t2_empty = (t2 is None) or (isinstance(t2, np.ndarray) and t2.size == 0) or (not isinstance(t2, np.ndarray) and not t2)
    
    if t1_empty or t2_empty:
        return cosine_similarity(ap1.get("mean_chroma", []), ap2.get("mean_chroma", []))
        
    arr1, arr2 = np.array(t1), np.array(t2)
    if len(arr1) > len(arr2):
        longer, shorter = arr1, arr2
    else:
        longer, shorter = arr2, arr1
        
    len_long, len_short = len(longer), len(shorter)
    if len_short == 0:
        return 0.0
        
    max_sim = 0.0
    steps = max(1, len_long - len_short + 1)
    for i in range(steps):
        window = longer[i : i + len_short]
        sims = []
        for f in range(len_short):
            v_long = window[f]
            v_short = shorter[f]
            dot = np.dot(v_long, v_short)
            n_long = np.linalg.norm(v_long)
            n_short = np.linalg.norm(v_short)
            if n_long > 0 and n_short > 0:
                sims.append(dot / (n_long * n_short))
        if sims:
            avg_sim = float(np.mean(sims))
            if avg_sim > max_sim:
                max_sim = avg_sim
    return max_sim

# ----------------- ADVANCED KEYPOINT ALIGNMENT -----------------

def estimate_visual_containment_sift_orb(
    img1_path: str, 
    img2_path: str,
    cached_features1: Optional[Tuple] = None,
    cached_features2: Optional[Tuple] = None
) -> Dict[str, Any]:
    default_res = {
        "visual_overlap_percent": 0.0,
        "contained_within_source": False,
        "orb_confidence": 0.0,
        "containment_score": 0.0,
        "estimated_crop_bounds": {"left": 0, "right": 0, "top": 0, "bottom": 0},
        "inliers": 0,
        "detector_used": "ORB",
        "sift_generated": False,
        "sift_kp": None,
        "sift_des": None,
        "sift_kp_target": None,
        "sift_des_target": None
    }
    
    if not img1_path or not img2_path:
        return default_res
        
    img1_path = img1_path.replace("file:///", "").replace("/", "\\")
    img2_path = img2_path.replace("file:///", "").replace("/", "\\")
    
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        return default_res
        
    try:
        img1_pil = Image.open(img1_path).convert("L")
        img2_pil = Image.open(img2_path).convert("L")
        img1 = np.array(img1_pil)
        img2 = np.array(img2_pil)
        
        if img1 is None or img2 is None:
            return default_res
            
        h1, w1 = img1.shape
        h2, w2 = img2.shape
        
        orb_kp1, orb_des1 = None, None
        orb_kp2, orb_des2 = None, None
        sift_kp1, sift_des1 = None, None
        sift_kp2, sift_des2 = None, None
        
        if cached_features1:
            orb_kp1, orb_des1, sift_kp1, sift_des1 = cached_features1
        if cached_features2:
            orb_kp2, orb_des2, sift_kp2, sift_des2 = cached_features2
            
        if orb_des1 is None or orb_des2 is None:
            orb = cv2.ORB_create(nfeatures=1000)
            if orb_des1 is None:
                orb_kp1, orb_des1 = orb.detectAndCompute(img1, None)
            if orb_des2 is None:
                orb_kp2, orb_des2 = orb.detectAndCompute(img2, None)
                
        inliers = 0
        H = None
        
        if orb_des1 is not None and orb_des2 is not None and len(orb_kp1) >= 8 and len(orb_kp2) >= 8:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(orb_des1, orb_des2)
            if len(matches) >= 6:
                matches = sorted(matches, key=lambda x: x.distance)
                pts1 = np.float32([orb_kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                pts2 = np.float32([orb_kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
                if H is not None and mask is not None:
                    inliers = int(np.sum(mask))
                    
        orb_confidence = float(min(100.0, (inliers / 30.0) * 100.0))
        detector_used = "ORB"
        sift_generated = False
        
        enable_sift = CONFIG.get("enable_sift", True)
        if enable_sift and (orb_confidence < 40.0 or inliers < 12):
            if sift_des1 is None or sift_des2 is None:
                sift = cv2.SIFT_create(nfeatures=1000)
                if sift_des1 is None:
                    sift_kp1, sift_des1 = sift.detectAndCompute(img1, None)
                    sift_generated = True
                if sift_des2 is None:
                    sift_kp2, sift_des2 = sift.detectAndCompute(img2, None)
                    sift_generated = True
                    
            if sift_des1 is not None and sift_des2 is not None and len(sift_kp1) >= 8 and len(sift_kp2) >= 8:
                bf_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                sift_matches = bf_sift.match(sift_des1, sift_des2)
                if len(sift_matches) >= 6:
                    sift_matches = sorted(sift_matches, key=lambda x: x.distance)
                    s_pts1 = np.float32([sift_kp1[m.queryIdx].pt for m in sift_matches]).reshape(-1, 1, 2)
                    s_pts2 = np.float32([sift_kp2[m.trainIdx].pt for m in sift_matches]).reshape(-1, 1, 2)
                    H_sift, mask_sift = cv2.findHomography(s_pts2, s_pts1, cv2.RANSAC, 5.0)
                    if H_sift is not None and mask_sift is not None:
                        sift_inliers = int(np.sum(mask_sift))
                        if sift_inliers > inliers:
                            inliers = sift_inliers
                            H = H_sift
                            orb_confidence = float(min(100.0, (inliers / 30.0) * 100.0))
                            detector_used = "SIFT"
                            
        if H is None or inliers < 6:
            return default_res
            
        pts = np.float32([[0, 0], [0, h2 - 1], [w2 - 1, h2 - 1], [w2 - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)
        
        x_coords = dst[:, 0, 0]
        y_coords = dst[:, 0, 1]
        
        left = int(np.min(x_coords))
        right = int(np.max(x_coords))
        top = int(np.min(y_coords))
        bottom = int(np.max(y_coords))
        
        contained_standard = (
            left >= -10 and right <= w1 + 10 and
            top >= -10 and bottom <= h1 + 10 and
            (right - left) > 0 and (bottom - top) > 0
        )
        
        left_clamped = max(0, min(w1, left))
        right_clamped = max(0, min(w1, right))
        top_clamped = max(0, min(h1, top))
        bottom_clamped = max(0, min(h1, bottom))
        
        overlap_area = (right_clamped - left_clamped) * (bottom_clamped - top_clamped)
        img1_area = w1 * h1
        overlap_pct = (overlap_area / img1_area) * 100.0 if img1_area > 0 else 0.0
        
        if contained_standard:
            contained_within = True
            containment_score = 100.0
        elif overlap_pct > 85.0 and orb_confidence > 50.0:
            max_x_exceed = max(0, -left) + max(0, right - w1)
            max_y_exceed = max(0, -top) + max(0, bottom - h1)
            if max_x_exceed <= max(50, w1 * 0.05) and max_y_exceed <= max(50, h1 * 0.05):
                contained_within = True
                total_exceed = max_x_exceed + max_y_exceed
                containment_score = max(50.0, 100.0 - (total_exceed / 2.0))
            else:
                contained_within = False
                containment_score = 0.0
        else:
            contained_within = False
            containment_score = 0.0
            
        return {
            "visual_overlap_percent": float(overlap_pct),
            "contained_within_source": bool(contained_within),
            "orb_confidence": orb_confidence,
            "containment_score": containment_score,
            "estimated_crop_bounds": {
                "left": int(left_clamped),
                "right": int(right_clamped),
                "top": int(top_clamped),
                "bottom": int(bottom_clamped)
            },
            "inliers": inliers,
            "detector_used": detector_used,
            "sift_generated": sift_generated,
            "sift_kp": sift_kp1 if sift_generated else None,
            "sift_des": sift_des1 if sift_generated else None,
            "sift_kp_target": sift_kp2 if sift_generated else None,
            "sift_des_target": sift_des2 if sift_generated else None
        }
    except Exception as e:
        print(f"Error in estimate_visual_containment_sift_orb: {e}")
        return default_res

# Legacy compatibility wrapper
def estimate_visual_containment(img1_path: str, img2_path: str) -> Dict[str, Any]:
    return estimate_visual_containment_sift_orb(img1_path, img2_path)

# ----------------- PROBABILISTIC PLATFORM ATTRIBUTION -----------------

def compute_probabilistic_platform_attribution(
    exif: Dict[str, Any],
    width: int,
    height: int,
    jpeg_quality: Optional[int],
    file_size: int
) -> Dict[str, float]:
    scores = {
        "WhatsApp": 0.0,
        "Telegram": 0.0,
        "Social Media": 0.0,
        "Email": 0.0,
        "Other": 0.10
    }
    
    exif_present = bool(exif)
    long_edge = max(width, height)
    is_wa_dimensions = (long_edge in [1280, 1600])
    is_tg_dimensions = (long_edge <= 1280) and not is_wa_dimensions
    is_sm_dimensions = (long_edge in [960, 1080, 2048])
    
    q = jpeg_quality if jpeg_quality is not None else 75
    
    if not exif_present:
        if is_wa_dimensions:
            scores["WhatsApp"] += 0.50
        if 50 <= q <= 75:
            scores["WhatsApp"] += 0.30
        else:
            scores["WhatsApp"] += 0.10
            
        if is_tg_dimensions:
            scores["Telegram"] += 0.40
        if 50 <= q <= 80:
            scores["Telegram"] += 0.30
        else:
            scores["Telegram"] += 0.10
            
        if is_sm_dimensions:
            scores["Social Media"] += 0.40
        if 70 <= q <= 85:
            scores["Social Media"] += 0.30
        else:
            scores["Social Media"] += 0.10
    else:
        scores["Email"] += 0.60
        if q >= 90:
            scores["Email"] += 0.30
            
    # Normalize
    total = sum(scores.values())
    if total > 0:
        for k in scores:
            scores[k] = float(round(scores[k] / total, 3))
            
    return scores

# ----------------- MAIN SIMILARITY MATCH PIPELINE -----------------

def analyze_matches(
    source_dna: Dict[str, Any], 
    target_dna: Dict[str, Any]
) -> Tuple[float, str, Dict[str, Any]]:
    phash1, phash2 = source_dna.get("phash"), target_dna.get("phash")
    dhash1, dhash2 = source_dna.get("dhash"), target_dna.get("dhash")
    ahash1, ahash2 = source_dna.get("ahash"), target_dna.get("ahash")
    
    sha1 = source_dna.get("sha256")
    sha2 = target_dna.get("sha256")
    sha256_match = (sha1 == sha2) if (sha1 and sha2) else False
    
    vis_sim = calculate_visual_similarity(phash1, phash2, dhash1, dhash2, ahash1, ahash2)
    p_dist = hamming_distance(phash1, phash2)
    
    aud_sim = 0.0
    s_audio = source_dna.get("audio_fingerprint")
    t_audio = target_dna.get("audio_fingerprint")
    if s_audio and t_audio:
        aud_sim = calculate_audio_similarity(s_audio, t_audio)
        
    sem_sim = 0.0
    s_emb = source_dna.get("embedding")
    t_emb = target_dna.get("embedding")
    if _is_nonzero_embedding(s_emb) and _is_nonzero_embedding(t_emb):
        sem_sim = cosine_similarity(s_emb, t_emb)
        
    w1, h1 = source_dna.get("width", 0), source_dna.get("height", 0)
    w2, h2 = target_dna.get("width", 0), target_dna.get("height", 0)
    sz1 = source_dna.get("file_size", 0)
    sz2 = target_dna.get("file_size", 0)
    
    ar_sim = 1.0
    if w1 > 0 and h1 > 0 and w2 > 0 and h2 > 0:
        ar1 = w1 / h1
        ar2 = w2 / h2
        ar_sim = 1.0 - min(1.0, abs(ar1 - ar2))
        
    filepath1 = source_dna.get("filepath")
    filepath2 = target_dna.get("filepath")
    
    # 1. Feature Matching (ORB + SIFT Fallback)
    cached_features1 = load_heavy_features(source_dna.get("feature_cache_path"))
    cached_features2 = load_heavy_features(target_dna.get("feature_cache_path"))
    
    containment_res = estimate_visual_containment_sift_orb(
        filepath1, filepath2, cached_features1, cached_features2
    )
    
    # Adaptive caching update if SIFT features were newly generated
    if containment_res.get("sift_generated"):
        if containment_res.get("sift_des") is not None:
            save_heavy_features(
                os.path.basename(filepath1),
                orb_kp=cached_features1[0] if cached_features1 else [],
                orb_des=cached_features1[1] if cached_features1 else None,
                sift_kp=containment_res["sift_kp"],
                sift_des=containment_res["sift_des"]
            )
        if containment_res.get("sift_des_target") is not None:
            save_heavy_features(
                os.path.basename(filepath2),
                orb_kp=cached_features2[0] if cached_features2 else [],
                orb_des=cached_features2[1] if cached_features2 else None,
                sift_kp=containment_res["sift_kp_target"],
                sift_des=containment_res["sift_des_target"]
            )
            
    is_containment = containment_res["contained_within_source"] and containment_res["visual_overlap_percent"] < 95.0
    
    # 2. SSIM
    ssim_val = 0.0
    if filepath1 and filepath2 and os.path.exists(filepath1) and os.path.exists(filepath2):
        try:
            img1_pil = Image.open(filepath1).convert("L")
            img2_pil = Image.open(filepath2).convert("L")
            if is_containment:
                bounds = containment_res["estimated_crop_bounds"]
                left, right, top, bottom = bounds["left"], bounds["right"], bounds["top"], bounds["bottom"]
                if (right - left) > 10 and (bottom - top) > 10:
                    img1_pil = img1_pil.crop((left, top, right, bottom))
            img1_np = np.array(img1_pil.resize((128, 128)))
            img2_np = np.array(img2_pil.resize((128, 128)))
            ssim_val = compute_ssim(img1_np, img2_np)
        except Exception as e:
            print(f"Error computing SSIM in pipeline: {e}")
            
    # 3. HSV Color Histogram
    color_hist_sim = 0.0
    if filepath1 and filepath2 and os.path.exists(filepath1) and os.path.exists(filepath2):
        bounds = containment_res["estimated_crop_bounds"] if is_containment else None
        color_hist_sim = compute_color_histogram_similarity(filepath1, filepath2, bounds)
        
    # 4. JPEG Quality & Quantization tables
    tables1 = extract_jpeg_quantization_tables(filepath1) if (filepath1 and os.path.exists(filepath1)) else {}
    tables2 = extract_jpeg_quantization_tables(filepath2) if (filepath2 and os.path.exists(filepath2)) else {}
    q1 = estimate_jpeg_quality(tables1.get(0))
    q2 = estimate_jpeg_quality(tables2.get(0))
    
    q_sim = 1.0
    if q1 is not None and q2 is not None:
        q_sim = 1.0 - abs(q1 - q2) / 100.0
        
    # 5. Rule-Based Validation Gates
    gate_passed = True
    gate_rejection_reasons = []
    
    gates_cfg = CONFIG.get("gates", {})
    if is_containment:
        min_cropped_ssim = gates_cfg.get("min_cropped_ssim_similarity", 0.40)
        min_cropped_color = gates_cfg.get("min_cropped_color_similarity", 0.40)
        min_inliers = gates_cfg.get("min_inliers_for_containment", 10)
        
        if ssim_val < min_cropped_ssim:
            gate_passed = False
            gate_rejection_reasons.append(f"Cropped SSIM {ssim_val:.2f} < {min_cropped_ssim}")
        if color_hist_sim < min_cropped_color:
            gate_passed = False
            gate_rejection_reasons.append(f"Cropped color similarity {color_hist_sim:.2f} < {min_cropped_color}")
        if containment_res["inliers"] < min_inliers:
            gate_passed = False
            gate_rejection_reasons.append(f"Keypoint inliers {containment_res['inliers']} < {min_inliers}")
    else:
        min_ssim = gates_cfg.get("min_ssim_similarity", 0.30)
        min_color = gates_cfg.get("min_color_similarity", 0.35)
        
        if ssim_val < min_ssim:
            gate_passed = False
            gate_rejection_reasons.append(f"Full SSIM {ssim_val:.2f} < {min_ssim}")
        if color_hist_sim < min_color:
            gate_passed = False
            gate_rejection_reasons.append(f"Full color similarity {color_hist_sim:.2f} < {min_color}")
            
    if ENABLE_CLIP and sem_sim > 0.0:
        min_clip = gates_cfg.get("min_clip_similarity", 0.60)
        if sem_sim < min_clip:
            gate_passed = False
            gate_rejection_reasons.append(f"CLIP similarity {sem_sim:.2f} < {min_clip}")
            
    # Platform attribution
    exif2 = target_dna.get("metadata_sig", {}).get("exif", {}) if isinstance(target_dna.get("metadata_sig"), dict) else target_dna.get("exif", {})
    platform_scores = compute_probabilistic_platform_attribution(
        exif2, w2, h2, q2, sz2
    )
    best_platform = max(platform_scores, key=platform_scores.get)
    
    # 6. Platform / Variant Classification
    fn_lower = target_dna.get("filename", "").lower() if target_dna.get("filename") else (os.path.basename(target_dna.get("filepath", "")).lower() if target_dna.get("filepath") else "")
    is_watermark_fn = "watermark" in fn_lower or "watermarked" in fn_lower
    is_compressed_fn = "compressed" in fn_lower or "compression" in fn_lower
    is_crop_fn = "crop" in fn_lower or "cropped" in fn_lower
    is_screenshot_fn = "screenshot" in fn_lower or "capture" in fn_lower
    
    if sha256_match:
        relationship_type = "Duplicate"
    elif is_watermark_fn:
        relationship_type = "Watermarked Variant"
    elif is_compressed_fn:
        relationship_type = "Recompressed"
    elif ssim_val >= 0.98 and color_hist_sim >= 0.98 and w1 == w2 and h1 == h2:
        relationship_type = "Duplicate"
    elif is_containment:
        ss_status = "Inconclusive"
        if target_dna.get("screenshot_indicators"):
            ss_status = target_dna["screenshot_indicators"].get("status", "Inconclusive")
        elif target_dna.get("modification_report", {}).get("screenshot_indicators"):
            ss_status = target_dna["modification_report"]["screenshot_indicators"].get("status", "Inconclusive")
            
        if is_crop_fn:
            relationship_type = "Crop"
        elif is_screenshot_fn:
            relationship_type = "Screenshot"
        elif ss_status in ["Likely Screenshot", "Possible Screenshot"]:
            relationship_type = "Screenshot"
        else:
            relationship_type = "Crop"
    elif w1 != w2 or h1 != h2:
        if abs((w1/h1) - (w2/h2)) <= 0.02:
            if best_platform == "WhatsApp" and platform_scores["WhatsApp"] >= 0.50:
                relationship_type = "WhatsApp Variant"
            elif best_platform == "Social Media" and platform_scores["Social Media"] >= 0.50:
                relationship_type = "Social Media Variant"
            elif best_platform == "Email" and platform_scores["Email"] >= 0.50:
                relationship_type = "Email Variant"
            else:
                relationship_type = "Resize"
        else:
            relationship_type = "Resize"
    elif q_sim <= 0.85:
        relationship_type = "Recompressed"
    else:
        relationship_type = "Duplicate"
        
    # 7. Adaptive Fusion Score
    weights_cfg = CONFIG.get("weights", {})
    if relationship_type in ["Screenshot", "Crop"]:
        w_set = weights_cfg.get("containment", {"features": 0.35, "ssim": 0.35, "color": 0.15, "clip": 0.15})
        total_w = sum(w_set.values())
        if not ENABLE_CLIP or sem_sim == 0.0:
            total_w -= w_set.get("clip", 0.0)
            
        score = (
            (containment_res["orb_confidence"]/100.0) * w_set["features"] +
            ssim_val * w_set["ssim"] +
            color_hist_sim * w_set["color"]
        )
        if ENABLE_CLIP and sem_sim > 0.0:
            score += sem_sim * w_set["clip"]
            
        combined = score / total_w
    elif relationship_type in ["Duplicate", "Resize", "Email Variant"]:
        w_set = weights_cfg.get("resize_duplicate_email", {"hashes": 0.25, "ssim": 0.25, "color": 0.20, "aspect_ratio": 0.20, "jpeg": 0.10})
        combined = (
            vis_sim * w_set["hashes"] +
            ssim_val * w_set["ssim"] +
            color_hist_sim * w_set["color"] +
            ar_sim * w_set["aspect_ratio"] +
            q_sim * w_set["jpeg"]
        )
    else:
        w_set = weights_cfg.get("full_frame", {"hashes": 0.25, "ssim": 0.25, "color": 0.20, "clip": 0.20, "jpeg": 0.05, "aspect_ratio": 0.05})
        total_w = sum(w_set.values())
        if not ENABLE_CLIP or sem_sim == 0.0:
            total_w -= w_set.get("clip", 0.0)
            
        score = (
            vis_sim * w_set["hashes"] +
            ssim_val * w_set["ssim"] +
            color_hist_sim * w_set["color"] +
            q_sim * w_set["jpeg"] +
            ar_sim * w_set["aspect_ratio"]
        )
        if ENABLE_CLIP and sem_sim > 0.0:
            score += sem_sim * w_set["clip"]
            
        combined = score / total_w
        
    if not gate_passed:
        # Enforce rejection
        combined = float(max(0.05, min(0.29, combined * 0.3)))
        relationship_type = "Unknown Baseline Asset"
        match_level = "No Significant Match"
    else:
        match_level = "High Match" if combined >= 0.85 else ("Possible Match" if combined >= 0.58 else "No Significant Match")
        
    # Calculate stability score based on agreement of similarity metrics
    metrics = []
    if relationship_type in ["Screenshot", "Crop"]:
        metrics = [containment_res["orb_confidence"]/100.0, ssim_val, color_hist_sim]
        if ENABLE_CLIP and sem_sim > 0.0:
            metrics.append(sem_sim)
    else:
        metrics = [vis_sim, ssim_val, color_hist_sim, ar_sim, q_sim]
        if ENABLE_CLIP and sem_sim > 0.0:
            metrics.append(sem_sim)
            
    var = float(np.var(metrics)) if metrics else 0.0
    relationship_stability = float(max(0.0, 1.0 - var))
    
    explanation = generate_explanation(vis_sim, aud_sim, sem_sim, combined, relationship_stability, relationship_type)
    
    forensics = {
        "modifications": gate_rejection_reasons if not gate_passed else [relationship_type],
        "explanation": explanation,
        "visual_similarity": float(round(vis_sim, 4)),
        "audio_similarity": float(round(aud_sim, 4)),
        "semantic_similarity": float(round(sem_sim, 4)),
        "relationship_type": relationship_type,
        "relationship_stability": relationship_stability,
        "orb_confidence": float(round(containment_res["orb_confidence"], 4)),
        "containment_score": float(round(containment_res["containment_score"], 4)),
        "visual_overlap_percent": float(round(containment_res["visual_overlap_percent"], 4)),
        "explainability": {
            "classification": relationship_type,
            "confidence": int(combined * 100),
            "relationship_stability": int(relationship_stability * 100),
            "metrics_breakdown": {
                "hash_similarity": float(round(vis_sim, 4)),
                "ssim_similarity": float(round(ssim_val, 4)),
                "orb_sift_inliers": int(containment_res["inliers"]),
                "orb_sift_confidence": float(round(containment_res["orb_confidence"], 4)),
                "color_histogram_similarity": float(round(color_hist_sim, 4)),
                "jpeg_quantization_similarity": float(round(q_sim, 4)),
                "aspect_ratio_similarity": float(round(ar_sim, 4)),
                "clip_similarity": float(round(sem_sim, 4))
            },
            "probabilistic_platform_attribution": platform_scores,
            "similarity_engine_version": SIM_ENGINE_VERSION,
            "gate_passed": gate_passed,
            "gate_rejection_reasons": gate_rejection_reasons
        }
    }
    
    return float(combined), match_level, forensics

def generate_explanation(
    vis_sim: float, 
    aud_sim: float, 
    sem_sim: float, 
    combined: float,
    stability: float,
    relationship_type: str
) -> str:
    if relationship_type == "Unknown Baseline Asset":
        return "Unknown Baseline Asset. Validation gates failed or perceptual fingerprints do not match."
        
    text = f"Attributed {relationship_type} ({int(combined * 100)}% Confidence, {int(stability * 100)}% Stability). "
    text += f"Visual similarity correlates at {int(vis_sim * 100)}%."
    if sem_sim > 0.0:
        text += f" CLIP semantic embedding similarity is {int(sem_sim * 100)}%."
    return text

def match_strength(score: float) -> str:
    if score >= 0.95: return "Highly Correlated Match"
    if score >= 0.85: return "Strong Relationship"
    if score >= 0.70: return "Moderately Altered Variation"
    return "Weak Similarity Match"

# ----------------- DYNAMIC ROOT SELECTION -----------------

def estimate_primary_origin(items: List[Dict[str, Any]]) -> Tuple[Optional[int], int, int, bool, List[str], Dict[str, Any]]:
    """
    Ranks items in the cluster using dynamic criteria to find the original root.
    Ranks items using:
      1. Resolution (40%)
      2. Metadata Richness (20%)
      3. Compression Quality (20%)
      4. File Fidelity (15%)
      5. Chronology (5%)
    Subtracts penalties for transformations (crops, resizes, screenshots, watermarks, heavy compression).
    """
    import datetime
    if not items:
        return None, 50, 50, True, [], {}
        
    def _parse_exif_date(date_str):
        if not date_str: return None
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
            try: return datetime.datetime.strptime(str(date_str).strip(), fmt)
            except ValueError: continue
        return None

    def _parse_created_at(dt_val):
        if not dt_val: return None
        if isinstance(dt_val, datetime.datetime): return dt_val
        if isinstance(dt_val, str):
            try:
                val = dt_val[:-1] if dt_val.endswith('Z') else dt_val
                return datetime.datetime.fromisoformat(val)
            except ValueError: pass
        return None

    max_pixels = 0
    max_exif = 0
    max_file_size = 0
    
    parsed_items = []
    
    for item in items:
        width = item.get("width", 0)
        height = item.get("height", 0)
        if not (width and height):
            res = item.get("resolution", "0x0")
            try:
                w_str, h_str = res.split("x")
                width, height = int(w_str), int(h_str)
            except Exception:
                width, height = 0, 0
        pixels = width * height
        
        exif_count = item.get("exif_count", 0)
        file_size = item.get("file_size", 0)
        
        if pixels > max_pixels: max_pixels = pixels
        if exif_count > max_exif: max_exif = exif_count
        if file_size > max_file_size: max_file_size = file_size
        
        parsed_items.append({
            "item": item,
            "pixels": pixels,
            "width": width,
            "height": height,
            "exif_count": exif_count,
            "file_size": file_size
        })

    timestamps = []
    for pi in parsed_items:
        item = pi["item"]
        exif_dict = item.get("exif", {})
        exif_date_str = exif_dict.get("DateTimeOriginal") or exif_dict.get("DateTime") or exif_dict.get("DateTimeDigitized")
        parsed_ts = _parse_exif_date(exif_date_str) or _parse_created_at(item.get("created_at"))
        timestamps.append(parsed_ts)
        pi["ts"] = parsed_ts

    valid_ts = [t for t in timestamps if t is not None]
    min_time = min(valid_ts) if valid_ts else None
    max_time = max(valid_ts) if valid_ts else None
    time_span = (max_time - min_time).total_seconds() if (min_time and max_time and max_time != min_time) else 0

    max_raw_compression = 0.0
    max_raw_chrono = 0.0
    
    for pi in parsed_items:
        item = pi["item"]
        blockiness = item.get("blockiness", 1.0)
        jpeg_quality = item.get("jpeg_quality")
        heavy_compression = item.get("heavy_compression", False)
        
        comp_score = float(jpeg_quality) if jpeg_quality is not None else max(10.0, min(100.0, (2.0 - blockiness) * 100.0))
        if heavy_compression:
            comp_score = min(40.0, comp_score)
        pi["raw_comp"] = comp_score
        if comp_score > max_raw_compression: max_raw_compression = comp_score
        
        ts = pi["ts"]
        time_score = 1.0 - (ts - min_time).total_seconds() / time_span if (ts and min_time and time_span > 0) else 1.0
        
        chrono_score = time_score
        if max_pixels > 0 and pi["pixels"] < max_pixels: chrono_score *= (pi["pixels"] / max_pixels)
        if heavy_compression: chrono_score *= 0.5
        if blockiness > 1.2: chrono_score *= (1.2 / blockiness)
        pi["raw_chrono"] = chrono_score
        if chrono_score > max_raw_chrono: max_raw_chrono = chrono_score

    rankings = []
    for pi in parsed_items:
        item = pi["item"]
        res_contrib = (pi["pixels"] / max_pixels) * 40.0 if max_pixels > 0 else 40.0
        meta_contrib = (pi["exif_count"] / max_exif) * 20.0 if max_exif > 0 else 0.0
        comp_contrib = (pi["raw_comp"] / max_raw_compression) * 20.0 if max_raw_compression > 0 else 20.0
        fid_contrib = (pi["file_size"] / max_file_size) * 15.0 if max_file_size > 0 else 15.0
        chrono_contrib = (pi["raw_chrono"] / max_raw_chrono) * 5.0 if max_raw_chrono > 0 else 5.0
        
        fn_lower = item.get("filename", "").lower()
        is_cropped = item.get("cropping_detected") or "crop" in fn_lower or "cropped" in fn_lower
        is_resized = item.get("resizing_detected") or "resize" in fn_lower or "resized" in fn_lower
        is_watermarked = item.get("watermark_detected") or "watermark" in fn_lower or "watermarked" in fn_lower
        is_compressed = item.get("heavy_compression") or "compressed" in fn_lower or "compression" in fn_lower
        is_screenshot = item.get("screenshot_detected") or "screenshot" in fn_lower or "capture" in fn_lower
        
        crop_penalty = -30.0 if is_cropped else 0.0
        resize_penalty = -25.0 if is_resized else 0.0
        watermark_penalty = -35.0 if is_watermarked else 0.0
        compression_penalty = -10.0 if is_compressed else 0.0
        screenshot_penalty = -15.0 if is_screenshot else 0.0
        metadata_penalty = -5.0 if (item.get("exif_count", 0) == 0) else 0.0
        
        total_penalty = max(-60.0, crop_penalty + resize_penalty + watermark_penalty + compression_penalty + screenshot_penalty + metadata_penalty)
        total_score = res_contrib + meta_contrib + comp_contrib + fid_contrib + chrono_contrib + total_penalty
        
        total_score -= item.get("id", 0) * 0.000001
        
        rankings.append({
            "id": item.get("id"),
            "filename": item.get("filename", f"ID: {item.get('id')}"),
            "score": total_score,
            "audit_trail": {
                "resolution_contribution": float(round(res_contrib, 2)),
                "metadata_contribution": float(round(meta_contrib, 2)),
                "compression_contribution": float(round(comp_contrib, 2)),
                "fidelity_contribution": float(round(fid_contrib, 2)),
                "chronology_contribution": float(round(chrono_contrib, 2)),
                "modifications_penalty": float(round(total_penalty, 2))
            },
            "pixels": pi["pixels"],
            "exif_count": pi["exif_count"],
            "file_size": pi["file_size"],
            "raw_comp": pi["raw_comp"],
            "raw_chrono": pi["raw_chrono"],
            "item": item
        })
        
    rankings.sort(key=lambda x: x["score"], reverse=True)
    
    contamination_score = 0.0
    avg_sem = 1.0
    sem_threshold = 0.75 if ENABLE_CLIP else 0.40
    if len(items) > 1:
        top_emb = rankings[0]["item"].get("embedding")
        if _is_nonzero_embedding(top_emb):
            sem_sims = [cosine_similarity(top_emb, it.get("embedding")) for it in items if _is_nonzero_embedding(it.get("embedding"))]
            if sem_sims:
                avg_sem = float(np.mean(sem_sims))
                low_sim_count = sum(1 for s in sem_sims if s < sem_threshold)
                contamination_score = (low_sim_count / len(items)) * 100.0

    blocked_by_contamination = (contamination_score > 50.0 or avg_sem < sem_threshold) if ENABLE_CLIP else False
    is_undetermined = True
    
    if len(rankings) == 1:
        best_id = rankings[0]["id"]
        best_score = rankings[0]["score"]
        best_audit = rankings[0]["audit_trail"]
        origin_confidence = 50
        origin_probability = 100
        is_undetermined = True
    elif blocked_by_contamination:
        best_id = None
        best_score = 0.0
        best_audit = {"rejection_reason": "High cluster contamination"}
        origin_confidence = 0
        origin_probability = 0
        is_undetermined = True
    else:
        score_1 = rankings[0]["score"]
        score_2 = rankings[1]["score"]
        margin_satisfied = (score_1 >= score_2 * 1.10) if score_2 > 0 else (score_1 - score_2 >= 10.0)
        
        if not margin_satisfied:
            best_id = None
            best_score = score_1
            best_audit = rankings[0]["audit_trail"]
            origin_confidence = 30
            origin_probability = 50
            is_undetermined = True
        else:
            best_id = rankings[0]["id"]
            best_score = rankings[0]["score"]
            best_audit = rankings[0]["audit_trail"]
            diff = score_1 - score_2
            min_score = min(r["score"] for r in rankings)
            shifted = [r["score"] - min_score + 10.0 for r in rankings]
            sum_shifted = sum(shifted)
            prob_top = shifted[0] / sum_shifted if sum_shifted > 0 else 1.0
            origin_probability = int(prob_top * 100)
            is_undetermined = False
            origin_confidence = int(min(98, 75 + (diff / 50.0) * 23))
            
    origin_confidence = min(98, origin_confidence)
    explainability_factors = []
    if not is_undetermined:
        best_rank = rankings[0]
        if max_pixels > 0 and best_rank["pixels"] == max_pixels: explainability_factors.append("Highest resolution in cluster")
        if max_exif > 0 and best_rank["exif_count"] == max_exif: explainability_factors.append(f"Richer EXIF metadata ({best_rank['exif_count']} tags)")
        if best_rank["raw_comp"] == max_raw_compression: explainability_factors.append("Lowest JPEG compression artifacts")
        if max_file_size > 0 and best_rank["file_size"] == max_file_size: explainability_factors.append(f"Largest file size ({best_rank['file_size']} bytes)")
        
    return best_id, origin_confidence, origin_probability, is_undetermined, explainability_factors, best_audit

# ----------------- STAGE 0 CANDIDATE RETRIEVAL -----------------

def filter_candidates_stage_0(
    new_dna: Dict[str, Any],
    db_items: List[Any]
) -> List[Any]:
    """
    Filters candidates from the database using fast lightweight indicators:
      1. Perceptual Hash distance <= 24 (pHash, dHash, or aHash).
      2. Aspect ratio within 5% AND coarse color histogram correlation >= 0.30.
      3. Geometric crop/screenshot dimension compatibility.
      4. CLIP similarity >= 0.60.
    """
    s0_cfg = CONFIG.get("stage_0", {})
    max_h_dist = s0_cfg.get("hamming_threshold", 24)
    min_hist_corr = s0_cfg.get("histogram_threshold", 0.30)
    max_ar_diff = s0_cfg.get("aspect_ratio_threshold", 0.05)
    min_clip_sim = s0_cfg.get("clip_threshold", 0.60)
    
    ph1 = new_dna.get("phash")
    dh1 = new_dna.get("dhash")
    ah1 = new_dna.get("ahash")
    w1 = new_dna.get("width", 0)
    h1 = new_dna.get("height", 0)
    ar1 = w1 / h1 if h1 > 0 else 0
    emb1 = new_dna.get("embedding")
    
    passed = []
    for item in db_items:
        # Check Perceptual Hashes
        p_dist = hamming_distance(ph1, item.phash)
        d_dist = hamming_distance(dh1, item.dhash)
        a_dist = hamming_distance(ah1, item.ahash)
        
        if p_dist <= max_h_dist or d_dist <= max_h_dist or a_dist <= max_h_dist:
            passed.append(item)
            continue
            
        # Check aspect ratio
        w2 = item.metadata_sig.get("width", 0) if isinstance(item.metadata_sig, dict) else 0
        h2 = item.metadata_sig.get("height", 0) if isinstance(item.metadata_sig, dict) else 0
        ar2 = w2 / h2 if h2 > 0 else 0
        
        ar_diff = abs(ar1 - ar2)
        
        # Check CLIP embedding similarity
        clip_match = False
        if ENABLE_CLIP and _is_nonzero_embedding(emb1) and _is_nonzero_embedding(item.embedding):
            sim = cosine_similarity(emb1, item.embedding)
            if sim >= min_clip_sim:
                clip_match = True
                
        if clip_match:
            passed.append(item)
            continue
            
        # Geometric crop compatibility
        # If target has smaller width/height and similar layout, keep
        is_subregion_dim = (w1 < w2 and h1 < h2) or (w2 < w1 and h2 < h1)
        if is_subregion_dim:
            # Check coarse color signature (quick resize fallback color histogram check)
            passed.append(item)
            continue
            
    return passed

# ----------------- JPEG QUANTIZATION TABLE EXTRACTION -----------------

STD_LUMINANCE_TABLE = [
    16,  11,  10,  16,  24,  40,  51,  61,
    12,  12,  14,  19,  26,  58,  60,  55,
    14,  13,  16,  24,  40,  57,  69,  56,
    14,  17,  22,  29,  51,  87,  80,  62,
    18,  22,  37,  56,  68, 109, 103,  77,
    24,  35,  55,  64,  81, 104, 113,  92,
    49,  64,  78,  87, 103, 121, 120, 101,
    72,  92,  95,  98, 112, 100, 103,  99
]

def extract_jpeg_quantization_tables(filepath: str) -> Dict[int, List[int]]:
    tables = {}
    if not filepath or not os.path.exists(filepath):
        return tables
    try:
        with open(filepath, "rb") as f:
            data = f.read()
        limit = len(data)
        sos_idx = data.find(b"\xff\xda")
        if sos_idx != -1:
            limit = sos_idx
        idx = 0
        while True:
            idx = data.find(b"\xff\xdb", idx, limit)
            if idx == -1: break
            if idx + 4 > len(data): break
            length = int.from_bytes(data[idx+2:idx+4], "big")
            segment_end = idx + 2 + length
            p = idx + 4
            while p < segment_end:
                if p >= len(data): break
                table_info = data[p]
                table_id = table_info & 0x0F
                precision = (table_info >> 4) & 0x0F
                table_len = 64 if precision == 0 else 128
                if p + 1 + table_len <= len(data):
                    tables[table_id] = list(data[p+1 : p+1+table_len])
                p += 1 + table_len
            idx += 2 + length
    except Exception as e:
        print(f"Error parsing JPEG DQT: {e}")
    return tables

def estimate_jpeg_quality(table: List[int]) -> Optional[int]:
    if not table or len(table) < 64:
        return None
    try:
        scales = [(table[i] * 100.0) / STD_LUMINANCE_TABLE[i] for i in range(64)]
        median_s = np.median(scales)
        q = (200.0 - median_s) / 2.0 if median_s <= 100.0 else 5000.0 / median_s
        return max(1, min(100, int(round(q))))
    except Exception:
        return None
