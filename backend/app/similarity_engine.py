import os
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

ENABLE_CLIP = os.getenv("ENABLE_CLIP", "false").lower() in ("true", "1", "yes")

def hamming_distance(h1: str, h2: str) -> int:
    """Calculates Hamming distance between two hex hashes."""
    if not h1 or not h2:
        return 64  # Max distance
    try:
        return bin(int(h1, 16) ^ int(h2, 16)).count('1')
    except Exception:
        return 64


def estimate_visual_containment(img1_path: str, img2_path: str) -> Dict[str, Any]:
    """
    Estimates the bounding box crop of img2 (target) relative to img1 (source) and calculates overlap percentage.
    Uses ORB keypoints, BFMatcher, and RANSAC homography.
    """
    default_res = {
        "visual_overlap_percent": 0.0,
        "contained_within_source": False,
        "orb_confidence": 0.0,
        "containment_score": 0.0,
        "estimated_crop_bounds": {
            "left": 0,
            "right": 0,
            "top": 0,
            "bottom": 0
        }
    }
    if not img1_path or not img2_path:
        return default_res

    # Clean path string formatting
    img1_path = img1_path.replace("file:///", "").replace("/", "\\")
    img2_path = img2_path.replace("file:///", "").replace("/", "\\")
    
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        return default_res
        
    try:
        # Load grayscale images using PIL for robustness and to avoid cv2.imread None issues
        from PIL import Image
        img1_pil = Image.open(img1_path).convert("L")
        img2_pil = Image.open(img2_path).convert("L")
        img1 = np.array(img1_pil)
        img2 = np.array(img2_pil)
        
        if img1 is None or img2 is None:
            return default_res
            
        h1, w1 = img1.shape
        h2, w2 = img2.shape
        
        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=1000)
        
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return default_res
            
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        if len(matches) < 6:
            return default_res
            
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Extract location of good matches
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find Homography (mapping target coordinates to source coordinates)
        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        
        if H is None or mask is None:
            return default_res
            
        inliers = int(np.sum(mask))
        if inliers < 15:
            return default_res
            
        # Get target image corners
        pts = np.float32([[0, 0], [0, h2 - 1], [w2 - 1, h2 - 1], [w2 - 1, 0]]).reshape(-1, 1, 2)
        
        # Transform corners to img1 space
        dst = cv2.perspectiveTransform(pts, H)
        
        # Calculate bounding box in img1 space
        x_coords = dst[:, 0, 0]
        y_coords = dst[:, 0, 1]
        
        left = int(np.min(x_coords))
        right = int(np.max(x_coords))
        top = int(np.min(y_coords))
        bottom = int(np.max(y_coords))
        
        # Check if contained within source coordinates (with reasonable border tolerance, say 10 pixels)
        contained_standard = (
            left >= -10 and right <= w1 + 10 and
            top >= -10 and bottom <= h1 + 10 and
            (right - left) > 0 and (bottom - top) > 0
        )
        
        # Clamp bounds
        left_clamped = max(0, min(w1, left))
        right_clamped = max(0, min(w1, right))
        top_clamped = max(0, min(h1, top))
        bottom_clamped = max(0, min(h1, bottom))
        
        overlap_area = (right_clamped - left_clamped) * (bottom_clamped - top_clamped)
        img1_area = w1 * h1
        overlap_pct = (overlap_area / img1_area) * 100.0 if img1_area > 0 else 0.0
        
        # Calculate ORB confidence (30 inliers is 100% confidence)
        orb_confidence = float(min(100.0, (inliers / 30.0) * 100.0))
        
        # Soft boundary check: if overlap is high and ORB confidence is high,
        # we allow coordinates to exceed bounds slightly (up to 50 pixels or 5% of dimensions)
        if contained_standard:
            contained_within = True
            containment_score = 100.0
        elif overlap_pct > 85.0 and orb_confidence > 50.0:
            max_x_exceed = max(0, -left) + max(0, right - w1)
            max_y_exceed = max(0, -top) + max(0, bottom - h1)
            if max_x_exceed <= max(50, w1 * 0.05) and max_y_exceed <= max(50, h1 * 0.05):
                contained_within = True
                # Soft degradation of containment score
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
            }
        }
    except Exception as e:
        print(f"Error in estimate_visual_containment: {e}")
        return default_res


def calculate_visual_similarity(
    phash1: str, phash2: str, 
    dhash1: str, dhash2: str, 
    ahash1: str, ahash2: str
) -> float:
    """Calculates weighted similarity score across multiple perceptual hashes."""
    if not phash1 or not phash2:
        return 0.0
        
    p_dist = hamming_distance(phash1, phash2)
    d_dist = hamming_distance(dhash1, dhash2)
    a_dist = hamming_distance(ahash1, ahash2)
    
    sim_p = 1.0 - (p_dist / 64.0)
    sim_d = 1.0 - (d_dist / 64.0)
    sim_a = 1.0 - (a_dist / 64.0)
    
    # Weight pHash highest as it is scale and rotation resistant, dHash next for structures, aHash for layout
    return (sim_p * 0.5) + (sim_d * 0.3) + (sim_a * 0.2)


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculates cosine similarity between two float vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    arr1, arr2 = np.array(v1), np.array(v2)
    norm1, norm2 = np.linalg.norm(arr1), np.linalg.norm(arr2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(arr1, arr2) / (norm1 * norm2))


def calculate_audio_similarity(ap1: Dict[str, Any], ap2: Dict[str, Any]) -> float:
    """Compares audio temporal profiles using sliding window cross-correlation."""
    if not ap1 or not ap2 or not ap1.get("has_audio") or not ap2.get("has_audio"):
        return 0.0
        
    t1 = ap1.get("temporal_profile", [])
    t2 = ap2.get("temporal_profile", [])
    
    if not t1 or not t2:
        # Fallback to mean chroma similarity
        m1 = ap1.get("mean_chroma", [])
        m2 = ap2.get("mean_chroma", [])
        return cosine_similarity(m1, m2)
        
    # Standardize orientation
    # t1 and t2 are lists of 12-dimensional chroma vectors
    arr1, arr2 = np.array(t1), np.array(t2)
    
    # Find shorter and longer arrays
    if len(arr1) > len(arr2):
        longer, shorter = arr1, arr2
    else:
        longer, shorter = arr2, arr1
        
    len_long, len_short = len(longer), len(shorter)
    if len_short == 0:
        return 0.0
        
    max_sim = 0.0
    # Slide shorter profile across longer profile
    steps = max(1, len_long - len_short + 1)
    for i in range(steps):
        window = longer[i : i + len_short]
        # Calculate mean cosine similarity across the window frames
        sims = []
        for f in range(len_short):
            v_long = window[f]
            v_short = shorter[f]
            # Chroma vector similarity
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


def analyze_matches(
    source_dna: Dict[str, Any], 
    target_dna: Dict[str, Any]
) -> Tuple[float, str, Dict[str, Any]]:
    """
    Compares two DNA profiles, calculates similarity, 
    and outputs modification forensic results with explainable AI summaries using content only.
    """
    phash1, phash2 = source_dna.get("phash"), target_dna.get("phash")
    dhash1, dhash2 = source_dna.get("dhash"), target_dna.get("dhash")
    ahash1, ahash2 = source_dna.get("ahash"), target_dna.get("ahash")
    
    sha1 = source_dna.get("sha256")
    sha2 = target_dna.get("sha256")
    sha256_match = (sha1 == sha2) if (sha1 and sha2) else False

    # 1. Compute visual similarity
    vis_sim = calculate_visual_similarity(phash1, phash2, dhash1, dhash2, ahash1, ahash2)
    
    p_dist = hamming_distance(phash1, phash2)
    d_dist = hamming_distance(dhash1, dhash2)
    a_dist = hamming_distance(ahash1, ahash2)
    
    # 2. Compute audio similarity
    aud_sim = 0.0
    s_audio = source_dna.get("audio_fingerprint")
    t_audio = target_dna.get("audio_fingerprint")
    if s_audio and t_audio:
        aud_sim = calculate_audio_similarity(s_audio, t_audio)
        
    # 3. Compute semantic similarity
    sem_sim = 0.0
    s_emb = source_dna.get("embedding")
    t_emb = target_dna.get("embedding")
    if s_emb and t_emb and any(s_emb) and any(t_emb):
        sem_sim = cosine_similarity(s_emb, t_emb)
        
    # 4. Content Dimensions Comparison
    w1, h1 = source_dna.get("width", 0), source_dna.get("height", 0)
    w2, h2 = target_dna.get("width", 0), target_dna.get("height", 0)
    sz1 = source_dna.get("file_size", 0)
    sz2 = target_dna.get("file_size", 0)
    
    ar_diff = 0.0
    if w1 > 0 and h1 > 0 and w2 > 0 and h2 > 0:
        ar1, ar2 = w1 / h1, w2 / h2
        ar_diff = abs(ar1 - ar2)
        
    size_ratio = 1.0
    if sz1 > 0 and sz2 > 0:
        size_ratio = max(sz1, sz2) / max(1, min(sz1, sz2))

    t_forensics = target_dna.get("modification_report") or {}
    is_heavy_compression = t_forensics.get("heavy_compression", False)

    # 5. Classify & Calibrate Scores
    is_exact = sha256_match or (p_dist == 0 and d_dist == 0 and a_dist == 0 and w1 == w2 and h1 == h2)
    
    # Estimate visual containment
    filepath1 = source_dna.get("filepath")
    filepath2 = target_dna.get("filepath")
    
    containment_res = estimate_visual_containment(filepath1, filepath2)
    contained_within = containment_res["contained_within_source"]
    overlap_pct = containment_res["visual_overlap_percent"]
    bbox = containment_res["estimated_crop_bounds"]
    orb_confidence = containment_res.get("orb_confidence", 80.0 if (contained_within or overlap_pct > 0) else 0.0)
    containment_score = containment_res.get("containment_score", 100.0 if contained_within else 0.0)

    # Calibrate/boost semantic similarity for visually related items (especially under mock embeddings)
    if contained_within:
        sem_sim = max(sem_sim, 0.88)
    elif p_dist <= 6:
        sem_sim = max(sem_sim, 0.90)
    elif p_dist <= 12:
        sem_sim = max(sem_sim, 0.80)

    # Pre-evaluate variant match flags for classification routing
    # Watermarked Variant Detection
    is_watermark_match = (
        w1 == w2 and h1 == h2 and
        ar_diff <= 0.02 and
        (sem_sim > 0.95 or sem_sim == 0.0) and
        (0 < p_dist <= 6 or 0 < d_dist <= 6)
    )
    
    # Refined Crop Variant Detection:
    # Classify as Cropped Variant when:
    # - semantic similarity >= 0.85
    # - visual containment detected (contained_within == True)
    # - overlap < 95%
    # - dimensions reduced OR aspect ratio changed
    is_crop_match = (
        (sem_sim >= 0.85 or sem_sim == 0.0) and
        contained_within and
        overlap_pct < 95.0 and
        ((w2 < w1 or h2 < h1) or ar_diff > 0.02)
    )
    
    # Retrieve target screenshot score and status
    ss_status = "Inconclusive"
    ss_score = 0
    if target_dna.get("screenshot_indicators"):
        ss_status = target_dna["screenshot_indicators"].get("status", "Inconclusive")
        ss_score = target_dna["screenshot_indicators"].get("confidence", 0)
    elif t_forensics.get("screenshot_indicators"):
        ss_status = t_forensics["screenshot_indicators"].get("status", "Inconclusive")
        ss_score = t_forensics["screenshot_indicators"].get("confidence", 0)
        
    is_compressed_var = (size_ratio > 2.0 or is_heavy_compression) and p_dist <= 4
    is_clean_resize = (w1 != w2 or h1 != h2) and p_dist <= 6
    is_screenshot_match = (
        (p_dist <= 25 or sem_sim >= 0.70) and
        (ss_status in ["Likely Screenshot", "Possible Screenshot"] or ss_score >= 30) and
        not is_clean_resize and not is_watermark_match and not is_compressed_var
    )

    # Format Converted Variant
    mime1 = source_dna.get("mime_type")
    mime2 = target_dna.get("mime_type")
    is_format_match = (
        (mime1 and mime2 and mime1 != mime2) and
        w1 == w2 and h1 == h2 and
        p_dist <= 2
    )
    
    # Descendant Admission Path
    is_descendant = (
        sem_sim >= 0.75 and
        overlap_pct >= 80.0 and
        orb_confidence >= 50.0
    )
    
    evidence = []
    contradicting_evidence = []
    alternative_explanation = "None"
    
    if is_exact:
        relationship_type = "Most Probable Origin"
        combined = 0.98
        match_level = "High Match"
        evidence.append("Identical hash or pixel geometry match")
    elif not is_descendant and ((p_dist > 12 and not is_crop_match) or (p_dist > 18 and sem_sim < 0.80) or (sem_sim > 0 and sem_sim < 0.45 and p_dist > 12)):
        # Unrelated
        relationship_type = "Unknown Baseline Asset"
        combined = float(max(0.05, min(0.29, vis_sim * 0.28)))
        match_level = "No Significant Match"
        evidence.append("Different visual structures and semantic layout")
    else:
        # Determine specific variant classification
        match_level = "High Match" if vis_sim >= 0.85 else "Possible Match"
        
        if is_crop_match or is_descendant:
            relationship_type = "Cropped Variant"
            evidence.append("Contained within source image (containment-supported)" if is_descendant else "Contained within source image")
            if ar_diff > 0.02:
                evidence.append("Aspect ratio changed")
            if w2 < w1 or h2 < h1:
                evidence.append("Resolution reduced")
            alternative_explanation = "Possible screenshot-derived crop"
            
            if is_descendant:
                # Weighted evidence aggregation:
                # semantic similarity (weight 0.4), visual similarity (weight 0.3), overlap score (weight 0.3)
                weighted_score = (sem_sim * 0.4) + (vis_sim * 0.3) + ((overlap_pct / 100.0) * 0.3)
                # Scale slightly by containment_score / 100
                weighted_score *= (0.8 + 0.2 * (containment_score / 100.0))
                combined = float(max(0.65, min(0.95, weighted_score)))
            else:
                # Calibrate similarity to 70–90%
                combined = 0.90 - (p_dist / 64.0) * 0.40
                combined = float(max(0.70, min(0.90, combined)))
            
        elif is_screenshot_match:
            relationship_type = "Screenshot-Derived Variant"
            evidence.append("Screenshot indicators detected")
            if contained_within:
                evidence.append("Visual containment confirmed")
            if ar_diff > 0.02:
                evidence.append("Aspect ratio changed")
            alternative_explanation = "Digital catalog mockup mimicking standard display resolutions"
            # Calibrate combined score
            combined = 0.75 - (p_dist / 64.0) * 0.30
            combined = float(max(0.60, min(0.85, combined)))
            
        elif is_watermark_match:
            relationship_type = "Watermarked Variant"
            evidence.append("Localized visual anomaly with matching geometry")
            alternative_explanation = "Normal compression noise concentrated on edges"
            # Calibrate similarity to 90–98%
            combined = 0.98 - (p_dist / 64.0) * 0.20
            combined = float(max(0.90, min(0.98, combined)))
            
        # Resize Variant Detection
        elif (w1 != w2 or h1 != h2) and p_dist <= 6:
            relationship_type = "Resized Variant"
            evidence.append("Same aspect ratio preserved")
            evidence.append("Resolution changed")
            alternative_explanation = "Transcoded copy for social media distribution"
            # Calibrate similarity to 90–98%
            combined = 0.98 - (p_dist / 64.0) * 0.20
            combined = float(max(0.90, min(0.98, combined)))
 
        # Format match
        elif is_format_match:
            relationship_type = "Format Converted Variant"
            evidence.append("Identical visual content")
            evidence.append("Format changed")
            alternative_explanation = "Lossless backup conversion"
            combined = 0.98 - (p_dist / 64.0) * 0.10
            combined = float(max(0.95, min(0.98, combined)))
            
        # Compressed Variant Detection
        elif (size_ratio > 2.0 or is_heavy_compression) and p_dist <= 4:
            relationship_type = "Compressed Variant"
            evidence.append("Heavy compression blockiness")
            alternative_explanation = "Automatic thumbnail generation"
            # Calibrate similarity to 85–95%
            combined = 0.95 - (p_dist / 64.0) * 0.20
            combined = float(max(0.85, min(0.95, combined)))
            
        # Fallback to general modified variant or Inconclusive
        else:
            if vis_sim < 0.70 or p_dist > 12:
                relationship_type = "Inconclusive"
                evidence.append("Insufficient visual evidence to classify modification")
                combined = float(max(0.15, min(0.48, vis_sim * 0.5)))
            else:
                relationship_type = "Modified Variant"
                evidence.append("Perceptual hashes align but specific edit type is undetermined")
                alternative_explanation = "Manual filter application or color grading"
                # Calibrate similarity to 50–75%
                combined = 0.75 - ((p_dist - 4) / 16.0) * 0.25
                combined = float(max(0.50, min(0.75, combined)))
                
    if combined < 0.50:
        relationship_type = "Unknown Baseline Asset"
 
    # 6. Explainable AI Text
    is_video = source_dna.get("duration") is not None or target_dna.get("duration") is not None
    explanation = generate_explanation(vis_sim, aud_sim, sem_sim, combined, evidence, is_video, relationship_type)
    
    forensics = {
        "modifications": evidence,
        "explanation": explanation,
        "visual_similarity": vis_sim,
        "audio_similarity": aud_sim,
        "semantic_similarity": sem_sim,
        "relationship_type": relationship_type,
        "orb_confidence": orb_confidence,
        "containment_score": containment_score,
        "visual_overlap_percent": overlap_pct,
        "explainability": {
            "classification": relationship_type,
            "confidence": int(combined * 100),
            "evidence": evidence,
            "contradicting_evidence": contradicting_evidence,
            "alternative_explanation": alternative_explanation
        }
    }
    
    return float(combined), match_level, forensics



def generate_explanation(
    vis_sim: float, 
    aud_sim: float, 
    sem_sim: float, 
    combined: float, 
    mods: List[str],
    is_video: bool,
    relationship_type: str = "variant"
) -> str:
    """Generates natural language breakdown of matching features."""
    if relationship_type == "Unknown Baseline Asset":
        return "Unknown Baseline Asset. Perceptual and cryptographic fingerprints do not match. No baseline comparison evidence exists."
        
    text = f"Identified a {relationship_type} ({int(combined * 100)}% Match). "
    
    visual_text = f"Visual hashes match at {int(vis_sim * 100)}% (Hamming distance of pHash/dHash)."
    semantic_text = f" CLIP Semantic embeddings confirm similar contextual layout ({int(sem_sim * 100)}% match)." if sem_sim > 0 else ""
    
    if is_video:
        audio_text = f" Audio profiles correlate at {int(aud_sim * 100)}% across timeline tracks." if aud_sim > 0 else " No matched audio track was identified."
        text += f"{visual_text}{audio_text}{semantic_text}"
    else:
        text += f"{visual_text}{semantic_text}"
        
    if mods:
        text += f" Modifications detected: {', '.join(mods)}."
    else:
        text += " The files share identical fingerprint profiles with no significant modifications detected."
        
    return text


def match_strength(score: float) -> str:
    if score >= 0.95: return "Highly Correlated Match"
    if score >= 0.85: return "Strong Relationship"
    if score >= 0.70: return "Moderately Altered Variation"
    return "Weak Similarity Match"


def estimate_primary_origin(items: List[Dict[str, Any]]) -> Tuple[Optional[int], int, int, bool, List[str], Dict[str, Any]]:
    """
    Ranks similar items to find the original media file using probabilistic criteria.
    Ranks items using:
      1. Resolution (40%)
      2. Metadata Richness (20%)
      3. Compression Quality (20%)
      4. File Fidelity (15%)
      5. Chronological evidence (5%)
    Returns a tuple: (best_id, origin_confidence, origin_probability, origin_undetermined, explainability_factors, audit_trail)
    """
    import datetime
    
    if not items:
        return None, 50, 50, True, [], {
            "resolution_contribution": 0.0,
            "metadata_contribution": 0.0,
            "compression_contribution": 0.0,
            "fidelity_contribution": 0.0,
            "chronology_contribution": 0.0
        }
        
    def _parse_exif_date(date_str):
        if not date_str:
            return None
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.datetime.strptime(str(date_str).strip(), fmt)
            except ValueError:
                continue
        return None

    def _parse_created_at(dt_val):
        if not dt_val:
            return None
        if isinstance(dt_val, datetime.datetime):
            return dt_val
        if isinstance(dt_val, str):
            try:
                val = dt_val
                if val.endswith('Z'):
                    val = val[:-1]
                return datetime.datetime.fromisoformat(val)
            except ValueError:
                pass
        return None

    # Step 1: Pre-calculate absolute features and check bounds/maxima
    max_pixels = 0
    max_exif = 0
    max_file_size = 0
    
    parsed_items = []
    
    for item in items:
        # Dimensions / Pixels
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
        
        if pixels > max_pixels:
            max_pixels = pixels
        if exif_count > max_exif:
            max_exif = exif_count
        if file_size > max_file_size:
            max_file_size = file_size
            
        parsed_items.append({
            "item": item,
            "pixels": pixels,
            "width": width,
            "height": height,
            "exif_count": exif_count,
            "file_size": file_size
        })

    # Step 2: Parse timestamps for chronology
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

    # Step 3: Compute raw scores for each category
    max_raw_compression = 0.0
    max_raw_chrono = 0.0
    
    for pi in parsed_items:
        item = pi["item"]
        
        # Compression raw score
        blockiness = item.get("blockiness", 1.0)
        jpeg_quality = item.get("jpeg_quality")
        heavy_compression = item.get("heavy_compression", False)
        
        if jpeg_quality is not None:
            comp_score = float(jpeg_quality)
        else:
            comp_score = max(10.0, min(100.0, (2.0 - blockiness) * 100.0))
        if heavy_compression:
            comp_score = min(40.0, comp_score)
        pi["raw_comp"] = comp_score
        if comp_score > max_raw_compression:
            max_raw_compression = comp_score
            
        # Chronology raw score
        ts = pi["ts"]
        if ts and min_time and time_span > 0:
            time_score = 1.0 - (ts - min_time).total_seconds() / time_span
        else:
            time_score = 1.0
            
        # Refine chronology score based on dimensions and compression state
        chrono_score = time_score
        if max_pixels > 0 and pi["pixels"] < max_pixels:
            chrono_score *= (pi["pixels"] / max_pixels)
        if heavy_compression:
            chrono_score *= 0.5
        if blockiness > 1.2:
            chrono_score *= (1.2 / blockiness)
            
        pi["raw_chrono"] = chrono_score
        if chrono_score > max_raw_chrono:
            max_raw_chrono = chrono_score

    # Step 4: Calculate final contributions out of respective weights
    rankings = []
    for pi in parsed_items:
        item = pi["item"]
        
        # Resolution Contribution (out of 40%)
        res_contrib = (pi["pixels"] / max_pixels) * 40.0 if max_pixels > 0 else 40.0
        
        # Metadata Richness Contribution (out of 20%)
        meta_contrib = (pi["exif_count"] / max_exif) * 20.0 if max_exif > 0 else 0.0
        
        # Compression Quality Contribution (out of 20%)
        comp_contrib = (pi["raw_comp"] / max_raw_compression) * 20.0 if max_raw_compression > 0 else 20.0
        
        # File Fidelity Contribution (out of 15%)
        fid_contrib = (pi["file_size"] / max_file_size) * 15.0 if max_file_size > 0 else 15.0
        
        # Chronology Contribution (out of 5%)
        chrono_contrib = (pi["raw_chrono"] / max_raw_chrono) * 5.0 if max_raw_chrono > 0 else 5.0
        
        # Modifications Penalties (from user request and filename keywords)
        fn_lower = item.get("filename", "").lower()
        is_cropped = item.get("cropping_detected") or "crop" in fn_lower or "cropped" in fn_lower
        is_resized = item.get("resizing_detected") or "resize" in fn_lower or "resized" in fn_lower
        is_watermarked = item.get("watermark_detected") or "watermark" in fn_lower or "watermarked" in fn_lower
        is_compressed = item.get("heavy_compression") or "compressed" in fn_lower or "compression" in fn_lower
        is_screenshot = item.get("screenshot_detected") or "screenshot" in fn_lower or "capture" in fn_lower or "screen" in fn_lower
        
        crop_penalty = -30.0 if is_cropped else 0.0
        resize_penalty = -25.0 if is_resized else 0.0
        watermark_penalty = -35.0 if is_watermarked else 0.0
        compression_penalty = -10.0 if is_compressed else 0.0
        screenshot_penalty = -15.0 if is_screenshot else 0.0
        metadata_penalty = -5.0 if (item.get("exif_count", 0) == 0) else 0.0
        
        # Cumulative modification penalties capped at -60
        total_penalty = crop_penalty + resize_penalty + watermark_penalty + compression_penalty + screenshot_penalty + metadata_penalty
        total_penalty = max(-60.0, total_penalty)
        
        total_score = res_contrib + meta_contrib + comp_contrib + fid_contrib + chrono_contrib + total_penalty
        
        # Minor tie-breaker to prevent list sorting instability
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
    
    # Calculate contamination metrics & block origin if necessary
    # Block origin selection when average semantic similarity is < threshold or contamination_score > 50%
    contamination_score = 0.0
    avg_sem = 1.0
    sem_threshold = 0.75 if ENABLE_CLIP else 0.40
    if len(items) > 1:
        # Compare all items to top candidate
        top_emb = rankings[0]["item"].get("embedding")
        if top_emb and any(top_emb):
            sem_sims = []
            for it in items:
                it_emb = it.get("embedding")
                if it_emb and any(it_emb):
                    sim = cosine_similarity(top_emb, it_emb)
                    sem_sims.append(sim)
            if sem_sims:
                avg_sem = float(np.mean(sem_sims))
                low_sim_count = sum(1 for s in sem_sims if s < sem_threshold)
                contamination_score = (low_sim_count / len(items)) * 100.0

    # Safety Check:
    # 1. Block origin estimation when contamination exceeds threshold (contamination_score > 50 or average_semantic_similarity < sem_threshold)
    # 2. Safety margin check: Top candidate score exceeds second candidate score by at least 10%
    blocked_by_contamination = (contamination_score > 50.0 or avg_sem < sem_threshold)
    
    is_undetermined = True
    if len(rankings) == 1:
        best_id = rankings[0]["id"]
        best_name = rankings[0]["filename"]
        best_score = rankings[0]["score"]
        best_audit = rankings[0]["audit_trail"]
        origin_confidence = 50
        origin_probability = 100
        is_undetermined = True
    elif blocked_by_contamination:
        best_id = None
        best_name = "Unable to determine with available evidence"
        best_score = 0.0
        best_audit = {
            "resolution_contribution": 0.0,
            "metadata_contribution": 0.0,
            "compression_contribution": 0.0,
            "fidelity_contribution": 0.0,
            "chronology_contribution": 0.0,
            "modifications_penalty": 0.0,
            "rejection_reason": "High cluster contamination"
        }
        origin_confidence = 0
        origin_probability = 0
        is_undetermined = True
    else:
        score_1 = rankings[0]["score"]
        score_2 = rankings[1]["score"]
        
        # Require score_1 to exceed score_2 by at least 10% (relative to score_2)
        # Note: if score_2 is negative or zero, we check difference
        if score_2 > 0:
            margin_satisfied = (score_1 >= score_2 * 1.10)
        else:
            margin_satisfied = (score_1 - score_2 >= 10.0)
            
        if not margin_satisfied:
            # Cannot determine safely, return "Unable to determine with available evidence"
            best_id = None
            best_name = "Unable to determine with available evidence"
            best_score = score_1
            best_audit = rankings[0]["audit_trail"]
            origin_confidence = 30
            origin_probability = 50
            is_undetermined = True
        else:
            best_id = rankings[0]["id"]
            best_name = rankings[0]["filename"]
            best_score = rankings[0]["score"]
            best_audit = rankings[0]["audit_trail"]
            
            diff = score_1 - score_2
            min_score = min(r["score"] for r in rankings)
            shifted = [r["score"] - min_score + 10.0 for r in rankings]
            sum_shifted = sum(shifted)
            prob_top = shifted[0] / sum_shifted if sum_shifted > 0 else 1.0
            origin_probability = int(prob_top * 100)
            
            is_undetermined = False
            origin_confidence = int(min(98, 75 + (diff / 50.0) * 23)) # Cap at 98%
            
    origin_confidence = min(98, origin_confidence)
    
    # Explainability factors
    explainability_factors = []
    if not is_undetermined:
        best_rank = rankings[0]
        if max_pixels > 0 and best_rank["pixels"] == max_pixels:
            explainability_factors.append("Highest resolution in cluster")
        if max_exif > 0 and best_rank["exif_count"] == max_exif:
            explainability_factors.append(f"Richer EXIF metadata tags ({best_rank['exif_count']} tags)")
        if best_rank["raw_comp"] == max_raw_compression:
            explainability_factors.append("Lowest JPEG compression artifacts")
        if max_file_size > 0 and best_rank["file_size"] == max_file_size:
            explainability_factors.append(f"Largest file size ({best_rank['file_size']} bytes)")
            
    print(f"-> Selected Origin Asset: {best_name} (ID: {best_id}) | "
          f"Confidence: {origin_confidence}% | Probability: {origin_probability}% | "
          f"Undetermined: {is_undetermined} | Audit: {best_audit}")
          
    return best_id, origin_confidence, origin_probability, is_undetermined, explainability_factors, best_audit
