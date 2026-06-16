import os
import sys

# Add backend to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.database import SessionLocal
from app.models import MediaItem, Case
from app.similarity_engine import estimate_primary_origin

db = SessionLocal()
try:
    case = db.query(Case).order_by(Case.id.desc()).first()
    print(f"Case ID: {case.id}")
    
    items = db.query(MediaItem).filter(MediaItem.case_id == case.id, MediaItem.filename.like("%human%")).all()
    cluster_dicts = [
        {
            "id": item.id,
            "filename": item.filename,
            "resolution": item.resolution,
            "file_size": item.file_size,
            "integrity_score": item.integrity_score,
            "width": item.metadata_sig.get("width", 0) if item.metadata_sig else 0,
            "height": item.metadata_sig.get("height", 0) if item.metadata_sig else 0,
            "exif_count": len(item.metadata_sig.get("exif", {})) if item.metadata_sig and item.metadata_sig.get("exif") else 0,
            "heavy_compression": item.modification_report.get("heavy_compression", False) if item.modification_report else False,
            "blockiness": item.metadata_sig.get("blockiness", 1.0) if item.metadata_sig else 1.0,
            "jpeg_quality": item.metadata_sig.get("jpeg_quality") if item.metadata_sig else None,
            "created_at": item.created_at,
            "exif": item.metadata_sig.get("exif", {}) if item.metadata_sig else {},
            "cropping_detected": item.modification_report.get("cropping_detected", False) if item.modification_report else False,
            "resizing_detected": item.modification_report.get("resizing_detected", False) if item.modification_report else False,
            "watermark_detected": item.modification_report.get("watermark_detected", False) if item.modification_report else False,
            "screenshot_detected": (item.modification_report.get("screenshot_indicators", {}).get("status") in ["Likely Screenshot", "Possible Screenshot"]) if item.modification_report else False,
            "embedding": item.embedding
        } for item in items
    ]
    
    # Let's inspect rankings inside estimate_primary_origin by copying its logic or printing it
    # We can just run estimate_primary_origin logic here
    # Step 1: Pre-calculate absolute features and check bounds/maxima
    max_pixels = 0
    max_exif = 0
    max_file_size = 0
    
    parsed_items = []
    
    for item in cluster_dicts:
        width = item.get("width", 0)
        height = item.get("height", 0)
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
            "exif_count": exif_count,
            "file_size": file_size
        })

    import datetime
    timestamps = []
    for pi in parsed_items:
        item = pi["item"]
        timestamps.append(item.get("created_at"))
        pi["ts"] = item.get("created_at")

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
        
        if jpeg_quality is not None:
            comp_score = float(jpeg_quality)
        else:
            comp_score = max(10.0, min(100.0, (2.0 - blockiness) * 100.0))
        if heavy_compression:
            comp_score = min(40.0, comp_score)
        pi["raw_comp"] = comp_score
        if comp_score > max_raw_compression:
            max_raw_compression = comp_score
            
        ts = pi["ts"]
        if ts and min_time and time_span > 0:
            time_score = 1.0 - (ts - min_time).total_seconds() / time_span
        else:
            time_score = 1.0
            
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
        is_screenshot = item.get("screenshot_detected") or "screenshot" in fn_lower or "capture" in fn_lower or "screen" in fn_lower
        
        crop_penalty = -15.0 if is_cropped else 0.0
        resize_penalty = -10.0 if is_resized else 0.0
        watermark_penalty = -20.0 if is_watermarked else 0.0
        compression_penalty = -25.0 if is_compressed else 0.0
        screenshot_penalty = -10.0 if is_screenshot else 0.0
        metadata_penalty = -5.0 if (item.get("exif_count", 0) == 0) else 0.0
        
        total_penalty = crop_penalty + resize_penalty + watermark_penalty + compression_penalty + screenshot_penalty + metadata_penalty
        total_penalty = max(-60.0, total_penalty)
        
        total_score = res_contrib + meta_contrib + comp_contrib + fid_contrib + chrono_contrib + total_penalty
        
        rankings.append({
            "filename": item.get("filename"),
            "score": total_score,
            "res": res_contrib,
            "meta": meta_contrib,
            "comp": comp_contrib,
            "fid": fid_contrib,
            "chrono": chrono_contrib,
            "penalty": total_penalty,
            "penalties_breakdown": {
                "crop": crop_penalty,
                "resize": resize_penalty,
                "watermark": watermark_penalty,
                "compression": compression_penalty,
                "screenshot": screenshot_penalty,
                "metadata": metadata_penalty
            }
        })
        
    rankings.sort(key=lambda x: x["score"], reverse=True)
    for r in rankings:
        print(f"\nCandidate: {r['filename']}")
        print(f"  Total Score: {r['score']}")
        print(f"  Contributions: Res={r['res']:.2f}, Meta={r['meta']:.2f}, Comp={r['comp']:.2f}, Fid={r['fid']:.2f}, Chrono={r['chrono']:.2f}")
        print(f"  Penalty: {r['penalty']}")
        print(f"  Breakdown: {r['penalties_breakdown']}")
    
finally:
    db.close()
