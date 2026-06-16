import os
import sys

# Add backend to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.database import SessionLocal
from app.models import MediaItem
from app.similarity_engine import analyze_matches, hamming_distance, estimate_visual_containment

db = SessionLocal()
try:
    drone = db.query(MediaItem).filter(MediaItem.filename == "drone_orignal.jpg").order_by(MediaItem.id.desc()).first()
    others = db.query(MediaItem).filter(MediaItem.case_id == drone.case_id, MediaItem.id < drone.id).all()
    
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    
    print(f"Drone ID: {drone.id}, filename: {drone.filename}")
    for item in others:
        # Resolve physical paths
        if "human_006" in item.filename:
            item_path = os.path.join(dataset_dir, "originals", item.filename.replace("_resize", "").replace("_crop", "").replace("_compressed", "").replace("_watermark", ""))
            if "_resize" in item.filename:
                item_path = os.path.join(dataset_dir, "resized", item.filename)
            elif "_crop" in item.filename:
                item_path = os.path.join(dataset_dir, "cropped", item.filename)
            elif "_compressed" in item.filename:
                item_path = os.path.join(dataset_dir, "compressed", item.filename)
            elif "_watermark" in item.filename:
                item_path = os.path.join(dataset_dir, "watermarked", item.filename)
        else:
            continue
            
        drone_path = os.path.join(dataset_dir, "case_intel_leak", "drone_orignal.jpg")
        
        src_dna = {
            "phash": item.phash, "dhash": item.dhash, "ahash": item.ahash,
            "embedding": item.embedding, "audio_fingerprint": item.audio_fingerprint,
            "width": item.metadata_sig.get("width", 0) if item.metadata_sig else 0,
            "height": item.metadata_sig.get("height", 0) if item.metadata_sig else 0,
            "file_size": item.file_size, "mime_type": item.mime_type,
            "sha256": item.sha256, "filepath": item_path
        }
        tgt_dna = {
            "phash": drone.phash, "dhash": drone.dhash, "ahash": drone.ahash,
            "embedding": drone.embedding, "audio_fingerprint": drone.audio_fingerprint,
            "width": drone.metadata_sig.get("width", 0) if drone.metadata_sig else 0,
            "height": drone.metadata_sig.get("height", 0) if drone.metadata_sig else 0,
            "file_size": drone.file_size, "mime_type": drone.mime_type,
            "sha256": drone.sha256, "filepath": drone_path
        }
        
        combined, level, details = analyze_matches(src_dna, tgt_dna)
        p_dist = hamming_distance(item.phash, drone.phash)
        sem_sim = details.get("semantic_similarity", 0.0)
        
        # Estimate visual containment
        containment_res = estimate_visual_containment(src_dna["filepath"], tgt_dna["filepath"])
        contained_within = containment_res["contained_within_source"]
        
        print(f"\nComparing to {item.filename} (ID: {item.id}, Cluster: {item.cluster_id}):")
        print(f"  combined_similarity: {combined:.4f}")
        print(f"  semantic_similarity: {sem_sim:.4f}")
        print(f"  pHash Distance: {p_dist}")
        print(f"  Contained within: {contained_within}")
        
        # Check gate logic
        gate_passed = (
            combined >= 0.50 and
            sem_sim >= 0.75 and
            (contained_within or p_dist <= 12)
        )
        print(f"  GATE PASSED: {gate_passed}")
        
finally:
    db.close()
