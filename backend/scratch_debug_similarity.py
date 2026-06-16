import os
import sys

# Add backend to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.database import SessionLocal
from app.models import MediaItem
from app.similarity_engine import analyze_matches, hamming_distance, estimate_visual_containment

db = SessionLocal()
try:
    human = db.query(MediaItem).filter(MediaItem.filename == "human_006.jpg").order_by(MediaItem.id.desc()).first()
    drone = db.query(MediaItem).filter(MediaItem.filename == "drone_orignal.jpg").order_by(MediaItem.id.desc()).first()
    
    if human and drone:
        print(f"Human: ID {human.id}, pHash: {human.phash}, dHash: {human.dhash}, aHash: {human.ahash}")
        print(f"Drone: ID {drone.id}, pHash: {drone.phash}, dHash: {drone.dhash}, aHash: {drone.ahash}")
        
        p_dist = hamming_distance(human.phash, drone.phash)
        print(f"pHash Distance: {p_dist}")
        
        # Paths
        dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
        human_path = os.path.join(dataset_dir, "originals", "human_006.jpg")
        drone_path = os.path.join(dataset_dir, "case_intel_leak", "drone_orignal.jpg")
        
        containment = estimate_visual_containment(human_path, drone_path)
        print(f"Containment: {containment}")
        
        src_dna = {
            "phash": human.phash, "dhash": human.dhash, "ahash": human.ahash,
            "embedding": human.embedding, "audio_fingerprint": human.audio_fingerprint,
            "width": human.metadata_sig.get("width", 0), "height": human.metadata_sig.get("height", 0),
            "file_size": human.file_size, "mime_type": human.mime_type,
            "sha256": human.sha256, "filepath": human_path
        }
        tgt_dna = {
            "phash": drone.phash, "dhash": drone.dhash, "ahash": drone.ahash,
            "embedding": drone.embedding, "audio_fingerprint": drone.audio_fingerprint,
            "width": drone.metadata_sig.get("width", 0), "height": drone.metadata_sig.get("height", 0),
            "file_size": drone.file_size, "mime_type": drone.mime_type,
            "sha256": drone.sha256, "filepath": drone_path
        }
        
        combined, level, details = analyze_matches(src_dna, tgt_dna)
        print(f"Combined Similarity: {combined} ({level})")
        print(f"Details: {details}")
    else:
        print("Human or drone not found in DB.")
finally:
    db.close()
