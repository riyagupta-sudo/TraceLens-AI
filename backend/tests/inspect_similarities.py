import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "OneDrive", "Desktop", "TraceLens AI", "backend")))
from app.database import SessionLocal
from app.models import MediaItem
from app.similarity_engine import analyze_matches

def run():
    db = SessionLocal()
    try:
        drone = db.query(MediaItem).filter(MediaItem.id == 20).first()
        human = db.query(MediaItem).filter(MediaItem.filename == 'human_006.jpg').first()
        if not drone or not human:
            print("Drone or Human missing.")
            return
        
        # Build DNA
        drone_dna = {
            "phash": drone.phash, "dhash": drone.dhash, "ahash": drone.ahash,
            "embedding": drone.embedding, "audio_fingerprint": drone.audio_fingerprint,
            "width": drone.metadata_sig.get("width", 0), "height": drone.metadata_sig.get("height", 0),
            "file_size": drone.file_size, "mime_type": drone.mime_type, "sha256": drone.sha256
        }
        human_dna = {
            "phash": human.phash, "dhash": human.dhash, "ahash": human.ahash,
            "embedding": human.embedding, "audio_fingerprint": human.audio_fingerprint,
            "width": human.metadata_sig.get("width", 0), "height": human.metadata_sig.get("height", 0),
            "file_size": human.file_size, "mime_type": human.mime_type, "sha256": human.sha256
        }
        
        combined, level, details = analyze_matches(drone_dna, human_dna)
        print(f"Compare Drone and Human:")
        print(f"  Combined: {combined}")
        print(f"  Level: {level}")
        print(f"  Details: {details}")
    finally:
        db.close()

if __name__ == "__main__":
    run()
