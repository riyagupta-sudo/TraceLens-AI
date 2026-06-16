import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.database import SessionLocal
from app.models import MediaItem
from app.similarity_engine import analyze_matches

db = SessionLocal()
try:
    crop = db.query(MediaItem).filter(MediaItem.filename == "building_008_crop.jpg").order_by(MediaItem.id.desc()).first()
    orig = db.query(MediaItem).filter(MediaItem.filename == "building_008.jpg").order_by(MediaItem.id.desc()).first()
    
    if crop and orig:
        print(f"Comparing Crop (ID {crop.id}) and Original (ID {orig.id}):")
        
        src_dna = {
            "phash": crop.phash, "dhash": crop.dhash, "ahash": crop.ahash,
            "embedding": crop.embedding, "audio_fingerprint": crop.audio_fingerprint,
            "width": crop.metadata_sig.get("width", 0) if crop.metadata_sig else 0,
            "height": crop.metadata_sig.get("height", 0) if crop.metadata_sig else 0,
            "file_size": crop.file_size, "mime_type": crop.mime_type,
            "sha256": crop.sha256
        }
        
        tgt_dna = {
            "phash": orig.phash, "dhash": orig.dhash, "ahash": orig.ahash,
            "embedding": orig.embedding, "audio_fingerprint": orig.audio_fingerprint,
            "width": orig.metadata_sig.get("width", 0) if orig.metadata_sig else 0,
            "height": orig.metadata_sig.get("height", 0) if orig.metadata_sig else 0,
            "file_size": orig.file_size, "mime_type": orig.mime_type,
            "sha256": orig.sha256,
            "modification_report": orig.modification_report
        }
        
        combined, level, details = analyze_matches(src_dna, tgt_dna)
        print(f"analyze_matches(crop, orig) ->")
        print(f"  Combined: {combined}")
        print(f"  Level: {level}")
        print(f"  Details: {details}")
        
        # Also test the other direction: analyze_matches(orig, crop)
        combined2, level2, details2 = analyze_matches(tgt_dna, src_dna)
        print(f"\nanalyze_matches(orig, crop) ->")
        print(f"  Combined: {combined2}")
        print(f"  Level: {level2}")
        print(f"  Details: {details2}")
        
    else:
        print("Could not find crop or original in DB.")
finally:
    db.close()
