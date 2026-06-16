from app.database import SessionLocal
from app.models import MediaItem

db = SessionLocal()
try:
    for cid in [2, 6, 7]:
        print(f"\n=== CASE ID: {cid} ===")
        items = db.query(MediaItem).filter(MediaItem.case_id == cid).all()
        for item in items:
            w = item.metadata_sig.get("width", 0) if item.metadata_sig else 0
            h = item.metadata_sig.get("height", 0) if item.metadata_sig else 0
            print(f"ID: {item.id} | Filename: {item.filename} | pHash: {item.phash} | size: {item.file_size} | Dim: {w}x{h} ({item.resolution})")
finally:
    db.close()
