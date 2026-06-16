from app.database import SessionLocal
from app.models import MediaItem
import json

db = SessionLocal()
try:
    items = db.query(MediaItem).filter(MediaItem.case_id == 2).all()
    for item in items:
        emb_summary = f"len={len(item.embedding or [])}"
        if item.embedding:
            emb_summary += f" sum={sum(item.embedding):.4f} nonzero={any(item.embedding)}"
        print(f"ID: {item.id} | Filename: {item.filename} | pHash: {item.phash} | size: {item.file_size} | Embedding: {emb_summary}")
finally:
    db.close()
