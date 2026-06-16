import os
import sys

# Add backend to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.database import SessionLocal
from app.models import MediaItem, Case

db = SessionLocal()
try:
    case = db.query(Case).order_by(Case.id.desc()).first()
    items = db.query(MediaItem).filter(MediaItem.case_id == case.id, MediaItem.filename.like("%human%")).all()
    for item in items:
        meta = item.metadata_sig or {}
        print(f"\nItem: {item.filename}")
        print(f"  jpeg_quality: {meta.get('jpeg_quality')}")
        print(f"  blockiness: {meta.get('blockiness')}")
        print(f"  heavy_compression (report): {item.modification_report.get('heavy_compression') if item.modification_report else 'N/A'}")
finally:
    db.close()
