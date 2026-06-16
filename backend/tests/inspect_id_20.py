import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "OneDrive", "Desktop", "TraceLens AI", "backend")))
from app.database import SessionLocal
from app.models import MediaItem

def run():
    db = SessionLocal()
    try:
        item = db.query(MediaItem).filter(MediaItem.id == 20).first()
        if item:
            print(f"ID: {item.id} | Filename: {item.filename} | Cluster: {item.cluster_id}")
            print(f"  Parent: {item.parent_id} | Origin: {item.estimated_origin_id}")
            print(f"  Integrity: {item.integrity_score} | Risk: {item.risk_score}")
            print(f"  Mime type: {item.mime_type} | Size: {item.file_size} | Res: {item.resolution}")
            print(f"  pHash: {item.phash}")
            print(f"  Classification: {item.modification_report.get('asset_classification') if item.modification_report else None}")
        else:
            print("Item ID 20 not found.")
    finally:
        db.close()

if __name__ == "__main__":
    run()
