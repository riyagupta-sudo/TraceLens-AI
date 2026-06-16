import os
import sys

# Add backend to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.database import SessionLocal
from app.models import MediaItem, Case

db = SessionLocal()
try:
    case = db.query(Case).order_by(Case.id.desc()).first()
    print(f"Case: {case.name} (ID: {case.id})")
    
    items = db.query(MediaItem).filter(MediaItem.case_id == case.id, MediaItem.filename.like("%human%")).all()
    for item in items:
        report = item.modification_report or {}
        print(f"\nItem: {item.filename} (ID: {item.id})")
        print(f"  estimated_origin_id: {item.estimated_origin_id}")
        print(f"  parent_id: {item.parent_id}")
        print(f"  asset_classification: {report.get('asset_classification')}")
        print(f"  relationship_type (relationship_analysis): {report.get('relationship_analysis', {}).get('relationship_type')}")
        print(f"  integrity_score: {item.integrity_score} | risk_score: {item.risk_score}")
finally:
    db.close()
