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
        report = item.modification_report or {}
        ss = report.get("screenshot_indicators", {})
        print(f"\nItem: {item.filename}")
        print(f"  screenshot status: {ss.get('status')}")
        print(f"  screenshot score: {ss.get('confidence')}")
        print(f"  screenshot evidence: {ss.get('evidence_matrix', {}).get('evidence_list')}")
        print(f"  asset_classification: {report.get('asset_classification')}")
finally:
    db.close()
