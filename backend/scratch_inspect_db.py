import os
import sys

# Add backend to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.database import SessionLocal
from app.models import MediaItem, Case

db = SessionLocal()
try:
    # Get latest case
    case = db.query(Case).order_by(Case.id.desc()).first()
    if case:
        print(f"Case ID: {case.id} | Name: {case.name}")
        items = db.query(MediaItem).filter(MediaItem.case_id == case.id).all()
        clusters = {}
        for item in items:
            clusters.setdefault(item.cluster_id, []).append(item)
            
        print(f"\nTotal clusters: {len(clusters)}")
        for cid, m_items in clusters.items():
            print(f"\nCluster: {cid} (Size: {len(m_items)})")
            for item in m_items:
                print(f"  ID: {item.id} | Filename: {item.filename} | Origin ID: {item.estimated_origin_id} | Class: {item.modification_report.get('asset_classification') if item.modification_report else 'N/A'}")
    else:
        print("No cases found.")
finally:
    db.close()
