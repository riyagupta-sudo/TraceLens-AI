import os
import sys
from collections import Counter

# Add backend to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.database import SessionLocal, SQLALCHEMY_DATABASE_URL
from app.models import MediaItem, Case

def investigate():
    print(f"Active SQLALCHEMY_DATABASE_URL config: {SQLALCHEMY_DATABASE_URL}")
    db_path = SQLALCHEMY_DATABASE_URL.replace("sqlite:///", "")
    abs_db_path = os.path.abspath(db_path)
    print(f"Absolute database file path: {abs_db_path}")
    print(f"File exists: {os.path.exists(abs_db_path)}")
    if os.path.exists(abs_db_path):
        print(f"File size: {os.path.getsize(abs_db_path)} bytes")
        print(f"Last modified: {os.path.getmtime(abs_db_path)}")
        
    db = SessionLocal()
    try:
        # 2. Report total media_items count
        total_items = db.query(MediaItem).count()
        print(f"\n2. Total media_items count: {total_items}")
        
        # 3. Report total cluster count
        all_items = db.query(MediaItem).all()
        clusters = [item.cluster_id for item in all_items if item.cluster_id]
        unique_clusters = set(clusters)
        print(f"3. Total cluster count: {len(unique_clusters)}")
        
        # 4. List the largest 10 clusters with family sizes
        cluster_counts = Counter(clusters)
        print("\n4. Largest 10 clusters:")
        for cid, count in cluster_counts.most_common(10):
            # Find representative filename or case
            repr_item = db.query(MediaItem).filter(MediaItem.cluster_id == cid).first()
            case_name = "Unknown Case"
            if repr_item:
                case = db.query(Case).filter(Case.id == repr_item.case_id).first()
                if case:
                    case_name = case.name
            print(f"   Cluster ID: {cid} | Size: {count} | Case: {case_name} | Representative File: {repr_item.filename if repr_item else 'None'}")
            
        # 5. Check whether human_006, drone_original, building_001, pahalgam, and osint assets exist in the same cluster
        print("\n5. Checking target asset cluster alignment:")
        targets = ["human_006", "drone", "building_001", "pahalgam", "pahalgram", "osint"]
        target_items = []
        for term in targets:
            items = db.query(MediaItem).filter(MediaItem.filename.like(f"%{term}%")).all()
            for item in items:
                case = db.query(Case).filter(Case.id == item.case_id).first()
                case_name = case.name if case else "Unknown"
                print(f"   Asset ID: {item.id} | Filename: {item.filename} | Cluster ID: {item.cluster_id} | Case: {case_name} (ID: {item.case_id})")
                target_items.append(item)
                
        # Group them by cluster to see if there is any overlap
        cluster_groups = {}
        for item in target_items:
            cluster_groups.setdefault(item.cluster_id, []).append(item.filename)
            
        print("\n   Grouped by Cluster ID:")
        for cid, filenames in cluster_groups.items():
            if len(filenames) > 1:
                print(f"   [OVERLAP] Cluster {cid} contains multiple target families: {filenames}")
            else:
                print(f"   Cluster {cid} contains: {filenames}")
                
    finally:
        db.close()

if __name__ == "__main__":
    investigate()
