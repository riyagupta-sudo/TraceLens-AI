from app.database import SessionLocal
from app.models import MediaItem, Case, ClusterMergeRecommendation

db = SessionLocal()
try:
    print("Listing Cases in DB:")
    cases = db.query(Case).all()
    for case in cases:
        print(f"Case ID: {case.id} | Name: {case.name} | Desc: {case.description}")
        
    print("\nListing Media Items in DB:")
    items = db.query(MediaItem).all()
    for item in items:
        print(f"ID: {item.id} | Case ID: {item.case_id} | Filename: {item.filename} | Cluster ID: {item.cluster_id} | Origin ID: {item.estimated_origin_id}")
        report = item.modification_report or {}
        print(f"  Classification: {report.get('asset_classification')} | Risk: {item.risk_score} | Integrity: {item.integrity_score}")
        print(f"  Mime: {item.mime_type} | Size: {item.file_size} bytes | Resolution: {item.resolution}")
        
    print("\nListing Merge Recommendations in DB:")
    recs = db.query(ClusterMergeRecommendation).all()
    for rec in recs:
        print(f"ID: {rec.id} | Case ID: {rec.case_id} | Src Cluster: {rec.source_cluster_id} | Tgt Cluster: {rec.target_cluster_id} | Status: {rec.status} | Conf: {rec.confidence:.4f}")
finally:
    db.close()
