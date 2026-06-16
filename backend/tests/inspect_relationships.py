import sys
import os
from sqlalchemy.orm import Session

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.database import SessionLocal
from app.models import MediaItem, MediaRelationship, Case

def inspect():
    db = SessionLocal()
    try:
        # Find the latest Case
        case = db.query(Case).order_by(Case.id.desc()).first()
        if not case:
            print("No cases found.")
            return
        print(f"Inspecting Case ID: {case.id} Name: {case.name}")
        
        # Query all MediaItems
        items = db.query(MediaItem).filter(MediaItem.case_id == case.id).all()
        print("\n--- Media Items ---")
        for item in items:
            print(f"ID: {item.id} | Name: {item.filename}")
            print(f"  Width: {item.metadata_sig.get('width')} | Height: {item.metadata_sig.get('height')} | Size: {item.file_size}")
            print(f"  pHash: {item.phash} | dHash: {item.dhash} | aHash: {item.ahash}")
            print(f"  Estimated Origin ID: {item.estimated_origin_id} | Parent ID: {item.parent_id}")
            print(f"  Integrity: {item.integrity_score} | Risk: {item.risk_score}")
            print(f"  Classification: {item.modification_report.get('asset_classification') if item.modification_report else None}")
            if item.modification_report and 'relationship_analysis' in item.modification_report:
                print(f"  Rel Analysis: {item.modification_report['relationship_analysis']}")
            print()
            
        # Query all MediaRelationships
        relationships = db.query(MediaRelationship).all()
        print("\n--- Media Relationships ---")
        for rel in relationships:
            # Let's see if the source or target belongs to our items
            src = db.query(MediaItem).filter(MediaItem.id == rel.source_id).first()
            tgt = db.query(MediaItem).filter(MediaItem.id == rel.target_id).first()
            if src and src.case_id == case.id:
                print(f"Rel: Source: {src.filename} (ID: {src.id}) -> Target: {tgt.filename} (ID: {tgt.id})")
                print(f"  Visual Sim: {rel.visual_similarity:.4f} | Combined: {rel.combined_score:.4f} | Type: {rel.relationship_type}")
                
    finally:
        db.close()

if __name__ == "__main__":
    inspect()
