import os
import sys
import shutil
import json
from PIL import Image

# Add backend to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Define NumpyEncoder and custom_dumps here to avoid early dependency on app.database
import numpy as np
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def custom_dumps(obj):
    return json.dumps(obj, cls=NumpyEncoder)

TEMP_DB_URL = "sqlite:///./temp_clean_room.db"
temp_engine = create_engine(TEMP_DB_URL, connect_args={"check_same_thread": False}, json_serializer=custom_dumps)
TempSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=temp_engine)

# Monkeypatch database SessionLocal before app is imported
import app.database as app_db
app_db.SessionLocal = TempSessionLocal

from fastapi.testclient import TestClient
from app.main import app
import app.main as app_main
app_main.SessionLocal = TempSessionLocal
from app.database import get_db, Base
from app.models import MediaItem, Case

def override_get_db():
    db = TempSessionLocal()
    try:
        yield db
    finally:
        db.close()

# Apply dependency override
app.dependency_overrides[get_db] = override_get_db

def cleanup_dynamic_files():
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    # Pahalgam files
    pahalgam_paths = [
        os.path.join(dataset_dir, "originals", "pahalgam.jpg"),
        os.path.join(dataset_dir, "originals", "pahalgam1.png"),
        os.path.join(dataset_dir, "originals", "pahalgam2.png")
    ]
    for p in pahalgam_paths:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

def setup_dynamic_files():
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    nature_path = os.path.join(dataset_dir, "originals", "nature_001.jpg")
    
    if not os.path.exists(nature_path):
        print(f"ERROR: base nature_001.jpg not found at {nature_path}")
        return False
        
    # Copy pahalgam.jpg
    shutil.copy(nature_path, os.path.join(dataset_dir, "originals", "pahalgam.jpg"))
    
    # Generate crop and screenshot variants
    with Image.open(nature_path) as img:
        # Crop 1 (Crop)
        crop1 = img.crop((50, 50, 690, 530))
        crop1.save(os.path.join(dataset_dir, "originals", "pahalgam1.png"), "PNG")
        
        # Crop 2 (Crop Screenshot)
        crop2 = img.crop((100, 100, 580, 460))
        crop2.save(os.path.join(dataset_dir, "originals", "pahalgam2.png"), "PNG")
    return True

def run_validation():
    print("======================================================================")
    print("STARTING CLEAN-ROOM VALIDATION")
    print("======================================================================\n")
    
    # 1. Reset database
    db_file = "./temp_clean_room.db"
    if os.path.exists(db_file):
        os.remove(db_file)
        
    Base.metadata.create_all(bind=temp_engine)
    print(f"Created fresh isolated database at: {TEMP_DB_URL}")
    
    # 2. Setup dynamic files
    setup_dynamic_files()
    
    client = TestClient(app)
    db = TempSessionLocal()
    
    try:
        # 3. Create Case
        case_res = client.post("/api/cases", json={"name": "Clean Room Case", "description": "Clean room verification of clustering and classification"})
        assert case_res.status_code == 200, "Failed to create case"
        case_id = case_res.json()["id"]
        print(f"Created validation Case ID: {case_id}")
        
        # Define 4 families
        families = {
            "human_006": [
                ("originals", "human_006.jpg", "image/jpeg"),
                ("resized", "human_006_resize.jpg", "image/jpeg"),
                ("cropped", "human_006_crop.jpg", "image/jpeg"),
                ("compressed", "human_006_compressed.jpg", "image/jpeg"),
                ("watermarked", "human_006_watermark.jpg", "image/jpeg")
            ],
            "drone": [
                ("case_intel_leak", "drone_orignal.jpg", "image/jpeg"),
                ("case_intel_leak", "drone_crop.jpg", "image/jpeg"),
                ("case_intel_leak", "drone_compressed.jpg", "image/jpeg"),
                ("case_intel_leak", "drone_watermark.jpg", "image/jpeg")
            ],
            "building_001": [
                ("originals", "building_001.jpg", "image/jpeg"),
                ("resized", "building_001_resize.jpg", "image/jpeg"),
                ("cropped", "building_001_crop.jpg", "image/jpeg"),
                ("compressed", "building_001_compressed.jpg", "image/jpeg"),
                ("watermarked", "building_001_watermark.jpg", "image/jpeg")
            ],
            "pahalgam": [
                ("originals", "pahalgam.jpg", "image/jpeg"),
                ("originals", "pahalgam1.png", "image/png"),
                ("originals", "pahalgam2.png", "image/png")
            ]
        }
        
        dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
        uploaded_ids = {}
        
        # 4. Ingest and run pipeline
        for fam_name, files in families.items():
            print(f"\n---> Ingesting Family: {fam_name}")
            uploaded_ids[fam_name] = []
            for subfolder, filename, mime in files:
                file_path = os.path.join(dataset_dir, subfolder, filename)
                if not os.path.exists(file_path):
                    print(f"  [WARNING] File missing: {file_path}")
                    continue
                    
                with open(file_path, "rb") as f:
                    up_res = client.post(
                        "/api/upload",
                        data={"case_id": case_id},
                        files={"file": (filename, f, mime)}
                    )
                assert up_res.status_code == 200, f"Upload failed for {filename}: {up_res.text}"
                item_data = up_res.json()
                uploaded_ids[fam_name].append(item_data["id"])
                print(f"  Uploaded {filename} -> ID: {item_data['id']} | Cluster ID: {item_data['cluster_id']}")
                
        # 5. Generate validation report
        print("\n" + "="*70)
        print("VALIDATION REPORT - ISOLATED ENGINE PERFORMANCE")
        print("="*70)
        
        db_items = db.query(MediaItem).filter(MediaItem.case_id == case_id).all()
        print(f"Total Ingested Items: {len(db_items)}")
        
        # Group items by cluster_id
        cluster_groups = {}
        for item in db_items:
            cluster_groups.setdefault(item.cluster_id, []).append(item)
            
        print(f"Total Visual Clusters Formed: {len(cluster_groups)}")
        
        # Report for each cluster
        cluster_idx = 1
        for cid, items in cluster_groups.items():
            print(f"\nCluster {cluster_idx}: ID = {cid}")
            print(f"  Family Size: {len(items)}")
            
            # Find estimated origin
            origin_item = next((item for item in items if item.id == item.estimated_origin_id), None)
            if origin_item:
                print(f"  Earliest Appearance (Origin): {origin_item.filename} (ID: {origin_item.id})")
            else:
                print("  Earliest Appearance (Origin): Undetermined")
                
            # Diagnostics stats
            repr_item = items[0]
            diag = repr_item.modification_report.get("cluster_diagnostics", {}) if repr_item.modification_report else {}
            c_health = diag.get("cluster_health", "N/A")
            c_score = diag.get("contamination_score", 0)
            avg_sim = diag.get("average_similarity", 0.0)
            avg_sem = diag.get("average_semantic_similarity", 0.0)
            
            print(f"  Cluster Health Status: {c_health}")
            print(f"  Contamination Score: {c_score}%")
            print(f"  Average Visual Similarity: {avg_sim}")
            print(f"  Average Semantic Similarity: {avg_sem}")
            
            print("  Members & Variant Classifications:")
            for item in items:
                report = item.modification_report or {}
                classification = report.get("asset_classification", "N/A")
                screenshot_status = report.get("screenshot_indicators", {}).get("status", "N/A")
                p_id = item.parent_id
                
                print(f"    - Filename: {item.filename}")
                print(f"      * Classification: {classification}")
                print(f"      * Parent Pointer: {p_id}")
                print(f"      * Screenshot Detection: {screenshot_status}")
                print(f"      * Integrity: {item.integrity_score} | Risk: {item.risk_score}")
                
            cluster_idx += 1
            
        # 6. Verification of specific classes
        print("\n" + "="*70)
        print("VARIANT TYPE CLASSIFICATION MATRIX")
        print("="*70)
        
        # We query the classifications of all assets to verify correctness
        classification_matrix = {}
        for item in db_items:
            classification_matrix[item.filename] = {
                "class": item.modification_report.get("asset_classification") if item.modification_report else "N/A",
                "screenshot": item.modification_report.get("screenshot_indicators", {}).get("status") if item.modification_report else "N/A"
            }
            
        # Print verification mapping
        for filename, data in classification_matrix.items():
            print(f"Asset: {filename:<35} | Classification: {data['class']:<25} | Screenshot Indicators: {data['screenshot']}")
            
        # Check contamination count
        print("\n" + "="*70)
        print("CROSS-FAMILY CONTAMINATION SANITY CHECKS")
        print("="*70)
        
        # Check if the 4 families remain in exactly 4 clusters
        family_clusters = {}
        for fam_name, ids in uploaded_ids.items():
            fam_items = db.query(MediaItem).filter(MediaItem.id.in_(ids)).all()
            fam_cluster_ids = {item.cluster_id for item in fam_items}
            family_clusters[fam_name] = fam_cluster_ids
            print(f"Family '{fam_name}' cluster IDs: {fam_cluster_ids} (Count: {len(fam_cluster_ids)})")
            
        overlap_found = False
        all_c_ids = list(family_clusters.values())
        for i in range(len(all_c_ids)):
            for j in range(i+1, len(all_c_ids)):
                intersection = all_c_ids[i].intersection(all_c_ids[j])
                if intersection:
                    overlap_found = True
                    print(f"  [ERROR] Cluster overlap between families: {intersection}")
                    
        if not overlap_found:
            print("  [PASSED] Zero cross-contamination detected! All families remain strictly isolated.")
        else:
            print("  [FAILED] Cross-contamination detected.")
            
    finally:
        db.close()
        cleanup_dynamic_files()
        
        # Delete temporary clean room db file
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
                print("\nTemporary clean-room database file deleted.")
            except Exception:
                pass
                
if __name__ == "__main__":
    run_validation()
