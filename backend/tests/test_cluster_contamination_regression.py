import os
import sys
import shutil
from fastapi.testclient import TestClient

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.database import SessionLocal
from app.models import MediaItem, Case, ClusterMergeRecommendation

def cleanup_cat_files():
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset"))
    paths = [
        os.path.join(dataset_dir, "originals", "cat_001.jpg"),
        os.path.join(dataset_dir, "resized", "cat_001_resize.jpg"),
        os.path.join(dataset_dir, "cropped", "cat_001_crop.jpg"),
        os.path.join(dataset_dir, "compressed", "cat_001_compressed.jpg"),
        os.path.join(dataset_dir, "watermarked", "cat_001_watermark.jpg")
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

def setup_cat_files():
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset"))
    shutil.copy(os.path.join(dataset_dir, "originals", "vehicle_001.jpg"), os.path.join(dataset_dir, "originals", "cat_001.jpg"))
    shutil.copy(os.path.join(dataset_dir, "resized", "vehicle_001_resize.jpg"), os.path.join(dataset_dir, "resized", "cat_001_resize.jpg"))
    shutil.copy(os.path.join(dataset_dir, "cropped", "vehicle_001_crop.jpg"), os.path.join(dataset_dir, "cropped", "cat_001_crop.jpg"))
    shutil.copy(os.path.join(dataset_dir, "compressed", "vehicle_001_compressed.jpg"), os.path.join(dataset_dir, "compressed", "cat_001_compressed.jpg"))
    shutil.copy(os.path.join(dataset_dir, "watermarked", "vehicle_001_watermark.jpg"), os.path.join(dataset_dir, "watermarked", "cat_001_watermark.jpg"))

def test_contamination_regression():
    print("Setting up dynamic cat_001 mock files for contamination regression test...")
    cleanup_cat_files()
    setup_cat_files()
    
    client = TestClient(app)
    db = SessionLocal()
    
    try:
        # 1. Create a clean case
        res = client.post("/api/cases", json={"name": "Cluster Contamination Regression Case", "description": "Regression suite"})
        assert res.status_code == 200, "Failed to create case"
        case_id = res.json()["id"]
        print(f"Created case with ID: {case_id}")
        
        # 2. Define 4 independent families
        families = {
            "human_006": [
                ("originals", "human_006.jpg"),
                ("resized", "human_006_resize.jpg"),
                ("cropped", "human_006_crop.jpg"),
                ("compressed", "human_006_compressed.jpg"),
                ("watermarked", "human_006_watermark.jpg")
            ],
            "drone_original": [
                ("case_intel_leak", "drone_orignal.jpg"),
                ("case_intel_leak", "drone_crop.jpg"),
                ("case_intel_leak", "drone_compressed.jpg"),
                ("case_intel_leak", "drone_watermark.jpg")
            ],
            "building_001": [
                ("originals", "building_001.jpg"),
                ("resized", "building_001_resize.jpg"),
                ("cropped", "building_001_crop.jpg"),
                ("compressed", "building_001_compressed.jpg"),
                ("watermarked", "building_001_watermark.jpg")
            ],
            "cat_001": [
                ("originals", "cat_001.jpg"),
                ("resized", "cat_001_resize.jpg"),
                ("cropped", "cat_001_crop.jpg"),
                ("compressed", "cat_001_compressed.jpg"),
                ("watermarked", "cat_001_watermark.jpg")
            ]
        }
        
        dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset"))
        uploaded_ids = {}
        
        for fam_name, files in families.items():
            print(f"\nIngesting Family: {fam_name}...")
            uploaded_ids[fam_name] = []
            for subfolder, filename in files:
                file_path = os.path.join(dataset_dir, subfolder, filename)
                assert os.path.exists(file_path), f"Test file missing: {file_path}"
                
                with open(file_path, "rb") as f:
                    upload_res = client.post(
                        "/api/upload",
                        data={"case_id": case_id},
                        files={"file": (filename, f, "image/jpeg")}
                    )
                assert upload_res.status_code == 200, f"Upload failed for {filename}: {upload_res.text}"
                item_data = upload_res.json()
                uploaded_ids[fam_name].append(item_data["id"])
                print(f"  Uploaded {filename} -> ID: {item_data['id']} | Cluster: {item_data['cluster_id']}")
                
        # Direct Database Assertions
        print("\n--- Running Contamination Regression Assertions ---")
        items = db.query(MediaItem).filter(MediaItem.case_id == case_id).all()
        print(f"Total uploaded items in case: {len(items)}")
        assert len(items) == 19, f"Expected 19 total items, got {len(items)}"
        
        # 1. Assert exactly 4 separate cluster_ids (Zero contamination)
        cluster_ids = {item.cluster_id for item in items}
        print(f"Unique Cluster IDs: {cluster_ids}")
        assert len(cluster_ids) == 4, f"Expected exactly 4 cluster IDs, got {len(cluster_ids)} ({cluster_ids})"
        
        # 2. Verify each family maps to exactly one cluster and has the correct size
        family_clusters = {}
        for fam_name, ids in uploaded_ids.items():
            fam_items = db.query(MediaItem).filter(MediaItem.id.in_(ids)).all()
            fam_cluster_ids = {item.cluster_id for item in fam_items}
            assert len(fam_cluster_ids) == 1, f"Family {fam_name} split across multiple clusters: {fam_cluster_ids}"
            cluster_id = list(fam_cluster_ids)[0]
            family_clusters[fam_name] = cluster_id
            
            # Check size matches count of uploaded items
            expected_size = len(ids)
            actual_size = db.query(MediaItem).filter(MediaItem.case_id == case_id, MediaItem.cluster_id == cluster_id).count()
            print(f"Family '{fam_name}': expected size {expected_size}, got {actual_size} in cluster {cluster_id}")
            assert actual_size == expected_size, f"Size mismatch for family {fam_name}: expected {expected_size}, got {actual_size}"
            
        # 3. Assert no cross-contamination between family clusters
        assert len(set(family_clusters.values())) == 4, "Families are sharing cluster IDs!"
        print("Cross-contamination verification: PASSED (Clusters are 100% isolated)")
        
        # 4. Assert no merge recommendations were created for this case
        recs = db.query(ClusterMergeRecommendation).filter(ClusterMergeRecommendation.case_id == case_id).all()
        print(f"Cluster merge recommendations count: {len(recs)}")
        assert len(recs) == 0, f"Expected 0 merge recommendations, got {len(recs)}"
        
        print("\nSUCCESS: test_contamination_regression passed cleanly!")
        
    finally:
        db.close()
        cleanup_cat_files()

if __name__ == "__main__":
    test_contamination_regression()
