import os
import sys
import shutil
import json
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

def test_isolated_clustering():
    print("Setting up dynamic cat_001 mock files...")
    cleanup_cat_files()
    setup_cat_files()
    
    client = TestClient(app)
    db = SessionLocal()
    
    try:
        # 1. Create a clean case
        res = client.post("/api/cases", json={"name": "Isolated Clustering Test Case", "description": "Verification Case"})
        assert res.status_code == 200, "Failed to create case"
        case_id = res.json()["id"]
        print(f"Created case with ID: {case_id}")
        
        # 2. Ingest 4 families
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
                
        # Query database directly to verify
        print("\n--- Direct Database Assertions ---")
        items = db.query(MediaItem).filter(MediaItem.case_id == case_id).all()
        print(f"Total uploaded items: {len(items)}")
        assert len(items) == 19, f"Expected 19 total items, got {len(items)}"
        
        # 1. Assert exactly 4 separate cluster_ids
        cluster_ids = {item.cluster_id for item in items}
        print(f"Detected Cluster IDs: {cluster_ids}")
        assert len(cluster_ids) == 4, f"Expected exactly 4 cluster IDs, got {len(cluster_ids)} ({cluster_ids})"
        
        # 2. Assert no cross-family contamination and correct family sizes
        family_clusters = {}
        for fam_name, ids in uploaded_ids.items():
            fam_items = db.query(MediaItem).filter(MediaItem.id.in_(ids)).all()
            fam_cluster_ids = {item.cluster_id for item in fam_items}
            assert len(fam_cluster_ids) == 1, f"Family {fam_name} split across multiple clusters: {fam_cluster_ids}"
            cluster_id = list(fam_cluster_ids)[0]
            family_clusters[fam_name] = cluster_id
            
            # Verify family size matches expected
            expected_size = len(ids)
            actual_size = db.query(MediaItem).filter(MediaItem.case_id == case_id, MediaItem.cluster_id == cluster_id).count()
            print(f"Family '{fam_name}': expected size {expected_size}, got {actual_size}")
            assert actual_size == expected_size, f"Size mismatch for family {fam_name}: expected {expected_size}, got {actual_size}"
            
        # 3. Assert no cross-contamination between families
        assert len(set(family_clusters.values())) == 4, "Some families share the same cluster ID!"
        print("Cross-contamination check: PASSED (All 4 families are in separate clusters)")
        
        # 4. Assert no merge recommendations are created for this case
        recs = db.query(ClusterMergeRecommendation).filter(ClusterMergeRecommendation.case_id == case_id).all()
        print(f"Generated merge recommendations count: {len(recs)}")
        assert len(recs) == 0, f"Expected 0 merge recommendations, got {len(recs)}"
        
        # 5. Verify classifications survive persistence
        # We check specific classification for each variant type in human_006 family
        human_items = db.query(MediaItem).filter(MediaItem.id.in_(uploaded_ids["human_006"])).all()
        classifications = {item.filename: item.modification_report.get("asset_classification") for item in human_items if item.modification_report}
        
        print(f"Human Family Classifications:")
        for fn, cls in classifications.items():
            print(f"  {fn}: {cls}")
            
        assert classifications.get("human_006.jpg") == "Most Probable Origin"
        assert classifications.get("human_006_resize.jpg") == "Resized Variant"
        assert classifications.get("human_006_crop.jpg") == "Cropped Variant"
        assert classifications.get("human_006_compressed.jpg") == "Compressed Variant"
        assert classifications.get("human_006_watermark.jpg") == "Watermarked Variant"
        
        # 6. Verify API classifications match DB classifications
        print("\n--- API Response Assertions ---")
        for fam_name, ids in uploaded_ids.items():
            for i_id in ids:
                res_detail = client.get(f"/api/media/{i_id}")
                assert res_detail.status_code == 200
                api_item = res_detail.json()
                
                db_item = db.query(MediaItem).filter(MediaItem.id == i_id).first()
                
                db_cls = db_item.modification_report.get("asset_classification")
                api_cls = api_item["modification_report"].get("asset_classification")
                assert api_cls == db_cls, f"API classification mismatch for ID {i_id}: DB='{db_cls}', API='{api_cls}'"
                
        print("All API classification matching assertions: PASSED")
        print("\nSUCCESS: test_isolated_clustering passed cleanly!")
        
    finally:
        db.close()
        cleanup_cat_files()

if __name__ == "__main__":
    test_isolated_clustering()
