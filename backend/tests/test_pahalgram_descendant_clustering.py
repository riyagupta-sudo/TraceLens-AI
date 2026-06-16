import os
import sys
import shutil
from PIL import Image
from fastapi.testclient import TestClient

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.database import SessionLocal
from app.models import MediaItem, Case

def cleanup_pahalgram_files():
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset"))
    paths = [
        os.path.join(dataset_dir, "originals", "pahalgram.jpg"),
        os.path.join(dataset_dir, "originals", "pahalgram1.png"),
        os.path.join(dataset_dir, "originals", "pahalgram2.png")
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception as e:
                print(f"Cleanup error for {p}: {e}")

def setup_pahalgram_files():
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset"))
    nature_path = os.path.join(dataset_dir, "originals", "building_009.jpg")
    
    assert os.path.exists(nature_path), f"Base file building_009.jpg not found at {nature_path}"
    
    # 1. pahalgram.jpg (Copy of building_009.jpg)
    orig_dest = os.path.join(dataset_dir, "originals", "pahalgram.jpg")
    shutil.copy(nature_path, orig_dest)
    
    # 2. pahalgram1.png (Cropped portion of building_009.jpg)
    with Image.open(nature_path) as img:
        # Crop to standard 4:3 aspect ratio, slightly smaller
        crop1 = img.crop((50, 50, 690, 530)) # 640x480 size
        crop1.save(os.path.join(dataset_dir, "originals", "pahalgram1.png"), "PNG")
        
        # 3. pahalgram2.png (A different cropped portion)
        crop2 = img.crop((100, 100, 580, 460)) # 480x360 size
        crop2.save(os.path.join(dataset_dir, "originals", "pahalgram2.png"), "PNG")

def test_pahalgram_descendant_clustering():
    print("Setting up pahalgram mock files on disk...")
    cleanup_pahalgram_files()
    setup_pahalgram_files()
    
    client = TestClient(app)
    db = SessionLocal()
    
    try:
        # 1. Create a clean case
        res = client.post("/api/cases", json={"name": "Pahalgram Descendant Clustering Case", "description": "Verification Case"})
        assert res.status_code == 200, "Failed to create case"
        case_id = res.json()["id"]
        print(f"Created case with ID: {case_id}")
        
        dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset"))
        uploaded_items = []
        
        # Files to upload
        files_to_upload = [
            ("pahalgram.jpg", "image/jpeg"),
            ("pahalgram1.png", "image/png"),
            ("pahalgram2.png", "image/png")
        ]
        
        for filename, mime in files_to_upload:
            file_path = os.path.join(dataset_dir, "originals", filename)
            assert os.path.exists(file_path), f"Test file missing: {file_path}"
            
            with open(file_path, "rb") as f:
                upload_res = client.post(
                    "/api/upload",
                    data={"case_id": case_id},
                    files={"file": (filename, f, mime)}
                )
            assert upload_res.status_code == 200, f"Upload failed for {filename}: {upload_res.text}"
            item_data = upload_res.json()
            uploaded_items.append(item_data)
            print(f"Uploaded {filename} -> ID: {item_data['id']} | Cluster: {item_data['cluster_id']}")
            
        # Ingest an unrelated file: building_001.jpg
        unrelated_filename = "building_001.jpg"
        unrelated_path = os.path.join(dataset_dir, "originals", unrelated_filename)
        assert os.path.exists(unrelated_path), f"Unrelated file missing: {unrelated_path}"
        
        with open(unrelated_path, "rb") as f:
            unrelated_res = client.post(
                "/api/upload",
                data={"case_id": case_id},
                files={"file": (unrelated_filename, f, "image/jpeg")}
            )
        assert unrelated_res.status_code == 200, f"Unrelated upload failed: {unrelated_res.text}"
        unrelated_data = unrelated_res.json()
        print(f"Uploaded unrelated {unrelated_filename} -> ID: {unrelated_data['id']} | Cluster: {unrelated_data['cluster_id']}")
        
        # Verify database records
        db_items = db.query(MediaItem).filter(MediaItem.case_id == case_id).all()
        assert len(db_items) == 4, f"Expected 4 items, got {len(db_items)}"
        
        # Assertions:
        # A. pahalgram.jpg, pahalgram1.png, and pahalgram2.png must belong to the SAME cluster.
        pahalgram_db_items = [it for it in db_items if "pahalgram" in it.filename]
        assert len(pahalgram_db_items) == 3, f"Expected 3 pahalgram items, got {len(pahalgram_db_items)}"
        pahalgram_cluster_ids = {it.cluster_id for it in pahalgram_db_items}
        assert len(pahalgram_cluster_ids) == 1, f"Expected exactly 1 cluster for pahalgram family, got {len(pahalgram_cluster_ids)}: {pahalgram_cluster_ids}"
        pahalgram_cluster_id = list(pahalgram_cluster_ids)[0]
        print(f"[PASSED] All pahalgram items grouped under same cluster: {pahalgram_cluster_id}")
        
        # B. pahalgram.jpg remains the origin.
        pahalgram_orig = next(it for it in pahalgram_db_items if it.filename == "pahalgram.jpg")
        assert pahalgram_orig.estimated_origin_id == pahalgram_orig.id, "pahalgram.jpg is not estimated as its own origin"
        assert pahalgram_orig.modification_report.get("asset_classification") == "Most Probable Origin"
        print("[PASSED] pahalgram.jpg successfully selected as Most Probable Origin.")
        
        # C. pahalgram1.png classified as Cropped/Screenshot Variant.
        pahalgram1_db = next(it for it in pahalgram_db_items if it.filename == "pahalgram1.png")
        cls1 = pahalgram1_db.modification_report.get("asset_classification")
        assert cls1 in ("Cropped Variant", "Screenshot-Derived Variant"), f"Unexpected classification for pahalgram1: {cls1}"
        print(f"[PASSED] pahalgram1.png classified as: {cls1}")
        
        # D. pahalgram2.png classified as Cropped/Screenshot Variant.
        pahalgram2_db = next(it for it in pahalgram_db_items if it.filename == "pahalgram2.png")
        cls2 = pahalgram2_db.modification_report.get("asset_classification")
        assert cls2 in ("Cropped Variant", "Screenshot-Derived Variant"), f"Unexpected classification for pahalgram2: {cls2}"
        print(f"[PASSED] pahalgram2.png classified as: {cls2}")
        
        # E. No false merge with unrelated assets.
        unrelated_db = next(it for it in db_items if it.filename == unrelated_filename)
        assert unrelated_db.cluster_id != pahalgram_cluster_id, "False merge detected! Unrelated asset building_001.jpg joined pahalgram cluster."
        print(f"[PASSED] Unrelated asset placed in separate cluster: {unrelated_db.cluster_id}")
        
        print("\nSUCCESS: test_pahalgram_descendant_clustering passed cleanly!")
        
    finally:
        db.close()
        cleanup_pahalgram_files()

if __name__ == "__main__":
    test_pahalgram_descendant_clustering()
