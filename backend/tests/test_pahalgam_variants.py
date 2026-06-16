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

def cleanup_pahalgam_files():
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset"))
    paths = [
        os.path.join(dataset_dir, "originals", "pahalgam.jpg"),
        os.path.join(dataset_dir, "originals", "pahalgam1.png"),
        os.path.join(dataset_dir, "originals", "pahalgam2.png")
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception as e:
                print(f"Cleanup error for {p}: {e}")

def setup_pahalgam_files():
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset"))
    nature_path = os.path.join(dataset_dir, "originals", "nature_001.jpg")
    
    assert os.path.exists(nature_path), f"Base file nature_001.jpg not found at {nature_path}"
    
    # 1. pahalgam.jpg (Copy of nature_001.jpg)
    orig_dest = os.path.join(dataset_dir, "originals", "pahalgam.jpg")
    shutil.copy(nature_path, orig_dest)
    
    # 2. pahalgam1.png (Cropped portion of nature_001.jpg)
    with Image.open(nature_path) as img:
        # Crop to standard 4:3 aspect ratio, slightly smaller
        crop1 = img.crop((50, 50, 690, 530)) # 640x480 size
        crop1.save(os.path.join(dataset_dir, "originals", "pahalgam1.png"), "PNG")
        
        # 3. pahalgam2.png (A different cropped portion)
        crop2 = img.crop((100, 100, 580, 460)) # 480x360 size
        crop2.save(os.path.join(dataset_dir, "originals", "pahalgam2.png"), "PNG")

def test_pahalgam_variants():
    print("Running Pahalgam Variants Forensic Test...")
    cleanup_pahalgam_files()
    setup_pahalgam_files()
    
    client = TestClient(app)
    db = SessionLocal()
    
    try:
        # Create case
        res = client.post("/api/cases", json={"name": "Pahalgam Forensic Case", "description": "Verification of Pahalgam Variants"})
        assert res.status_code == 200, "Failed to create case"
        case_id = res.json()["id"]
        print(f"Created case with ID: {case_id}")
        
        dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset"))
        uploaded_items = []
        
        files_to_upload = [
            "pahalgam.jpg",
            "pahalgam1.png",
            "pahalgam2.png"
        ]
        
        for filename in files_to_upload:
            file_path = os.path.join(dataset_dir, "originals", filename)
            mime = "image/png" if filename.endswith(".png") else "image/jpeg"
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
            
        # Get dynamic stats from DB
        items = db.query(MediaItem).filter(MediaItem.case_id == case_id).all()
        assert len(items) == 3, f"Expected 3 items in case, got {len(items)}"
        
        # Verify all belong to same cluster (Family isolation/containment check)
        cluster_ids = {item.cluster_id for item in items}
        assert len(cluster_ids) == 1, f"Expected exactly 1 cluster for pahalgam family, got {len(cluster_ids)}"
        cluster_id = list(cluster_ids)[0]
        
        # Verify origin estimation re-evaluated from DB correctly
        origin_item = next((item for item in items if item.id == item.estimated_origin_id), None)
        assert origin_item is not None, "Origin item is not set or not in cluster"
        assert origin_item.filename == "pahalgam.jpg", f"Expected origin to be pahalgam.jpg, got {origin_item.filename}"
        
        # Verify each item's classifications and scores
        for item in items:
            report = item.modification_report
            classification = report.get("asset_classification")
            print(f"\nItem: {item.filename}")
            print(f"  Classification: {classification}")
            print(f"  Integrity: {item.integrity_score} | Risk: {item.risk_score}")
            print(f"  Screenshot status: {report.get('screenshot_indicators', {}).get('status')}")
            print(f"  Cropping detected: {report.get('cropping_detected')}")
            
            if item.filename == "pahalgam.jpg":
                # Origin checks
                assert classification == "Most Probable Origin"
                assert item.integrity_score == 100
                assert item.risk_score == 0
            else:
                # Variant checks
                assert classification in ("Cropped Variant", "Screenshot-Derived Variant")
                # Variant integrity must be strictly lower than origin
                assert item.integrity_score < 100, f"Variant {item.filename} has integrity {item.integrity_score}, expected < 100"
                # Variant risk must be strictly positive
                assert item.risk_score > 0, f"Variant {item.filename} has risk {item.risk_score}, expected > 0"
                # Both must be detected as cropped and screenshots
                assert report.get("cropping_detected") is True, f"Cropping not detected for variant {item.filename}"
                assert report.get("screenshot_indicators", {}).get("status") in ("Likely Screenshot", "Possible Screenshot"), f"Incorrect screenshot status for {item.filename}"
                
        # Assertions on final score relations
        origin = next(x for x in items if x.filename == "pahalgam.jpg")
        variant1 = next(x for x in items if x.filename == "pahalgam1.png")
        variant2 = next(x for x in items if x.filename == "pahalgam2.png")
        
        # Integrity: Origin > Variants (asserting variant cannot exceed origin)
        assert origin.integrity_score > variant1.integrity_score, "Origin integrity should be higher than variant1"
        assert origin.integrity_score > variant2.integrity_score, "Origin integrity should be higher than variant2"
        
        # Risk: Origin < Variants
        assert origin.risk_score < variant1.risk_score, "Origin risk should be lower than variant1"
        assert origin.risk_score < variant2.risk_score, "Origin risk should be lower than variant2"
        
        print("\nSUCCESS: test_pahalgam_variants passed cleanly!")
        
    finally:
        db.close()
        cleanup_pahalgam_files()

if __name__ == "__main__":
    test_pahalgam_variants()
