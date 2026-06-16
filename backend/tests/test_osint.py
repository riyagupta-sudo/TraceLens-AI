import os
import sys
import json
import time

# Adjust path to import from app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app.main import app
from app.database import get_db, engine
from app.models import MediaItem, OSINTScan, OSINTResult
from sqlalchemy.orm import Session

client = TestClient(app)

def test_osint_flow():
    print("=" * 60)
    print("RUNNING OSINT PIPELINE AUTOMATED TESTS")
    print("=" * 60)

    # 1. Get a media item from the database
    with Session(engine) as db:
        media_item = db.query(MediaItem).first()
        if not media_item:
            print("No media items found in database to run OSINT test on.")
            # Create a quick dummy media item for testing
            media_item = MediaItem(
                case_id=1,
                filename="test_drone_telemetry.jpg",
                filepath="/media/uploads/test_drone_telemetry.jpg",
                mime_type="image/jpeg",
                sha256="da39a3ee5e6b4b0d3255bfef95601890afd80709",
                phash="1234567890abcdef",
                file_size=1024,
                risk_score=10,
                integrity_score=90
            )
            db.add(media_item)
            db.commit()
            db.refresh(media_item)
            print(f"Created temporary media item ID: {media_item.id}")

        media_id = media_item.id
        print(f"Testing OSINT pipeline for Media ID: {media_id} (Filename: {media_item.filename})")
        
        # Clean up any existing scan and results to ensure clean start
        db.query(OSINTScan).filter(OSINTScan.media_id == media_id).delete()
        db.query(OSINTResult).filter(OSINTResult.media_id == media_id).delete()
        db.commit()

    # 2. Check initial status is Not Started
    resp = client.get(f"/api/osint/status/{media_id}")
    assert resp.status_code == 200
    status_data = resp.json()
    print(f"Initial Status Check: {status_data['status']}")
    assert status_data["status"] == "Not Started"
    assert status_data["tags"] == []

    # 3. Trigger the OSINT scan
    print("\nTriggering OSINT scan...")
    resp = client.post(f"/api/osint/scan/{media_id}")
    assert resp.status_code == 200
    scan_data = resp.json()
    print(f"Scan Trigger Response Status: {scan_data['status']}")
    assert scan_data["status"] in ["Pending", "Running"]

    # 4. Poll status until Completed
    print("\nPolling status until completed...")
    max_retries = 10
    completed = False
    for i in range(max_retries):
        time.sleep(1.0)
        resp = client.get(f"/api/osint/status/{media_id}")
        assert resp.status_code == 200
        status_data = resp.json()
        print(f"Retry {i+1}: Status = {status_data['status']}, Tags = {status_data.get('tags')}")
        if status_data["status"] in ["Completed", "Verified Matches Found", "No Matches Found", "Provider Unavailable"]:
            completed = True
            break
        elif status_data["status"] == "Failed":
            print(f"Scan failed: {status_data.get('error_message')}")
            break

    assert completed, "OSINT scan did not complete within timeout"

    # Verify tags were generated
    assert len(status_data["tags"]) > 0
    print(f"Generated Tags: {status_data['tags']}")

    # 5. Fetch OSINT Results
    print("\nRetrieving OSINT results...")
    resp = client.get(f"/api/osint/results/{media_id}")
    assert resp.status_code == 200
    results = resp.json()
    print(f"Retrieved {len(results)} search findings.")
    assert len(results) > 0

    # 6. Verify results schema & fields
    for res in results:
        assert "url" in res
        assert "title" in res
        assert "source" in res
        assert "confidence" in res
        assert "reason" in res
        
        # Verify confidence is a valid percentage
        assert 0 <= res["confidence"] <= 100
        
        # Verify confidence reason is populated
        assert res["reason"] is not None
        
        print(f"\nMatch Source: {res['source']}")
        print(f"  Title: {res['title']}")
        print(f"  URL: {res['url']}")
        print(f"  Confidence: {res['confidence']}%")
        print(f"  Reason: {res['reason']}")

    print("\n" + "=" * 60)
    print("ALL OSINT PIPELINE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_osint_flow()
