import os
import sys
from fastapi.testclient import TestClient

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.database import SessionLocal
from app.models import MediaItem, Case

def test_media_profile_persistence():
    print("======================================================================")
    print("STARTING MEDIA PROFILE PERSISTENCE REGRESSION TEST")
    print("======================================================================\n")
    
    client = TestClient(app)
    db = SessionLocal()
    
    try:
        # 1. Create a Case
        res = client.post("/api/cases", json={"name": "Persistence Regression Case", "description": "Verification Case"})
        assert res.status_code == 200, "Failed to create case"
        case_id = res.json()["id"]
        print(f"Created case with ID: {case_id}")
        
        # 2. Upload a test image
        dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset"))
        test_file = os.path.join(dataset_dir, "originals", "human_006.jpg")
        assert os.path.exists(test_file), f"Test file missing: {test_file}"
        
        with open(test_file, "rb") as f:
            upload_res = client.post(
                "/api/upload",
                data={"case_id": case_id},
                files={"file": ("human_006.jpg", f, "image/jpeg")}
            )
        assert upload_res.status_code == 200, f"Upload failed: {upload_res.text}"
        media_id = upload_res.json()["id"]
        print(f"Uploaded human_006.jpg -> ID: {media_id}")
        
        # 3. Verify Media Record exists in DB
        db_item = db.query(MediaItem).filter(MediaItem.id == media_id).first()
        assert db_item is not None, "Media record does not exist in the database!"
        print("[PASSED] Media record exists in SQLite database.")
        
        # 4. Verify all endpoint status codes
        endpoints = [
            (f"/api/media/{media_id}", 200, "Main Detail"),
            (f"/api/media/{media_id}/phash-steps", 200, "pHash Steps"),
            (f"/api/media/{media_id}/relationship-graph", 200, "Relationship Graph"),
            (f"/api/media/{media_id}/similar", 200, "Similar Assets")
        ]
        
        for url, expected_status, name in endpoints:
            res = client.get(url)
            assert res.status_code == expected_status, f"{name} endpoint failed with status {res.status_code}: {res.text}"
            print(f"[PASSED] {name} endpoint returned status {expected_status}.")
            
            # Additional structural verification
            if "relationship-graph" in url:
                graph_data = res.json()
                assert "nodes" in graph_data, "Graph data missing 'nodes'"
                assert "links" in graph_data, "Graph data missing 'links'"
                assert "timeline_confidence" in graph_data, "Graph data missing 'timeline_confidence'"
                print(f"         Timeline confidence: {graph_data['timeline_confidence']}")
                
        # 5. Verify Graph Endpoint Fault Tolerance (Defensive fallback test)
        # We query the relationship graph of an invalid ID to trigger the exception block (since target is None, it returns 404, wait).
        # Let's test what happens if an unhandled error inside get_relationship_graph occurs by mocking database session to raise an error
        # during query execution.
        print("\nTesting defensive exception handling in relationship-graph endpoint...")
        # Trigger exception by passing an invalid media_id that causes an error, or we can just mock it.
        # Actually, let's pass a string or cause a type error/exception in the query.
        # In get_relationship_graph, target = db.query(MediaItem).filter(MediaItem.id == media_id).first()
        # If media_id is a string that cannot be cast, SQLite might handle it, but what if we pass None or mock DB?
        # Let's mock DB session or target.
        # We can test by calling `/api/media/invalid_id/relationship-graph`.
        # Since media_id is annotated as `int`, FastAPI will return 422 validation error for string, which is standard.
        # But we can verify that the fallback payload format is correct if we call it with a mock or if an error is handled.
        # In app.main, if db raises an error, the try-except Exception will catch it and return the fallback.
        # Let's test this by calling get_relationship_graph manually with a mock session that throws an exception:
        from app.main import get_relationship_graph
        class MockSession:
            def query(self, *args, **kwargs):
                raise Exception("Mock Database Failure")
                
        fallback_res = get_relationship_graph(media_id=media_id, db=MockSession())
        assert fallback_res["error"] == "graph_generation_failed", "Fallback did not return error flag"
        assert fallback_res["nodes"] == [], "Fallback did not return empty nodes"
        assert fallback_res["links"] == [], "Fallback did not return empty links"
        assert fallback_res["edges"] == [], "Fallback did not return empty edges"
        assert fallback_res["timeline_confidence"] == 0, "Fallback did not return 0 timeline confidence"
        print("[PASSED] Fault tolerance fallback payload successfully verified!")
        
        print("\nSUCCESS: test_media_profile_persistence passed cleanly!")
        
    finally:
        db.close()

if __name__ == "__main__":
    test_media_profile_persistence()
