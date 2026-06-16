import requests
import os
import sys

def main():
    backend_url = "http://localhost:8000"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_dir = os.path.join(base_dir, "dataset", "case_intel_leak")
    
    # Files to upload
    orig_path = os.path.join(dataset_dir, "moon_original.jpg")
    comp_path = os.path.join(dataset_dir, "moon_compressed.jpg")
    
    if not os.path.exists(orig_path) or not os.path.exists(comp_path):
        print("Error: moon test images do not exist. Run test_forensics_verify.py first!")
        sys.exit(1)
        
    print("Uploading moon_original.jpg to Case 1...")
    with open(orig_path, "rb") as f:
        files = {"file": ("moon_original.jpg", f, "image/jpeg")}
        data = {"case_id": 1}
        res_orig = requests.post(f"{backend_url}/api/upload", files=files, data=data)
        
    if res_orig.status_code != 200:
        print(f"Failed to upload original image: {res_orig.text}")
        sys.exit(1)
        
    orig_data = res_orig.json()
    orig_id = orig_data["id"]
    print(f"Uploaded successfully. ID: {orig_id} | Integrity: {orig_data['integrity_score']} | Risk: {orig_data['risk_score']}")
    
    print("\nUploading moon_compressed.jpg with parent_id...")
    with open(comp_path, "rb") as f:
        files = {"file": ("moon_compressed.jpg", f, "image/jpeg")}
        data = {"case_id": 1, "parent_id": orig_id}
        res_comp = requests.post(f"{backend_url}/api/upload", files=files, data=data)
        
    if res_comp.status_code != 200:
        print(f"Failed to upload compressed image: {res_comp.text}")
        sys.exit(1)
        
    comp_data = res_comp.json()
    comp_id = comp_data["id"]
    print(f"Uploaded successfully. ID: {comp_id} | Integrity: {comp_data['integrity_score']} | Risk: {comp_data['risk_score']}")
    
    # Assert new probabilistic origin fields are present
    assert "modification_report" in comp_data, "Missing modification_report in response"
    report = comp_data["modification_report"]
    assert "relationship_analysis" in report, "Missing relationship_analysis in report"
    rel = report["relationship_analysis"]
    assert "origin_confidence" in rel, "Missing origin_confidence in relationship_analysis"
    assert "origin_probability" in rel, "Missing origin_probability"
    assert "origin_undetermined" in rel, "Missing origin_undetermined"
    print("Verification of probabilistic origin fields in API response: PASSED")
    
    # 3. Retrieve and print relationship graph for moon_original.jpg
    print("\nFetching relationship graph for moon_original.jpg...")
    res_graph = requests.get(f"{backend_url}/api/media/{orig_id}/relationship-graph")
    if res_graph.status_code == 200:
        graph_data = res_graph.json()
        print("Graph Nodes:")
        for node in graph_data.get("nodes", []):
            print(f"  Node ID: {node['id']} | Label: {node['label']} | Type: {node['type']} | Integrity: {node['integrity']} | Risk: {node['risk']}")
        print("Graph Links:")
        for link in graph_data.get("links", []):
            print(f"  Link: {link['source']} -> {link['target']} | Score: {link['score']} | Type: {link['type']}")
    else:
        print(f"Failed to fetch relationship graph: {res_graph.text}")
        sys.exit(1)

if __name__ == "__main__":
    main()
