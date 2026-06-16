import requests
import os
import sys

def main():
    backend_url = "http://localhost:8000"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_dir = os.path.join(base_dir, "dataset", "case_intel_leak")
    
    file_large_path = os.path.join(dataset_dir, "WhatsApp Image 2026-06-08 at 3.36.27 PM (1).jpeg")
    file_small_path = os.path.join(dataset_dir, "WhatsApp Image 2026-06-08 at 3.36.27 PM.jpeg")
    
    if not os.path.exists(file_large_path) or not os.path.exists(file_small_path):
        print("Error: WhatsApp test images do not exist.")
        sys.exit(1)
        
    # First, let's upload the smaller one (simulating upload of variant first)
    print("Uploading small WhatsApp image (simulating variant first)...")
    with open(file_small_path, "rb") as f:
        files = {"file": ("whatsapp_small.jpg", f, "image/jpeg")}
        data = {"case_id": 1}
        res1 = requests.post(f"{backend_url}/api/upload", files=files, data=data)
        
    if res1.status_code != 200:
        print(f"Failed to upload small image: {res1.text}")
        sys.exit(1)
        
    data1 = res1.json()
    id1 = data1["id"]
    print(f"Uploaded successfully. ID: {id1} | Integrity: {data1['integrity_score']} | Risk: {data1['risk_score']} | Class: {data1['modification_report'].get('asset_classification')}")
    
    # Second, upload the larger one (the original/baseline asset)
    print("\nUploading large WhatsApp image (the original)...")
    with open(file_large_path, "rb") as f:
        files = {"file": ("whatsapp_large.jpg", f, "image/jpeg")}
        data = {"case_id": 1}
        res2 = requests.post(f"{backend_url}/api/upload", files=files, data=data)
        
    if res2.status_code != 200:
        print(f"Failed to upload large image: {res2.text}")
        sys.exit(1)
        
    data2 = res2.json()
    id2 = data2["id"]
    print(f"Uploaded successfully. ID: {id2} | Integrity: {data2['integrity_score']} | Risk: {data2['risk_score']} | Class: {data2['modification_report'].get('asset_classification')}")
    
    # Retrieve details of the first upload to verify it was re-evaluated and corrected!
    print("\nFetching details of the small image after the original was uploaded...")
    res1_updated = requests.get(f"{backend_url}/api/media/{id1}")
    if res1_updated.status_code == 200:
        data1_updated = res1_updated.json()
        print(f"Small Image (ID: {id1}):")
        print(f"  Estimated Origin ID: {data1_updated['estimated_origin_id']}")
        print(f"  Parent ID:           {data1_updated['parent_id']}")
        print(f"  Integrity Score:     {data1_updated['integrity_score']}")
        print(f"  Risk Score:          {data1_updated['risk_score']}")
        print(f"  Classification:      {data1_updated['modification_report'].get('asset_classification')}")
    else:
        print(f"Failed to fetch updated small image: {res1_updated.text}")
        
    # Retrieve details of the second upload to verify it is selected as the origin and clean!
    print("\nFetching details of the large image...")
    res2_updated = requests.get(f"{backend_url}/api/media/{id2}")
    if res2_updated.status_code == 200:
        data2_updated = res2_updated.json()
        print(f"Large Image (ID: {id2}):")
        print(f"  Estimated Origin ID: {data2_updated['estimated_origin_id']}")
        print(f"  Parent ID:           {data2_updated['parent_id']}")
        print(f"  Integrity Score:     {data2_updated['integrity_score']}")
        print(f"  Risk Score:          {data2_updated['risk_score']}")
        print(f"  Classification:      {data2_updated['modification_report'].get('asset_classification')}")
    else:
        print(f"Failed to fetch updated large image: {res2_updated.text}")

if __name__ == "__main__":
    main()
