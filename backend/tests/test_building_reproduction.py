import requests
import os
import sys

def main():
    backend_url = "http://localhost:8000"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    # Paths to the building_008 images
    file_original_path = os.path.join(base_dir, "dataset", "originals", "building_008.jpg")
    file_resize_path = os.path.join(base_dir, "dataset", "resized", "building_008_resize.jpg")
    file_crop_path = os.path.join(base_dir, "dataset", "cropped", "building_008_crop.jpg")
    
    for p in [file_original_path, file_resize_path, file_crop_path]:
        if not os.path.exists(p):
            print(f"Error: Path does not exist: {p}")
            sys.exit(1)
            
    # Create a new Case to run the test in isolation
    print("Creating a new case for building_008 reproduction...")
    case_name = f"Building Reproduction {int(os.getpid())}"
    res_case = requests.post(f"{backend_url}/api/cases", json={"name": case_name, "description": "Reproducing bug with building_008"})
    if res_case.status_code != 200:
        print(f"Failed to create case: {res_case.text}")
        sys.exit(1)
    case_id = res_case.json()["id"]
    print(f"Created Case with ID: {case_id}")
    
    # Upload 1: building_008_resize.jpg
    print("\n--- Upload 1: building_008_resize.jpg ---")
    with open(file_resize_path, "rb") as f:
        files = {"file": ("building_008_resize.jpg", f, "image/jpeg")}
        res = requests.post(f"{backend_url}/api/upload", files=files, data={"case_id": case_id})
    if res.status_code != 200:
        print(f"Failed to upload resize: {res.text}")
        sys.exit(1)
    data_resize = res.json()
    print(f"Upload 1 Response: ID: {data_resize['id']} | Origin: {data_resize.get('estimated_origin_id')} | Integrity: {data_resize['integrity_score']} | Risk: {data_resize['risk_score']} | Class: {data_resize['modification_report'].get('asset_classification')}")
    
    # Fetch case media to verify DB state
    res_list = requests.get(f"{backend_url}/api/media", params={"case_id": case_id})
    print("Case media after Upload 1:")
    for item in res_list.json():
        print(f"  ID: {item['id']} | Filename: {item['filename']} | Origin: {item.get('estimated_origin_id')} | Parent: {item.get('parent_id')} | Integrity: {item['integrity_score']} | Risk: {item['risk_score']} | Class: {item['modification_report'].get('asset_classification') if item.get('modification_report') else 'None'}")
        
    # Upload 2: building_008_crop.jpg
    print("\n--- Upload 2: building_008_crop.jpg ---")
    with open(file_crop_path, "rb") as f:
        files = {"file": ("building_008_crop.jpg", f, "image/jpeg")}
        res = requests.post(f"{backend_url}/api/upload", files=files, data={"case_id": case_id})
    if res.status_code != 200:
        print(f"Failed to upload crop: {res.text}")
        sys.exit(1)
    data_crop = res.json()
    print(f"Upload 2 Response: ID: {data_crop['id']} | Origin: {data_crop.get('estimated_origin_id')} | Integrity: {data_crop['integrity_score']} | Risk: {data_crop['risk_score']} | Class: {data_crop['modification_report'].get('asset_classification')}")
    
    # Fetch case media to verify DB state
    res_list = requests.get(f"{backend_url}/api/media", params={"case_id": case_id})
    print("Case media after Upload 2:")
    for item in res_list.json():
        print(f"  ID: {item['id']} | Filename: {item['filename']} | Origin: {item.get('estimated_origin_id')} | Parent: {item.get('parent_id')} | Integrity: {item['integrity_score']} | Risk: {item['risk_score']} | Class: {item['modification_report'].get('asset_classification') if item.get('modification_report') else 'None'}")

    # Upload 3: building_008.jpg (original)
    print("\n--- Upload 3: building_008.jpg (original) ---")
    with open(file_original_path, "rb") as f:
        files = {"file": ("building_008.jpg", f, "image/jpeg")}
        res = requests.post(f"{backend_url}/api/upload", files=files, data={"case_id": case_id})
    if res.status_code != 200:
        print(f"Failed to upload original: {res.text}")
        sys.exit(1)
    data_original = res.json()
    print(f"Upload 3 Response: ID: {data_original['id']} | Origin: {data_original.get('estimated_origin_id')} | Integrity: {data_original['integrity_score']} | Risk: {data_original['risk_score']} | Class: {data_original['modification_report'].get('asset_classification')}")
    
    # Fetch case media to verify DB state
    res_list = requests.get(f"{backend_url}/api/media", params={"case_id": case_id})
    print("Case media after Upload 3 (via /api/media):")
    for item in res_list.json():
        print(f"  ID: {item['id']} | Filename: {item['filename']} | Origin: {item.get('estimated_origin_id')} | Parent: {item.get('parent_id')} | Integrity: {item['integrity_score']} | Risk: {item['risk_score']} | Class: {item['modification_report'].get('asset_classification') if item.get('modification_report') else 'None'}")

    # Fetch via /api/media/{id} for each to be extra sure
    print("\nFetching individually via /api/media/{id} to double-check persistence:")
    for item in res_list.json():
        r_detail = requests.get(f"{backend_url}/api/media/{item['id']}")
        detail = r_detail.json()
        print(f"  ID: {detail['id']} | Filename: {detail['filename']} | Origin: {detail.get('estimated_origin_id')} | Parent: {detail.get('parent_id')} | Integrity: {detail['integrity_score']} | Risk: {detail['risk_score']} | Class: {detail['modification_report'].get('asset_classification') if detail.get('modification_report') else 'None'}")

if __name__ == "__main__":
    main()
