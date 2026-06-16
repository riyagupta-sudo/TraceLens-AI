import requests
import json
import os

url = "http://127.0.0.1:8000/api/upload"
filepath = r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\case_intel_leak\birds.jpg"

if not os.path.exists(filepath):
    print(f"Error: File {filepath} not found.")
    exit(1)

files = {
    'file': (os.path.basename(filepath), open(filepath, 'rb'), 'image/jpeg')
}
# From inspect_live_db, we saw that Case ID 1 exists. We will use case_id = 1.
data = {
    'case_id': 1
}

print(f"Uploading {filepath} to {url}...")
try:
    response = requests.post(url, files=files, data=data)
    print("Status Code:", response.status_code)
    if response.status_code == 200:
        res_json = response.json()
        print("\n--- API RESPONSE JSON ---")
        print(json.dumps(res_json, indent=2))
        
        # Verify the ML prediction keys in the response
        report = res_json.get("modification_report", {})
        print("\nVerification of ML keys in JSON:")
        print("ml_tampering_probability:", report.get("ml_tampering_probability"))
        print("ml_classification:", report.get("ml_classification"))
        if "investigation_summary" in report:
            print("investigation_summary.ml_tampering_probability:", report["investigation_summary"].get("ml_tampering_probability"))
            print("investigation_summary.ml_classification:", report["investigation_summary"].get("ml_classification"))
    else:
        print("Failed to upload. Error:", response.text)
except Exception as e:
    print("Request failed:", e)
