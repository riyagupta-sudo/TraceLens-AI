import os
import sys
import pprint

# Add the backend root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.ai_editing_engine import detect_ai_editing

def run_tests():
    uploads_dir = os.path.join("app", "uploads")
    
    # Test images
    test_files = [
        "drone_telemetry_original.jpg",
        "drone_telemetry_original_variant_compressed.jpg",
        "satellite_recon_original.jpg",
        "crypto_leak_original.jpg"
    ]
    
    for filename in test_files:
        filepath = os.path.join(uploads_dir, filename)
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
            
        print("\n" + "="*60)
        print(f"Analyzing: {filepath}")
        print("="*60)
        
        result = detect_ai_editing(filepath)
        pprint.pprint(result, width=120)

if __name__ == "__main__":
    run_tests()
