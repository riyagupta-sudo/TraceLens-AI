import os
import sys
import time
import psutil
from PIL import Image
import numpy as np

# Adjust path to import from app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) # MB

def main():
    print("=" * 50)
    print("CLIP READINESS OFFLINE BENCHMARK")
    print("=" * 50)
    
    # Check physical test image
    image_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "dataset", "case_intel_leak", "drone_orignal.jpg"
    )
    image_path = os.path.abspath(image_path)
    if not os.path.exists(image_path):
        print(f"Error: Test image not found at {image_path}")
        sys.exit(1)
        
    print(f"Test Image: {image_path}")
    print(f"Initial RAM Usage: {get_ram_usage():.2f} MB")
    
    # Measure imports and load
    print("\n[1/3] Loading torch and transformers (lazy)...")
    start_load = time.time()
    import torch
    from transformers import CLIPProcessor, CLIPModel
    end_load = time.time()
    print(f"Library load time: {end_load - start_load:.3f} seconds")
    print(f"RAM Usage: {get_ram_usage():.2f} MB")
    
    # Measure model init
    print("\n[2/3] Initializing openai/clip-vit-base-patch32...")
    start_init = time.time()
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.eval()
    end_init = time.time()
    print(f"Model initialization time: {end_init - start_init:.3f} seconds")
    print(f"RAM Usage: {get_ram_usage():.2f} MB")
    
    # Measure embedding generation
    print("\n[3/3] Generating semantic embedding...")
    start_infer = time.time()
    
    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")
        inputs = processor(images=rgb_img, return_tensors="pt")
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            emb = image_features.cpu().numpy()[0].tolist()
            
    end_infer = time.time()
    print(f"Inference time: {end_infer - start_infer:.3f} seconds")
    print(f"Embedding dimensions: {len(emb)}")
    print(f"Final RAM Usage: {get_ram_usage():.2f} MB")
    print("=" * 50)
    print("CLIP OFFLINE BENCHMARK COMPLETED")
    print("=" * 50)

if __name__ == "__main__":
    main()
