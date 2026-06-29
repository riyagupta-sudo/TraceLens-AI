import os
import json
from PIL import Image

project_root = r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI"

def main():
    exclude_dirs = [
        ".git",
        "venv",
        "node_modules",
        "__pycache__",
        "ai_detection",
        "ai_detection_v2",
        "validation_pack"
    ]
    
    print("Searching the entire workspace for AI datasets and metadata...")
    
    ai_candidates = []
    metadata_files = []
    
    for root, dirs, files in os.walk(project_root):
        # Exclude directories inline
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for f in files:
            fl = f.lower()
            filepath = os.path.join(root, f)
            
            # Check for AI image matches
            if fl.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                if any(x in fl for x in ['midjourney', 'flux', 'sdxl', 'chatgpt', 'dalle', 'dall-e', 'stable_diffusion', 'generated']):
                    ai_candidates.append(filepath)
            
            # Check for prompt files, sidecars, metadata json, etc.
            elif fl.endswith(('.json', '.txt', '.yaml', '.csv', '.tsv')):
                if any(x in fl for x in ['prompt', 'generation', 'metadata', 'sidecar', 'manifest']):
                    if "node_modules" not in filepath and "venv" not in filepath:
                        metadata_files.append(filepath)
                        
    print(f"Search completed.")
    print(f"Total AI image candidates found: {len(ai_candidates)}")
    print(f"Total metadata/prompt files found: {len(metadata_files)}")
    
    if ai_candidates:
        print("Sample image paths:")
        for x in ai_candidates[:5]:
            print(f"  {x}")
    if metadata_files:
        print("Sample metadata/prompt file paths:")
        for x in metadata_files[:5]:
            print(f"  {x}")

if __name__ == "__main__":
    main()
