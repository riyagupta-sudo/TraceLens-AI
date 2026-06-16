import os
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Any

def generate_image_variants(original_path: str, output_dir: str) -> List[Dict[str, Any]]:
    """
    Generates 4 variations of an original image:
    1. Cropped: Center cropped by 30%
    2. Watermarked: Text 'TraceLens AI' superimposed
    3. Compressed: Highly compressed (JPEG 10% quality)
    4. Resized: Scaled down by 50%
    Returns a list of dictionaries with type, path, and filename.
    """
    variants = []
    
    if not os.path.exists(original_path):
        print(f"Original image not found: {original_path}")
        return variants
        
    os.makedirs(output_dir, exist_ok=True)
    base_name, ext = os.path.splitext(os.path.basename(original_path))
    
    try:
        # Load original image
        with Image.open(original_path) as img:
            w, h = img.size
            
            # --- 1. Crop Variant ---
            crop_name = f"{base_name}_variant_cropped.jpg"
            crop_path = os.path.join(output_dir, crop_name)
            left = int(w * 0.15)
            top = int(h * 0.15)
            right = int(w * 0.85)
            bottom = int(h * 0.85)
            cropped_img = img.crop((left, top, right, bottom))
            cropped_img.convert("RGB").save(crop_path, "JPEG", quality=90)
            variants.append({
                "type": "Cropped",
                "filepath": crop_path,
                "filename": crop_name,
                "relation": "cropped"
            })
            
            # --- 2. Watermark Variant ---
            watermark_name = f"{base_name}_variant_watermarked.jpg"
            watermark_path = os.path.join(output_dir, watermark_name)
            wm_img = img.copy().convert("RGB")
            draw = ImageDraw.Draw(wm_img)
            
            # Try to draw a large watermark box in the center
            text = "TRACELENS AI - SECURE"
            # Draw a dark backing rectangle for visibility
            box_w, box_h = int(w * 0.6), int(h * 0.1)
            box_x = int((w - box_w) / 2)
            box_y = int((h - box_h) / 2)
            draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], fill=(10, 10, 10, 150))
            
            # Draw white overlay text in backing rectangle
            # Using default font to ensure it works on any system without font files
            draw.text((box_x + 20, box_y + 10), text, fill=(0, 229, 255)) # neon cyan color
            wm_img.save(watermark_path, "JPEG", quality=90)
            variants.append({
                "type": "Watermarked",
                "filepath": watermark_path,
                "filename": watermark_name,
                "relation": "watermarked"
            })
            
            # --- 3. Compressed Variant ---
            compressed_name = f"{base_name}_variant_compressed.jpg"
            compressed_path = os.path.join(output_dir, compressed_name)
            img.copy().convert("RGB").save(compressed_path, "JPEG", quality=8) # 8% quality
            variants.append({
                "type": "Compressed",
                "filepath": compressed_path,
                "filename": compressed_name,
                "relation": "compressed"
            })
            
            # --- 4. Resized Variant ---
            resized_name = f"{base_name}_variant_resized.webp"
            resized_path = os.path.join(output_dir, resized_name)
            resized_img = img.resize((w // 2, h // 2), Image.Resampling.BILINEAR)
            resized_img.save(resized_path, "WEBP", quality=80)
            variants.append({
                "type": "Resized & Re-encoded",
                "filepath": resized_path,
                "filename": resized_name,
                "relation": "resized"
            })
            
    except Exception as e:
        print(f"Error generating image variants: {e}")
        
    return variants
