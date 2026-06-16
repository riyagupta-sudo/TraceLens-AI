import io
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import scipy.fftpack
from typing import Dict, Any

def get_p_hash_steps(image_path: str) -> Dict[str, Any]:
    """
    Computes pHash and returns a step-by-step visualization of the process.
    Steps:
      1. Original Image (resized for display)
      2. Grayscale Conversion
      3. Resize to 32x32
      4. 2D Discrete Cosine Transform (DCT)
      5. Low Frequency extraction (8x8 grid)
      6. Hash Generation (bit matrix)
    """
    steps = {}
    
    try:
        # Load original image
        orig_img = Image.open(image_path)
        
        # Step 1: Original (keep aspect ratio, max 300x300)
        orig_img.thumbnail((300, 300))
        steps["step1"] = image_to_base64(orig_img)
        
        # Step 2: Grayscale
        gray_img = orig_img.convert("L")
        steps["step2"] = image_to_base64(gray_img)
        
        # Step 3: Resize to 32x32
        # Use a full-size image to calculate the actual pHash grid
        actual_img = Image.open(image_path).convert("L").resize((32, 32), Image.Resampling.BILINEAR)
        # Scale it up for visual display
        scaled_32x32 = actual_img.resize((256, 256), Image.Resampling.NEAREST)
        steps["step3"] = image_to_base64(scaled_32x32)
        
        # Step 4: 2D DCT
        # Convert image to numpy array
        pixels = np.array(actual_img, dtype=float)
        # Perform 2D DCT
        dct_data = scipy.fftpack.dct(scipy.fftpack.dct(pixels.T, norm='ortho').T, norm='ortho')
        
        # Visual DCT: Log transform and normalize to 0-255 for display
        dct_abs = np.abs(dct_data)
        # Avoid log(0)
        dct_log = np.log(dct_abs + 1e-5)
        dct_min, dct_max = dct_log.min(), dct_log.max()
        if dct_max > dct_min:
            dct_norm = ((dct_log - dct_min) / (dct_max - dct_min) * 255).astype(np.uint8)
        else:
            dct_norm = np.zeros_like(pixels, dtype=np.uint8)
            
        dct_img = Image.fromarray(dct_norm)
        scaled_dct = dct_img.resize((256, 256), Image.Resampling.NEAREST)
        steps["step4"] = image_to_base64(scaled_dct)
        
        # Step 5: Extract 8x8 Low Frequency
        # The top-left 8x8 contains the lowest frequencies
        dct_8x8 = dct_data[:8, :8]
        # Normalize just the 8x8 block for visualization
        dct_8x8_abs = np.abs(dct_8x8)
        dct_8x8_log = np.log(dct_8x8_abs + 1e-5)
        d8_min, d8_max = dct_8x8_log.min(), dct_8x8_log.max()
        if d8_max > d8_min:
            d8_norm = ((dct_8x8_log - d8_min) / (d8_max - d8_min) * 255).astype(np.uint8)
        else:
            d8_norm = np.zeros((8, 8), dtype=np.uint8)
            
        d8_img = Image.fromarray(d8_norm).resize((256, 256), Image.Resampling.NEAREST)
        steps["step5"] = image_to_base64(d8_img)
        
        # Step 6: Hash Generation
        # Standard pHash: compare values to the median, excluding the DC term (0, 0)
        flat_8x8 = dct_8x8.flatten()
        median_val = np.median(flat_8x8)
        
        # Binary grid
        bits = flat_8x8 > median_val
        bit_matrix = bits.reshape((8, 8))
        
        # Render a visualization of the binary matrix (green blocks for 1, dark red for 0)
        block_size = 32
        hash_visual = Image.new("RGB", (256, 256), color=(20, 20, 20))
        draw = ImageDraw.Draw(hash_visual)
        
        for r in range(8):
            for c in range(8):
                val = bit_matrix[r, c]
                color = (0, 229, 255) if val else (124, 58, 237) # neon cyan for 1, violet for 0
                x1 = c * block_size
                y1 = r * block_size
                x2 = x1 + block_size - 2
                y2 = y1 + block_size - 2
                draw.rectangle([x1, y1, x2, y2], fill=color)
                
        steps["step6"] = image_to_base64(hash_visual)
        
        # Calculate hex string
        hex_str = ""
        for i in range(0, 64, 4):
            nibble = bits[i:i+4]
            val = sum([b * (2**(3-j)) for j, b in enumerate(nibble)])
            hex_str += hex(val)[2:]
            
        steps["hash"] = hex_str
        
    except Exception as e:
        print(f"Error in pHash visualization generator: {e}")
        steps["error"] = str(e)
        
    return steps


def image_to_base64(img: Image.Image) -> str:
    """Helper to convert a PIL image to base64 jpeg string."""
    buffered = io.BytesIO()
    # Convert RGBA or other unsupported modes to RGB for JPEG compatibility
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(buffered, format="JPEG", quality=90)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"
