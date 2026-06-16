import os
import subprocess
import cv2
import imagehash
from PIL import Image
from typing import List, Dict, Any, Tuple
from .dna_engine import compute_audio_fingerprint

def analyze_video(
    video_path: str, 
    keyframes_dir: str, 
    uploads_dir: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    Analyzes a video file:
    1. Extracts keyframes every 2 seconds.
    2. Computes pHash for each keyframe.
    3. Extracts audio as WAV and computes fingerprint.
    4. Returns keyframes list, audio fingerprint, and video metadata.
    """
    keyframes = []
    audio_fingerprint = {"has_audio": False, "mean_chroma": [], "temporal_profile": []}
    video_metadata = {}
    
    # Verify paths exist
    os.makedirs(keyframes_dir, exist_ok=True)
    os.makedirs(uploads_dir, exist_ok=True)
    
    # 1. Read Video Properties using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return keyframes, audio_fingerprint, video_metadata
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    video_metadata = {
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "frame_count": frame_count,
        "resolution": f"{width}x{height}",
        "format": os.path.splitext(video_path)[1].upper()[1:]
    }
    
    # 2. Extract keyframes every 2 seconds
    interval_seconds = 2.0
    frame_interval = int(fps * interval_seconds) if fps > 0 else 60
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % frame_interval == 0:
            timestamp = count / fps if fps > 0 else 0.0
            
            # Save keyframe image
            keyframe_name = f"frame_{os.path.basename(video_path)}_{int(timestamp)}.jpg"
            keyframe_path = os.path.join(keyframes_dir, keyframe_name)
            
            # Convert OpenCV frame (BGR) to RGB and save via Pillow for consistent pHash
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            pil_img.save(keyframe_path, "JPEG", quality=85)
            
            # Compute pHash of this keyframe
            ph = str(imagehash.phash(pil_img))
            
            keyframes.append({
                "timestamp": timestamp,
                "filepath": f"/media/keyframes/{keyframe_name}",  # Web serving path
                "phash": ph
            })
            
        count += 1
    cap.release()
    
    # 3. Extract Audio via ffmpeg
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    wav_path = os.path.join(uploads_dir, f"{base_name}_temp.wav")
    
    # Run ffmpeg as a subprocess
    try:
        command = [
            "ffmpeg", "-y", "-i", video_path, 
            "-vn", "-acodec", "pcm_s16le", 
            "-ar", "11025", "-ac", "1", wav_path
        ]
        # Redirect stderr to devnull to avoid cluttering logs unless debugging
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=30)
        
        # If WAV was generated, compute fingerprint
        if os.path.exists(wav_path):
            audio_fingerprint = compute_audio_fingerprint(wav_path)
            # Remove temp WAV to save disk space
            try:
                os.remove(wav_path)
            except Exception as ex:
                print(f"Error removing temp WAV: {ex}")
    except subprocess.TimeoutExpired:
        print("FFmpeg audio extraction timed out.")
    except Exception as e:
        print(f"FFmpeg not found or audio extraction failed: {e}. Audio processing skipped.")
        
    return keyframes, audio_fingerprint, video_metadata
