import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.database import SessionLocal
from app.models import MediaItem
from app.similarity_engine import hamming_distance, cosine_similarity

db = SessionLocal()
try:
    crop = db.query(MediaItem).filter(MediaItem.filename == "building_008_crop.jpg").order_by(MediaItem.id.desc()).first()
    orig = db.query(MediaItem).filter(MediaItem.filename == "building_008.jpg").order_by(MediaItem.id.desc()).first()
    
    if crop and orig:
        p_dist = hamming_distance(crop.phash, orig.phash)
        d_dist = hamming_distance(crop.dhash, orig.dhash)
        a_dist = hamming_distance(crop.ahash, orig.ahash)
        sem_sim = cosine_similarity(crop.embedding, orig.embedding)
        w1, h1 = crop.metadata_sig.get("width"), crop.metadata_sig.get("height")
        w2, h2 = orig.metadata_sig.get("width"), orig.metadata_sig.get("height")
        ar_diff = abs((w1 / h1) - (w2 / h2))
        
        print(f"p_dist: {p_dist}")
        print(f"d_dist: {d_dist}")
        print(f"a_dist: {a_dist}")
        print(f"sem_sim: {sem_sim}")
        print(f"ar_diff: {ar_diff}")
    else:
        print("Missing crop or original")
finally:
    db.close()
