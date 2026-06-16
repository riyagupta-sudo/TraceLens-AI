import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.database import SessionLocal
from app.models import MediaItem
from app.similarity_engine import analyze_matches, estimate_visual_containment, hamming_distance

def evaluate_pair(item_x, item_y):
    src_dna = {
        "phash": item_x.phash, "dhash": item_x.dhash, "ahash": item_x.ahash,
        "embedding": item_x.embedding, "audio_fingerprint": item_x.audio_fingerprint,
        "width": item_x.metadata_sig.get("width", 0) if item_x.metadata_sig else 0,
        "height": item_x.metadata_sig.get("height", 0) if item_x.metadata_sig else 0,
        "file_size": item_x.file_size, "mime_type": item_x.mime_type,
        "sha256": item_x.sha256,
        "filepath": os.path.join("app", "uploads", os.path.basename(item_x.filepath))
    }
    
    forensics = item_y.modification_report or {}
    tgt_dna = {
        "phash": item_y.phash, "dhash": item_y.dhash, "ahash": item_y.ahash,
        "embedding": item_y.embedding, "audio_fingerprint": item_y.audio_fingerprint,
        "width": item_y.metadata_sig.get("width", 0) if item_y.metadata_sig else 0,
        "height": item_y.metadata_sig.get("height", 0) if item_y.metadata_sig else 0,
        "file_size": item_y.file_size, "mime_type": item_y.mime_type,
        "modification_report": forensics,
        "sha256": item_y.sha256,
        "filepath": os.path.join("app", "uploads", os.path.basename(item_y.filepath))
    }
    
    combined, level, details = analyze_matches(src_dna, tgt_dna)
    
    sem_sim = details.get("semantic_similarity", 0.0)
    p_dist = hamming_distance(item_x.phash, item_y.phash)
    
    containment_res = estimate_visual_containment(src_dna["filepath"], tgt_dna["filepath"])
    contained_within = containment_res["contained_within_source"]
    overlap_pct = containment_res["visual_overlap_percent"]
    orb_confidence = containment_res.get("orb_confidence", 80.0 if (contained_within or overlap_pct > 0) else 0.0)
    containment_score = containment_res.get("containment_score", 100.0 if contained_within else 0.0)
    
    standard_passed = (
        combined >= 0.50 and
        sem_sim >= 0.75 and
        (contained_within or p_dist <= 12 or orb_confidence >= 30.0)
    )
    
    descendant_passed = (
        sem_sim >= 0.75 and
        overlap_pct >= 80.0 and
        orb_confidence >= 50.0
    )
    
    gate_passed = standard_passed or descendant_passed
    
    print(f"Comparison: {item_x.filename} (ID {item_x.id}) vs {item_y.filename} (ID {item_y.id})")
    print(f"  Combined Similarity: {combined:.4f}")
    print(f"  Semantic Similarity: {sem_sim:.4f}")
    print(f"  Hamming distance (pHash): {p_dist}")
    print(f"  Overlap: {overlap_pct:.2f}%")
    print(f"  ORB Confidence: {orb_confidence:.2f}%")
    print(f"  Containment Score: {containment_score:.2f}")
    print(f"  Contained within source: {contained_within}")
    print(f"  Admission decision: {'PASSED' if gate_passed else 'REJECTED'}")
    if not gate_passed:
        reasons = []
        if combined < 0.50: reasons.append(f"combined_similarity {combined:.2f} < 0.50")
        if sem_sim < 0.75: reasons.append(f"semantic_similarity {sem_sim:.2f} < 0.75")
        if not (contained_within or p_dist <= 12 or orb_confidence >= 30.0):
            reasons.append("failed containment checklist (not contained, pHash > 12, and ORB confidence < 30%)")
        if not descendant_passed:
            reasons.append("failed descendant admission path (semantic < 0.75 or overlap < 80% or ORB < 50%)")
        print(f"  Rejection Reason: {', '.join(reasons)}")

db = SessionLocal()
try:
    print("=== CASE 6 MEDIA ITEMS ===")
    items = db.query(MediaItem).filter(MediaItem.case_id == 6).all()
    for item in items:
        print(f"ID: {item.id} | Filename: {item.filename} | Cluster: {item.cluster_id} | Path: {item.filepath}")
        
    print("\n=== GATE EVALUATION IN CASE 6 ===")
    for item_x in items:
        for item_y in items:
            if item_x.id >= item_y.id:
                continue
            evaluate_pair(item_x, item_y)
finally:
    db.close()
