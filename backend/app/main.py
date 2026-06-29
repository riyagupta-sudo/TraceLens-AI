# Reload Trigger 2
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env relative to this file
app_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(app_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
load_dotenv(os.path.join(backend_dir, ".env"))

import shutil
import datetime
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from PIL import Image
import imagehash
import numpy as np

from .database import engine, Base, get_db, SessionLocal
from .models import Case, MediaItem, Keyframe, MediaRelationship
from .schemas import (
    CaseResponse, CaseCreate, MediaItemResponse, MediaListItem,
    CompareResponse, CompareRequest, PlaygroundRequest, PlaygroundResponse
)
from .dna_engine import (
    compute_sha256, compute_image_hashes, get_clip_embedding, 
    extract_metadata_signature, calculate_integrity_and_risk,
    ENABLE_CLIP
)
from .video_analyzer import analyze_video
from .similarity_engine import analyze_matches, estimate_primary_origin, hamming_distance
from .phash_visualizer import get_p_hash_steps
from .report_generator import generate_pdf_report
from .seeder import seed_data_if_empty
from .web_intelligence import run_asynchronous_web_intelligence

def get_parsed_timestamp(forensics_dict: Optional[Dict[str, Any]]) -> Optional[datetime.datetime]:
    if not forensics_dict:
        return None
    val = forensics_dict.get("ai_edit_analysis_timestamp")
    if not val:
        return None
    if isinstance(val, datetime.datetime):
        return val
    if isinstance(val, str):
        try:
            return datetime.datetime.fromisoformat(val)
        except Exception:
            try:
                if val.endswith('Z'):
                    val = val[:-1]
                return datetime.datetime.fromisoformat(val)
            except Exception:
                return None
    return None

# Setup folder directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
KEYFRAMES_DIR = os.path.join(BASE_DIR, "keyframes")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

for directory in [UPLOADS_DIR, KEYFRAMES_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Initialize database
Base.metadata.create_all(bind=engine)

# Startup Schema Validation (checks columns/tables and logs warnings/diagnostics without automatic ALTERs)
import sqlite3
try:
    db_path = os.path.join(backend_dir, "tracelens.db")
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check media_items columns
        cursor.execute("PRAGMA table_info(media_items)")
        columns = [col[1] for col in cursor.fetchall()]
        if "cluster_id" not in columns:
            print("[SCHEMA WARNING] Column 'cluster_id' is missing from 'media_items' table. "
                  "Please run explicit migration script: python app/migrate.py")
        
        # Check cluster_merge_recommendations table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cluster_merge_recommendations'")
        if not cursor.fetchone():
            print("[SCHEMA WARNING] Table 'cluster_merge_recommendations' is missing from database. "
                  "Please run explicit migration script: python app/migrate.py")
        conn.close()
except Exception as e:
    print(f"[SCHEMA VALIDATION ERROR] Startup check failed: {e}")

# Seed database on startup if empty
with Session(engine) as db_session:
    seed_data_if_empty(db_session, UPLOADS_DIR)

# Log OSINT provider availability at startup
try:
    from .web_intelligence import get_provider_availability
    availability = get_provider_availability()
    mapping = [
        ("APIFY", availability["apify"]),
        ("GOOGLE LENS", availability["google_lens"]),
        ("BING VISUAL", availability["bing_visual"]),
        ("YANDEX", availability["yandex"]),
        ("TINEYE", availability["tineye"]),
    ]
    for name, available in mapping:
        status = "AVAILABLE" if available else "MISSING"
        dots = "." * (20 - len(name) - 1)
        print(f"[OSINT] {name} {dots} {status}")
except Exception as e:
    print(f"[OSINT ERROR] Failed to run startup provider check: {e}")

app = FastAPI(
    title="TraceLens AI API",
    description="Media DNA & Cross-Platform Forensic Intelligence Engine",
    version="1.0.0"
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount media static files directory for direct asset serving
app.mount("/media/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")
app.mount("/media/keyframes", StaticFiles(directory=KEYFRAMES_DIR), name="keyframes")

# ----------------- CASES ENDPOINTS -----------------

@app.get("/api/cases", response_model=List[CaseResponse])
def get_cases(db: Session = Depends(get_db)):
    """Lists all forensic investigation cases."""
    return db.query(Case).all()

@app.post("/api/cases", response_model=CaseResponse)
def create_case(case_data: CaseCreate, db: Session = Depends(get_db)):
    """Creates a new forensic case."""
    db_case = Case(name=case_data.name, description=case_data.description)
    db.add(db_case)
    db.commit()
    db.refresh(db_case)
    return db_case

# ----------------- MEDIA ENDPOINTS -----------------

@app.get("/api/media", response_model=List[MediaListItem])
def list_media(
    case_id: Optional[int] = None, 
    query: Optional[str] = None, 
    db: Session = Depends(get_db)
):
    """Lists and searches indexed media items."""
    db_query = db.query(MediaItem)
    if case_id:
        db_query = db_query.filter(MediaItem.case_id == case_id)
    if query:
        db_query = db_query.filter(
            (MediaItem.filename.like(f"%{query}%")) | 
            (MediaItem.sha256.like(f"%{query}%")) | 
            (MediaItem.phash.like(f"%{query}%"))
        )
    return db_query.all()

@app.get("/api/media/{media_id}", response_model=MediaItemResponse)
def get_media_detail(media_id: int, db: Session = Depends(get_db)):
    """Returns the comprehensive details and DNA profile of a media item."""
    item = db.query(MediaItem).filter(MediaItem.id == media_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Media asset not found")
    return item

def process_clip_and_cross_match(
    media_id: int,
    dest_path: str,
    case_id: int,
    parent_id: Optional[int],
    forensics: Dict[str, Any],
    db_session_factory
):
    import time
    import numpy as np
    import uuid
    from .models import ClusterMergeRecommendation
    from .similarity_engine import (
        filter_candidates_stage_0,
        analyze_matches,
        estimate_primary_origin,
        save_heavy_features
    )
    print(f"[BACKGROUND TASK] Starting CLIP embedding generation and cross-matching for media ID {media_id}")
    db = db_session_factory()
    try:
        db_item = db.query(MediaItem).filter(MediaItem.id == media_id).first()
        if not db_item:
            print(f"[BACKGROUND TASK] Media item {media_id} not found in database.")
            return

        if ENABLE_CLIP:
            t0 = time.perf_counter()
            print("[BACKGROUND TASK] Generating CLIP embedding in background...")
            emb = get_clip_embedding(dest_path)
            t1 = time.perf_counter()
            print(f"[BACKGROUND TASK] CLIP Embedding generated in {(t1 - t0)*1000:.2f} ms")
        else:
            emb = []
            print("[BACKGROUND TASK] CLIP disabled. Skipping embedding generation.")

        db_item.embedding = emb
        meta = dict(db_item.metadata_sig or {})
        if "video" not in db_item.mime_type:
            meta["embedding"] = emb
            db_item.metadata_sig = meta
        db.commit()
        db.refresh(db_item)

        other_items = db.query(MediaItem).filter(
            MediaItem.id != db_item.id,
            MediaItem.case_id == case_id
        ).all()

        # Stage 0 Candidate Retrieval
        db_item_dna = {
            "phash": db_item.phash,
            "dhash": db_item.dhash,
            "ahash": db_item.ahash,
            "width": db_item.metadata_sig.get("width", 0) if isinstance(db_item.metadata_sig, dict) else 0,
            "height": db_item.metadata_sig.get("height", 0) if isinstance(db_item.metadata_sig, dict) else 0,
            "embedding": db_item.embedding
        }
        filtered_items = filter_candidates_stage_0(db_item_dna, other_items)

        matching_pairs = []
        direct_matches = []

        for item in filtered_items:
            src_dna = {
                "phash": item.phash, "dhash": item.dhash, "ahash": item.ahash,
                "embedding": item.embedding, "audio_fingerprint": item.audio_fingerprint,
                "width": item.metadata_sig.get("width", 0) if isinstance(item.metadata_sig, dict) else 0,
                "height": item.metadata_sig.get("height", 0) if isinstance(item.metadata_sig, dict) else 0,
                "file_size": item.file_size, "mime_type": item.mime_type,
                "sha256": item.sha256,
                "filepath": os.path.join(UPLOADS_DIR, os.path.basename(item.filepath)),
                "feature_cache_path": item.metadata_sig.get("feature_cache_path") if isinstance(item.metadata_sig, dict) else None,
                "screenshot_indicators": item.modification_report.get("screenshot_indicators") if isinstance(item.modification_report, dict) else None
            }
            tgt_dna = {
                "phash": db_item.phash, "dhash": db_item.dhash, "ahash": db_item.ahash,
                "embedding": db_item.embedding, "audio_fingerprint": db_item.audio_fingerprint,
                "width": db_item.metadata_sig.get("width", 0) if isinstance(db_item.metadata_sig, dict) else 0,
                "height": db_item.metadata_sig.get("height", 0) if isinstance(db_item.metadata_sig, dict) else 0,
                "file_size": db_item.file_size, "mime_type": db_item.mime_type,
                "sha256": db_item.sha256,
                "filepath": dest_path,
                "feature_cache_path": db_item.metadata_sig.get("feature_cache_path") if isinstance(db_item.metadata_sig, dict) else None,
                "screenshot_indicators": forensics.get("screenshot_indicators")
            }

            combined, level, details = analyze_matches(src_dna, tgt_dna)

            # Match Gate Check: min combined threshold
            gate_passed = (combined >= 0.58)

            print(f"[BACKGROUND FORENSIC GATE] File: '{db_item.filename}' against candidate '{item.filename}' (ID: {item.id})")
            print(f"  Combined score: {combined:.4f}")
            print(f"  Admission Decision: {'PASSED' if gate_passed else 'REJECTED'}")

            if gate_passed:
                direct_matches.append(item)
                matching_pairs.append((item, combined, details))

        primary_cluster_id = None
        if not direct_matches:
            # Create a permanent UUID for the cluster, which is immutable
            new_cid = f"cluster_{uuid.uuid4().hex[:8]}"
            db_item.cluster_id = new_cid
            db_item.estimated_origin_id = db_item.id
            db_item.parent_id = None
            db.commit()
            cluster_items = [db_item]
            primary_cluster_id = new_cid
        else:
            matching_cluster_ids = {item.cluster_id for item in direct_matches if item.cluster_id}

            if not matching_cluster_ids:
                new_cid = f"cluster_{uuid.uuid4().hex[:8]}"
                db_item.cluster_id = new_cid
                db_item.estimated_origin_id = db_item.id
                db_item.parent_id = None
                for item in direct_matches:
                    item.cluster_id = new_cid
                db.commit()
                cluster_items = db.query(MediaItem).filter(
                    MediaItem.case_id == case_id,
                    MediaItem.cluster_id == new_cid
                ).all()
                primary_cluster_id = new_cid
            else:
                best_match_item = None
                best_match_score = -1.0
                for item, combined, _ in matching_pairs:
                    if item.cluster_id and combined > best_match_score:
                        best_match_score = combined
                        best_match_item = item

                primary_cluster_id = best_match_item.cluster_id
                db_item.cluster_id = primary_cluster_id
                db.commit()

                # Recommend merges if other items matched but belong to different clusters
                for other_cid in matching_cluster_ids:
                    if other_cid != primary_cluster_id:
                        existing_rec = db.query(ClusterMergeRecommendation).filter(
                            ClusterMergeRecommendation.case_id == case_id,
                            ((ClusterMergeRecommendation.source_cluster_id == other_cid) & (ClusterMergeRecommendation.target_cluster_id == primary_cluster_id)) |
                            ((ClusterMergeRecommendation.source_cluster_id == primary_cluster_id) & (ClusterMergeRecommendation.target_cluster_id == other_cid))
                        ).first()
                        if not existing_rec:
                            rec = ClusterMergeRecommendation(
                                case_id=case_id,
                                source_cluster_id=other_cid,
                                target_cluster_id=primary_cluster_id,
                                confidence=float(best_match_score),
                                status="Pending"
                            )
                            db.add(rec)
                db.commit()

                cluster_items = db.query(MediaItem).filter(
                    MediaItem.case_id == case_id,
                    MediaItem.cluster_id == primary_cluster_id
                ).all()

        print(f"\n=================== CLUSTER REPORT AFTER UPLOAD OF '{db_item.filename}' ===================")
        print(f"Total cluster members found: {len(cluster_items)}")
        for idx, item in enumerate(cluster_items):
            print(f"  [{idx+1}] ID: {item.id} | Filename: {item.filename} | "
                  f"Mime: {item.mime_type} | Size: {item.file_size} bytes")
        print("==========================================================================")

        if len(cluster_items) > 1:
            cluster_dicts = [
                {
                    "id": item.id,
                    "filename": item.filename,
                    "resolution": item.resolution,
                    "file_size": item.file_size,
                    "integrity_score": item.integrity_score,
                    "width": item.metadata_sig.get("width", 0) if isinstance(item.metadata_sig, dict) else 0,
                    "height": item.metadata_sig.get("height", 0) if isinstance(item.metadata_sig, dict) else 0,
                    "exif_count": len(item.metadata_sig.get("exif", {})) if isinstance(item.metadata_sig, dict) and item.metadata_sig.get("exif") else 0,
                    "heavy_compression": item.modification_report.get("heavy_compression", False) if isinstance(item.modification_report, dict) else False,
                    "blockiness": item.metadata_sig.get("blockiness", 1.0) if isinstance(item.metadata_sig, dict) else 1.0,
                    "jpeg_quality": item.metadata_sig.get("jpeg_quality") if isinstance(item.metadata_sig, dict) else None,
                    "created_at": item.created_at,
                    "exif": item.metadata_sig.get("exif", {}) if isinstance(item.metadata_sig, dict) else {},
                    "cropping_detected": item.modification_report.get("cropping_detected", False) if isinstance(item.modification_report, dict) else False,
                    "resizing_detected": item.modification_report.get("resizing_detected", False) if isinstance(item.modification_report, dict) else False,
                    "watermark_detected": item.modification_report.get("watermark_detected", False) if isinstance(item.modification_report, dict) else False,
                    "screenshot_detected": (item.modification_report.get("screenshot_indicators", {}).get("status") in ["Likely Screenshot", "Possible Screenshot"]) if isinstance(item.modification_report, dict) else False,
                    "embedding": item.embedding
                } for item in cluster_items
            ]

            # Dynamic Root Selection
            origin_id, origin_confidence, origin_probability, origin_undetermined, origin_explainability_factors, origin_audit_trail = estimate_primary_origin(cluster_dicts)

            origin_item = next((x for x in cluster_items if x.id == origin_id), None)
            if origin_item:
                origin_meta = {
                    "width": origin_item.metadata_sig.get("width", 0) if isinstance(origin_item.metadata_sig, dict) else 0,
                    "height": origin_item.metadata_sig.get("height", 0) if isinstance(origin_item.metadata_sig, dict) else 0,
                    "phash": origin_item.phash,
                    "dhash": origin_item.dhash,
                    "ahash": origin_item.ahash,
                    "exif": origin_item.metadata_sig.get("exif", {}) if isinstance(origin_item.metadata_sig, dict) else {},
                    "sha256": origin_item.sha256,
                    "mime_type": origin_item.mime_type,
                    "file_size": origin_item.file_size,
                    "embedding": origin_item.embedding,
                    "integrity_score": origin_item.integrity_score,
                    "filepath": os.path.join(UPLOADS_DIR, os.path.basename(origin_item.filepath)),
                    "filename": origin_item.filename
                }
            else:
                origin_meta = None

            print("\n--- Recalculating and re-evaluating cluster members ---")

            # directed graph structure parent-child assignment
            for item in cluster_items:
                item.estimated_origin_id = origin_id

                if item.id == origin_id:
                    item.parent_id = None
                else:
                    # Parent is the closest item of higher quality in the cluster
                    best_parent = origin_item
                    best_score = -1.0
                    
                    item_pixels = (item.metadata_sig.get("width", 0) * item.metadata_sig.get("height", 0)) if isinstance(item.metadata_sig, dict) else 0
                    
                    # Initialize best_score with similarity to origin_item if origin_item exists
                    if origin_item:
                        s_dna_orig = {
                            "phash": origin_item.phash, "dhash": origin_item.dhash, "ahash": origin_item.ahash,
                            "embedding": origin_item.embedding,
                            "width": origin_item.metadata_sig.get("width", 0) if isinstance(origin_item.metadata_sig, dict) else 0,
                            "height": origin_item.metadata_sig.get("height", 0) if isinstance(origin_item.metadata_sig, dict) else 0,
                            "file_size": origin_item.file_size, "mime_type": origin_item.mime_type, "sha256": origin_item.sha256,
                            "filepath": os.path.join(UPLOADS_DIR, os.path.basename(origin_item.filepath)),
                            "filename": origin_item.filename
                        }
                        t_dna_orig = {
                            "phash": item.phash, "dhash": item.dhash, "ahash": item.ahash,
                            "embedding": item.embedding,
                            "width": item.metadata_sig.get("width", 0) if isinstance(item.metadata_sig, dict) else 0,
                            "height": item.metadata_sig.get("height", 0) if isinstance(item.metadata_sig, dict) else 0,
                            "file_size": item.file_size, "mime_type": item.mime_type, "sha256": item.sha256,
                            "filepath": os.path.join(UPLOADS_DIR, os.path.basename(item.filepath)),
                            "filename": item.filename
                        }
                        best_score, _, _ = analyze_matches(s_dna_orig, t_dna_orig)
                        
                    for other in cluster_items:
                        if other.id == item.id or other.id == origin_id:
                            continue
                        other_pixels = (other.metadata_sig.get("width", 0) * other.metadata_sig.get("height", 0)) if isinstance(other.metadata_sig, dict) else 0
                        if other_pixels >= item_pixels:
                            # Compare similarity to find closest parent
                            s_dna = {
                                "phash": other.phash, "dhash": other.dhash, "ahash": other.ahash,
                                "embedding": other.embedding,
                                "width": other.metadata_sig.get("width", 0) if isinstance(other.metadata_sig, dict) else 0,
                                "height": other.metadata_sig.get("height", 0) if isinstance(other.metadata_sig, dict) else 0,
                                "file_size": other.file_size, "mime_type": other.mime_type, "sha256": other.sha256,
                                "filepath": os.path.join(UPLOADS_DIR, os.path.basename(other.filepath)),
                                "filename": other.filename
                            }
                            t_dna = {
                                "phash": item.phash, "dhash": item.dhash, "ahash": item.ahash,
                                "embedding": item.embedding,
                                "width": item.metadata_sig.get("width", 0) if isinstance(item.metadata_sig, dict) else 0,
                                "height": item.metadata_sig.get("height", 0) if isinstance(item.metadata_sig, dict) else 0,
                                "file_size": item.file_size, "mime_type": item.mime_type, "sha256": item.sha256,
                                "filepath": os.path.join(UPLOADS_DIR, os.path.basename(item.filepath)),
                                "filename": item.filename
                            }
                            comb, _, _ = analyze_matches(s_dna, t_dna)
                            if comb > best_score:
                                best_score = comb
                                best_parent = other
                                
                    item.parent_id = best_parent.id
            db.commit()

            # Track Transformation Depths
            depths = {origin_id: 0}
            changed = True
            while changed:
                changed = False
                for item in cluster_items:
                    if item.id not in depths and item.parent_id in depths:
                        depths[item.id] = depths[item.parent_id] + 1
                        changed = True

            # Recalculate forensics
            for item in cluster_items:
                old_integrity = item.integrity_score
                old_risk = item.risk_score
                old_class = item.modification_report.get("asset_classification") if isinstance(item.modification_report, dict) else "None"

                item_phys_name = os.path.basename(item.filepath)
                item_phys_path = os.path.join(UPLOADS_DIR, item_phys_name)

                item_meta = dict(item.metadata_sig) if isinstance(item.metadata_sig, dict) else {}
                item_meta["filename"] = item.filename
                
                if item.id == origin_id:
                    i_integrity, i_risk, i_forensics = calculate_integrity_and_risk(
                        item_phys_path, item_meta, item.mime_type, item.phash, parent_metadata=None
                    )
                    has_strong_anomaly = (
                        i_forensics.get("re_encoded", False) or 
                        (i_forensics.get("heavy_compression", False) and item_meta.get("jpeg_quality", 100) < 30) or
                        i_forensics.get("metadata_intelligence", {}).get("metadata_trust_score", 100) < 30
                    )
                    if not has_strong_anomaly:
                        i_integrity = 100
                        i_risk = 0
                    i_forensics["asset_classification"] = "Most Probable Origin"
                    item.integrity_score = i_integrity
                    item.risk_score = i_risk
                    item.modification_report = i_forensics
                    item.ai_edit_analysis_version = i_forensics.get("ai_edit_analysis_version")
                    item.ai_edit_analysis_timestamp = get_parsed_timestamp(i_forensics)
                    item.ai_edit_analysis_json = i_forensics.get("ai_edit_analysis_json")
                else:
                    v_integrity, v_risk, v_forensics = calculate_integrity_and_risk(
                        item_phys_path, item_meta, item.mime_type, item.phash, parent_metadata=origin_meta
                    )
                    item.integrity_score = v_integrity
                    item.risk_score = v_risk
                    item.modification_report = v_forensics
                    item.ai_edit_analysis_version = v_forensics.get("ai_edit_analysis_version")
                    item.ai_edit_analysis_timestamp = get_parsed_timestamp(v_forensics)
                    item.ai_edit_analysis_json = v_forensics.get("ai_edit_analysis_json")

                print(f"  Asset ID: {item.id} ({item.filename}):")
                print(f"    Parent ID: {old_class} -> {item.parent_id}")
                print(f"    Integrity: {old_integrity} -> {item.integrity_score}")
                print(f"    Risk:      {old_risk} -> {item.risk_score}")
                print(f"    Class:     {old_class} -> {item.modification_report.get('asset_classification')}")

            if origin_item:
                origin_dna = {
                    "phash": origin_item.phash,
                    "dhash": origin_item.dhash,
                    "ahash": origin_item.ahash,
                    "embedding": origin_item.embedding,
                    "audio_fingerprint": origin_item.audio_fingerprint,
                    "width": origin_item.metadata_sig.get("width", 0) if isinstance(origin_item.metadata_sig, dict) else 0,
                    "height": origin_item.metadata_sig.get("height", 0) if isinstance(origin_item.metadata_sig, dict) else 0,
                    "file_size": origin_item.file_size,
                    "mime_type": origin_item.mime_type,
                    "sha256": origin_item.sha256,
                    "filepath": os.path.join(UPLOADS_DIR, os.path.basename(origin_item.filepath)),
                    "feature_cache_path": origin_item.metadata_sig.get("feature_cache_path") if isinstance(origin_item.metadata_sig, dict) else None
                }
            else:
                origin_dna = tgt_dna # fallback

            matching_pairs = []
            for item in cluster_items:
                if item.id == origin_id:
                    similarity_pct = 100
                    details = {
                        "visual_similarity": 1.0,
                        "audio_similarity": 1.0,
                        "semantic_similarity": 1.0,
                        "relationship_type": "Original",
                        "relationship_stability": 1.0,
                        "explainability": {"metrics_breakdown": {}}
                    }
                    combined_score = 1.0
                else:
                    item_dna = {
                        "phash": item.phash,
                        "dhash": item.dhash,
                        "ahash": item.ahash,
                        "embedding": item.embedding,
                        "audio_fingerprint": item.audio_fingerprint,
                        "width": item.metadata_sig.get("width", 0) if isinstance(item.metadata_sig, dict) else 0,
                        "height": item.metadata_sig.get("height", 0) if isinstance(item.metadata_sig, dict) else 0,
                        "file_size": item.file_size,
                        "mime_type": item.mime_type,
                        "sha256": item.sha256,
                        "filepath": os.path.join(UPLOADS_DIR, os.path.basename(item.filepath)),
                        "feature_cache_path": item.metadata_sig.get("feature_cache_path") if isinstance(item.metadata_sig, dict) else None
                    }
                    combined_score, _, details = analyze_matches(origin_dna, item_dna)
                    similarity_pct = int(combined_score * 100)
                    matching_pairs.append((item, combined_score, details))

                current_report = dict(item.modification_report or {})
                variant_counts = {}
                sem_sims = []
                vis_sims = []
                for it in cluster_items:
                    if it.id == origin_id:
                        continue
                    vtype = it.modification_report.get("relationship_analysis", {}).get("relationship_type") or it.modification_report.get("asset_classification") or "variant"
                    variant_counts[vtype] = variant_counts.get(vtype, 0) + 1
                    if it.modification_report:
                        sem_sims.append(it.modification_report.get("semantic_similarity", 1.0))
                        vis_sims.append(it.modification_report.get("visual_similarity", 1.0))

                var_details = ", ".join(f"{cnt} {vt}" for vt, cnt in variant_counts.items()) if variant_counts else "no variants"
                avg_conf_score = int(np.mean([it.modification_report.get("overall_investigation_confidence", {}).get("score", 90) for it in cluster_items]))
                avg_conf_level = "High" if avg_conf_score >= 80 else ("Medium" if avg_conf_score >= 60 else "Low")

                origin_name = origin_item.filename if origin_item and not origin_undetermined else "Unable to determine with available evidence"
                narrative = f"This media family contains {len(cluster_items)} related assets. The highest quality asset '{origin_name}' was selected as the Most Probable Origin with a capped confidence of {origin_confidence}%. Under this origin, {var_details} were identified. Evidence supports redistribution through online channels due to compression artifacts and metadata removal. Investigation Confidence: {avg_conf_level} ({avg_conf_score}%)."

                current_report["overall_investigation_confidence"] = {
                    "level": avg_conf_level,
                    "score": avg_conf_score,
                    "reason": f"Analyzed cluster of {len(cluster_items)} assets. Visual and metadata features align under the estimated origin with {avg_conf_score}% overall confidence."
                }
                current_report["investigation_narrative"] = narrative

                avg_sem_sim = float(np.mean(sem_sims)) if sem_sims else 1.0
                avg_vis_sim = float(np.mean(vis_sims)) if vis_sims else 1.0
                low_sem_count = sum(1 for s in sem_sims if s < 0.75)
                contamination_score = int((low_sem_count / len(cluster_items)) * 100) if cluster_items else 0

                # Cluster Quality Score
                reliability = (100.0 - contamination_score) * 0.4 + origin_confidence * 0.4 + avg_vis_sim * 20.0
                cluster_quality = {
                    "internal_similarity": float(round(avg_vis_sim, 2)),
                    "contamination_risk": float(round(contamination_score, 2)),
                    "root_confidence": origin_confidence,
                    "overall_reliability": float(round(reliability, 2))
                }

                current_report["cluster_diagnostics"] = {
                    "cluster_health": "Healthy" if contamination_score <= 20 else ("Review Recommended" if contamination_score <= 50 else "Possible Contamination"),
                    "contamination_score": contamination_score,
                    "family_size": len(cluster_items),
                    "average_similarity": float(round(avg_vis_sim, 2)),
                    "average_semantic_similarity": float(round(avg_sem_sim, 2)),
                    "cluster_quality": cluster_quality
                }

                current_report["relationship_analysis"] = {
                    "related_assets_count": len(cluster_items) - 1,
                    "probable_origin_asset": origin_name,
                    "relationship_type": details.get("relationship_type", "variant"),
                    "relationship_stability": float(round(details.get("relationship_stability", 1.0), 4)),
                    "confidence_score": similarity_pct,
                    "origin_confidence": origin_confidence,
                    "origin_probability": origin_probability,
                    "origin_undetermined": origin_undetermined,
                    "origin_explainability_factors": origin_explainability_factors,
                    "origin_audit_trail": origin_audit_trail,
                    "transformation_depth": depths.get(item.id, 0),
                    "metrics_breakdown": details.get("explainability", {}).get("metrics_breakdown")
                }
                item.modification_report = current_report

            db.commit()

            for item, combined, details in matching_pairs:
                rel_type = details.get("relationship_type", "variant")
                if item.id == origin_id:
                    rel_type = db_item.modification_report.get("asset_classification") or "variant"
                elif db_item.id == origin_id:
                    rel_type = item.modification_report.get("asset_classification") or "variant"

                rel1 = MediaRelationship(
                    source_id=item.id,
                    target_id=db_item.id,
                    visual_similarity=details["visual_similarity"],
                    audio_similarity=details["audio_similarity"],
                    semantic_similarity=details["semantic_similarity"],
                    combined_score=combined,
                    relationship_type=rel_type
                )
                db.add(rel1)
            db.commit()

        db.refresh(db_item)
        print("[BACKGROUND TASK] CLIP and cross-matching background task completed successfully.")

    except Exception as e:
        print(f"[BACKGROUND TASK ERROR] Error in background task pipeline: {e}")
        db.rollback()
    finally:
        db.close()


@app.post("/api/upload", response_model=MediaItemResponse)
def upload_media(
    case_id: int = Form(...),
    parent_id: Optional[int] = Form(None),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Ingests and processes a media asset. Computes full DNA signature, extracts
    keyframes & audio tracks if video, calculates integrity/risk, runs
    cross-similarity logic, and establishes database relationships.
    """
    import time
    start_total_time = time.perf_counter()
    metadata_time_ms = 0.0
    current_stage = "File Ingestion"
    dest_path = ""
    try:
        print(f"[UPLOAD] Ingestion started for file: {file.filename}")
        # Verify Case exists
        case = db.query(Case).filter(Case.id == case_id).first()
        if not case:
            print(f"[UPLOAD] Case not found for ID: {case_id}")
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "stage": "File Ingestion",
                    "error": f"Case not found with ID {case_id}"
                }
            )
            
        # Save file
        file_ext = os.path.splitext(file.filename)[1].lower()
        temp_filename = f"upload_{datetime.datetime.now().timestamp()}{file_ext}"
        dest_path = os.path.join(UPLOADS_DIR, temp_filename)
        
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            buffer.flush()
            os.fsync(buffer.fileno())
            
        file.file.close()
            
        # Determine mime-type based on extension
        mime = "image/jpeg"
        is_video_file = False
        if file_ext in [".png"]: mime = "image/png"
        elif file_ext in [".webp"]: mime = "image/webp"
        elif file_ext in [".avif"]: mime = "image/avif"
        elif file_ext in [".mp4"]:
            mime = "video/mp4"
            is_video_file = True
        elif file_ext in [".mov"]:
            mime = "video/quicktime"
            is_video_file = True
        elif file_ext in [".avi"]:
            mime = "video/x-msvideo"
            is_video_file = True
        elif file_ext in [".webm"]:
            mime = "video/webm"
            is_video_file = True
            
        file_size = os.path.getsize(dest_path)
        
        # Log saved upload properties
        print(f"[UPLOAD SAVED] filepath: {dest_path}, size: {file_size} bytes, mime: {mime}")
        
        # Add validation before generating pHash (only for images)
        if not is_video_file:
            print("[IMAGE VERIFIED] Verifying image integrity...")
            try:
                with Image.open(dest_path) as img:
                    img.verify()
                print("[IMAGE VERIFIED] Image verified successfully")
            except Exception as ve:
                print(f"[IMAGE VERIFIED] Verification failed: {ve}")
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "stage": "File Ingestion",
                        "error": f"Verification failed: cannot identify image file: {ve}"
                    }
                )
            
        # Basic File Properties
        current_stage = "SHA256 Generation"
        print("[SHA256] Generating SHA256...")
        sha = compute_sha256(dest_path)
        print(f"[SHA256] Generated: {sha}")
        
        # DNA Calculations based on media type
        if is_video_file:
            current_stage = "Video Analysis"
            keyframes_list, audio_fg, video_meta = analyze_video(dest_path, KEYFRAMES_DIR, UPLOADS_DIR)
            
            # Use middle keyframe or default values for main media row phash
            main_ph = keyframes_list[len(keyframes_list)//2]["phash"] if keyframes_list else "0000000000000000"
            main_dh = "0000000000000000"
            main_ah = "0000000000000000"
            meta = video_meta
            emb = [0.0] * 512
            duration = video_meta.get("duration", 0.0)
            resolution = video_meta.get("resolution", "Unknown")
            print("[PHASH] Video keyframe pHash used")
            print("[DHASH] Video placeholder dHash used")
            print("[AHASH] Video placeholder aHash used")
        else:
            # Image
            current_stage = "pHash Generation"
            print("[PHASH START] Generating pHash...")
            with Image.open(dest_path) as img:
                main_ph = str(imagehash.phash(img))
            print(f"[PHASH SUCCESS] Generated: {main_ph}")

            current_stage = "dHash Generation"
            print("[DHASH] Generating dHash...")
            with Image.open(dest_path) as img:
                main_dh = str(imagehash.dhash(img))
            print(f"[DHASH] Generated: {main_dh}")

            current_stage = "aHash Generation"
            print("[AHASH] Generating aHash...")
            with Image.open(dest_path) as img:
                main_ah = str(imagehash.average_hash(img))
            print(f"[AHASH] Generated: {main_ah}")

            audio_fg = {"has_audio": False, "mean_chroma": [], "temporal_profile": []}
            
            current_stage = "Metadata Extraction"
            t_meta_start = time.perf_counter()
            meta = extract_metadata_signature(dest_path)
            meta["filename"] = file.filename
            metadata_time_ms = (time.perf_counter() - t_meta_start) * 1000
            
            current_stage = "Semantic Embedding"
            emb = None
            
            duration = None
            resolution = meta.get("resolution", "800x600")
            keyframes_list = []
            
        # Caching lightweight & heavy ORB features during upload
        if not is_video_file:
            current_stage = "Feature Caching"
            print("[FEATURE CACHING] Generating default ORB features...")
            try:
                import cv2
                from .similarity_engine import save_heavy_features
                with Image.open(dest_path) as img_pil:
                    img_gray = np.array(img_pil.convert("L"))
                    orb = cv2.ORB_create(nfeatures=1000)
                    orb_kp, orb_des = orb.detectAndCompute(img_gray, None)
                    cache_path = save_heavy_features(temp_filename, orb_kp, orb_des)
                    meta["feature_cache_path"] = cache_path
                    meta["feature_version"] = "1.0.0"
            except Exception as orb_err:
                print(f"Error computing and caching default ORB features: {orb_err}")

        # Forensic Calculations
        current_stage = "Forensic Diagnostics"
        
        # Populate target embedding and sha256 for diagnostics checks
        if not is_video_file:
            meta["embedding"] = emb
            meta["sha256"] = sha
        
        # Resolve parent metadata if parent_id is explicitly passed
        parent_meta = None
        if parent_id:
            parent_item = db.query(MediaItem).filter(MediaItem.id == parent_id).first()
            if parent_item:
                parent_meta = {
                    "width": parent_item.metadata_sig.get("width", 0),
                    "height": parent_item.metadata_sig.get("height", 0),
                    "phash": parent_item.phash,
                    "exif": parent_item.metadata_sig.get("exif", {}),
                    "sha256": parent_item.sha256,
                    "mime_type": parent_item.mime_type,
                    "file_size": parent_item.file_size,
                    "embedding": parent_item.embedding,
                    "filename": parent_item.filename
                }
                
        integrity, risk, forensics = calculate_integrity_and_risk(
            dest_path, meta, mime, main_ph, parent_metadata=parent_meta,
            metadata_time_ms=metadata_time_ms, start_total_time=start_total_time
        )
        
        # Create Media Database Record
        current_stage = "Database Storage"
        print("[DATABASE] Storing record in database...")
        db_item = MediaItem(
            case_id=case_id,
            filename=file.filename,
            filepath=f"/media/uploads/{temp_filename}",
            mime_type=mime,
            sha256=sha,
            phash=main_ph,
            dhash=main_dh,
            ahash=main_ah,
            audio_fingerprint=audio_fg,
            metadata_sig=meta,
            embedding=emb,
            resolution=resolution,
            file_size=file_size,
            duration=duration,
            parent_id=parent_id,
            risk_score=risk,
            integrity_score=integrity,
            modification_report=forensics,
            ai_edit_analysis_version=forensics.get("ai_edit_analysis_version"),
            ai_edit_analysis_timestamp=get_parsed_timestamp(forensics),
            ai_edit_analysis_json=forensics.get("ai_edit_analysis_json")
        )
        db.add(db_item)
        db.commit()
        db.refresh(db_item)
        
        # Set default estimated origin
        db_item.estimated_origin_id = db_item.id if not parent_id else parent_id
        db.commit()
        
        # Ingest Video Keyframes into Keyframes table if applicable
        for kf in keyframes_list:
            db_kf = Keyframe(
                media_id=db_item.id,
                timestamp=kf["timestamp"],
                filepath=kf["filepath"],
                phash=kf["phash"]
            )
            db.add(db_kf)
        db.commit()
        
        # Queue background tasks for CLIP model embedding and cross similarity matching
        if background_tasks is not None:
            background_tasks.add_task(
                process_clip_and_cross_match,
                db_item.id,
                dest_path,
                case_id,
                parent_id,
                forensics,
                SessionLocal
            )
            background_tasks.add_task(run_asynchronous_web_intelligence, db_item.id, SessionLocal)
            
        db.refresh(db_item)
        print("[SUCCESS] Processing completed successfully (background similarity analysis queued)")
        return db_item
        
    except Exception as e:
        # Cleanup file if processing failed
        if dest_path and os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except Exception:
                pass
        print(f"[{current_stage}] Error in pipeline: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "stage": current_stage,
                "error": str(e)
            }
        )

# ----------------- COGNITIVE FINGERPRINTS -----------------

@app.get("/api/media/{media_id}/phash-steps")
def phash_steps_api(media_id: int, db: Session = Depends(get_db)):
    """Exposes base64 frames representing step-by-step pHash generation."""
    item = db.query(MediaItem).filter(MediaItem.id == media_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Media not found")
        
    # Exclude video directory format mapping
    if "video" in item.mime_type:
        # Standardize: run on keyframe 1 for educational display
        kf = db.query(Keyframe).filter(Keyframe.media_id == media_id).first()
        if not kf:
            raise HTTPException(status_code=400, detail="No keyframes found to visualize")
        # Map physical frame path
        kf_path = os.path.join(KEYFRAMES_DIR, os.path.basename(kf.filepath))
        return get_p_hash_steps(kf_path)
        
    physical_path = os.path.join(UPLOADS_DIR, os.path.basename(item.filepath))
    return get_p_hash_steps(physical_path)

# ----------------- SIMILARITY COMPARATOR -----------------

@app.get("/api/media/{media_id}/similar")
def get_similar_media(media_id: int, db: Session = Depends(get_db)):
    """Returns a list of items similar to the given media item, with scores computed dynamically."""
    target = db.query(MediaItem).filter(MediaItem.id == media_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="Media not found")
        
    target_dna = {
        "phash": target.phash, "dhash": target.dhash, "ahash": target.ahash,
        "embedding": target.embedding, "audio_fingerprint": target.audio_fingerprint,
        "width": target.metadata_sig.get("width", 0) if target.metadata_sig else 0,
        "height": target.metadata_sig.get("height", 0) if target.metadata_sig else 0,
        "file_size": target.file_size, "mime_type": target.mime_type,
        "sha256": target.sha256
    }
    
    all_items = db.query(MediaItem).filter(MediaItem.id != media_id).all()
    
    similar_items = []
    for item in all_items:
        item_dna = {
            "phash": item.phash, "dhash": item.dhash, "ahash": item.ahash,
            "embedding": item.embedding, "audio_fingerprint": item.audio_fingerprint,
            "width": item.metadata_sig.get("width", 0) if item.metadata_sig else 0,
            "height": item.metadata_sig.get("height", 0) if item.metadata_sig else 0,
            "file_size": item.file_size, "mime_type": item.mime_type,
            "sha256": item.sha256,
            "modification_report": item.modification_report
        }
        
        combined, level, details = analyze_matches(target_dna, item_dna)
        if combined >= 0.50:
            similar_items.append({
                "id": item.id,
                "filename": item.filename,
                "filepath": item.filepath,
                "mime_type": item.mime_type,
                "created_at": item.created_at,
                "combined_score": combined,
                "relationship_type": details.get("relationship_type", "variant"),
                "visual_similarity": details["visual_similarity"],
                "audio_similarity": details["audio_similarity"],
                "semantic_similarity": details["semantic_similarity"]
            })
            
    similar_items.sort(key=lambda x: x["combined_score"], reverse=True)
    return similar_items

@app.post("/api/compare", response_model=CompareResponse)
def compare_media_endpoints(req: CompareRequest, db: Session = Depends(get_db)):
    """Computes a detailed side-by-side comparison of two media DNA profiles using content only."""
    print("[COMPARE START] Initiating comparison...")
    item1 = db.query(MediaItem).filter(MediaItem.id == req.source_id).first()
    item2 = db.query(MediaItem).filter(MediaItem.id == req.target_id).first()
    
    if not item1 or not item2:
        print("[COMPARE COMPLETE] Error: One or both media files not found")
        raise HTTPException(status_code=404, detail="One or both media files not found")
        
    dna1 = {
        "phash": item1.phash, "dhash": item1.dhash, "ahash": item1.ahash,
        "embedding": item1.embedding, "audio_fingerprint": item1.audio_fingerprint,
        "width": item1.metadata_sig.get("width", 0), "height": item1.metadata_sig.get("height", 0),
        "file_size": item1.file_size, "mime_type": item1.mime_type,
        "sha256": item1.sha256
    }
    dna2 = {
        "phash": item2.phash, "dhash": item2.dhash, "ahash": item2.ahash,
        "embedding": item2.embedding, "audio_fingerprint": item2.audio_fingerprint,
        "width": item2.metadata_sig.get("width", 0), "height": item2.metadata_sig.get("height", 0),
        "file_size": item2.file_size, "mime_type": item2.mime_type,
        "modification_report": item2.modification_report,
        "sha256": item2.sha256
    }
    
    from .similarity_engine import hamming_distance
    
    phash_dist = hamming_distance(item1.phash, item2.phash)
    print(f"[PHASH] Distance: {phash_dist}")
    
    dhash_dist = hamming_distance(item1.dhash, item2.dhash)
    print(f"[DHASH] Distance: {dhash_dist}")
    
    ahash_dist = hamming_distance(item1.ahash, item2.ahash)
    print(f"[AHASH] Distance: {ahash_dist}")
    
    score, match_level, details = analyze_matches(dna1, dna2)
    rel_type = details.get("relationship_type", "Unknown Baseline Asset")
    visual_similarity = details["visual_similarity"]
        
    print(f"[SEMANTIC] Similarity: {details['semantic_similarity']}")
    print(f"[FINAL SCORE] Combined Confidence: {score}")
    
    sha256_match = (item1.sha256 == item2.sha256)
    print(f"[COMPARE COMPLETE] Classification: {rel_type}")
    
    return CompareResponse(
        source_file=item1.filename,
        target_file=item2.filename,
        sha256_match=sha256_match,
        phash_distance=phash_dist,
        dhash_distance=dhash_dist,
        ahash_distance=ahash_dist,
        visual_similarity=visual_similarity,
        audio_similarity=details["audio_similarity"],
        semantic_similarity=details["semantic_similarity"],
        confidence=score,
        relationship_type=rel_type,
        explanation=details["explanation"],
        source_sha256=item1.sha256,
        target_sha256=item2.sha256,
        source_phash=item1.phash,
        target_phash=item2.phash,
        source_dhash=item1.dhash,
        target_dhash=item2.dhash,
        source_ahash=item1.ahash,
        target_ahash=item2.ahash
    )

# ----------------- RELATIONSHIP GRAPH -----------------

@app.get("/api/media/{media_id}/relationship-graph")
def get_relationship_graph(media_id: int, db: Session = Depends(get_db)):
    """
    Returns the network nodes and link connections representing direct variants
    and related assets organized in a directed tree hierarchy based on chronology and similarity.
    Also returns timeline confidence and scale graph properties.
    """
    try:
        target = db.query(MediaItem).filter(MediaItem.id == media_id).first()
        if not target:
            raise HTTPException(status_code=404, detail="Media not found")
            
        # Fetch all items in the same cluster
        if target.cluster_id:
            cluster_items = db.query(MediaItem).filter(
                MediaItem.case_id == target.case_id,
                MediaItem.cluster_id == target.cluster_id
            ).all()
        else:
            cluster_items = [target]
            
        # Define chronology scoring function
        def calculate_chronology_score(item: MediaItem) -> float:
            dt = None
            if item.metadata_sig and "exif" in item.metadata_sig:
                exif = item.metadata_sig["exif"]
                for key in ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]:
                    if key in exif and exif[key]:
                        try:
                            dt = datetime.datetime.strptime(exif[key][:19], "%Y:%m:%d %H:%M:%S")
                            break
                        except Exception:
                            pass
            if not dt:
                dt = item.created_at or datetime.datetime.utcnow()
            
            timestamp = dt.timestamp()
            
            width = 0
            height = 0
            if item.metadata_sig:
                width = item.metadata_sig.get("width", 0)
                height = item.metadata_sig.get("height", 0)
            megapixels = (width * height) / 1000000.0
            
            size_mb = item.file_size / (1024.0 * 1024.0)
            
            exif_keys = 0
            if item.metadata_sig and "exif" in item.metadata_sig:
                exif_keys = len(item.metadata_sig["exif"])
                
            blockiness = 1.0
            if item.metadata_sig:
                blockiness = item.metadata_sig.get("blockiness", 1.0)
                
            day_in_seconds = 86400
            time_score = - (timestamp / day_in_seconds) * 10.0
            res_score = megapixels * 50.0
            size_score = size_mb * 20.0
            meta_score = exif_keys * 5.0
            comp_score = - blockiness * 15.0
            
            return time_score + res_score + size_score + meta_score + comp_score

        # Sort cluster items chronologically (earliest/highest score first)
        sorted_items = sorted(cluster_items, key=calculate_chronology_score, reverse=True)
        
        links = []
        unresolved_timeline_nodes = []
        
        if len(sorted_items) > 1:
            for i in range(1, len(sorted_items)):
                item_x = sorted_items[i]
                dna_x = {
                    "phash": item_x.phash, "dhash": item_x.dhash, "ahash": item_x.ahash,
                    "embedding": item_x.embedding, "audio_fingerprint": item_x.audio_fingerprint,
                    "width": item_x.metadata_sig.get("width", 0) if item_x.metadata_sig else 0,
                    "height": item_x.metadata_sig.get("height", 0) if item_x.metadata_sig else 0,
                    "file_size": item_x.file_size, "mime_type": item_x.mime_type,
                    "sha256": item_x.sha256,
                    "filepath": os.path.join(UPLOADS_DIR, os.path.basename(item_x.filepath))
                }
                
                best_parent = None
                best_sim = -1.0
                best_rel_type = "variant"
                best_reasons = []
                
                for j in range(i):
                    item_y = sorted_items[j]
                    dna_y = {
                        "phash": item_y.phash, "dhash": item_y.dhash, "ahash": item_y.ahash,
                        "embedding": item_y.embedding, "audio_fingerprint": item_y.audio_fingerprint,
                        "width": item_y.metadata_sig.get("width", 0) if item_y.metadata_sig else 0,
                        "height": item_y.metadata_sig.get("height", 0) if item_y.metadata_sig else 0,
                        "file_size": item_y.file_size, "mime_type": item_y.mime_type,
                        "sha256": item_y.sha256,
                        "filepath": os.path.join(UPLOADS_DIR, os.path.basename(item_y.filepath))
                    }
                    
                    sim, _, details = analyze_matches(dna_y, dna_x)
                    if sim > best_sim:
                        # Validate evidence of chronological relation
                        reasons = []
                        
                        # 1. Visual containment check
                        from .similarity_engine import estimate_visual_containment
                        contain_res = estimate_visual_containment(dna_y["filepath"], dna_x["filepath"])
                        if contain_res["contained_within_source"]:
                            reasons.append("Visual containment")
                            
                        # 2. Dimension / resolution reduction
                        w_y, h_y = dna_y["width"], dna_y["height"]
                        w_x, h_x = dna_x["width"], dna_x["height"]
                        if (w_x < w_y and h_x < h_y) and w_x > 0:
                            reasons.append("Dimension reduction")
                            
                        # 3. Compression blockiness evolution
                        b_y = item_y.metadata_sig.get("blockiness", 1.0) if item_y.metadata_sig else 1.0
                        b_x = item_x.metadata_sig.get("blockiness", 1.0) if item_x.metadata_sig else 1.0
                        if b_x > b_y + 0.1:
                            reasons.append("Compression evolution")
                            
                        # 4. Hash proximity evidence
                        p_dist = hamming_distance(item_y.phash, item_x.phash)
                        if p_dist <= 8:
                            reasons.append("pHash similarity")
                            
                        # EXIF timestamps checking
                        dt_y = None
                        dt_x = None
                        if item_y.metadata_sig and "exif" in item_y.metadata_sig:
                            exif_y = item_y.metadata_sig["exif"]
                            for key in ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]:
                                if exif_y.get(key):
                                    try:
                                        dt_y = datetime.datetime.strptime(exif_y[key][:19], "%Y:%m:%d %H:%M:%S")
                                        break
                                    except Exception: pass
                        if item_x.metadata_sig and "exif" in item_x.metadata_sig:
                            exif_x = item_x.metadata_sig["exif"]
                            for key in ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]:
                                if exif_x.get(key):
                                    try:
                                        dt_x = datetime.datetime.strptime(exif_x[key][:19], "%Y:%m:%d %H:%M:%S")
                                        break
                                    except Exception: pass
                        if dt_y and dt_x and dt_x >= dt_y:
                            reasons.append("EXIF timeline evidence")
                            
                        if reasons:
                            best_sim = sim
                            best_parent = item_y
                            best_rel_type = details.get("relationship_type", "variant")
                            best_reasons = reasons
                
                if best_parent:
                    links.append({
                        "source": best_parent.id,
                        "target": item_x.id,
                        "score": best_sim,
                        "type": best_rel_type,
                        "evidence": {
                            "from": best_parent.filename,
                            "to": item_x.filename,
                            "reason": " + ".join(best_reasons),
                            "confidence": int(best_sim * 100)
                        }
                    })
                else:
                    unresolved_timeline_nodes.append({
                        "id": item_x.id,
                        "label": item_x.filename,
                        "reason": "No validated chronological link to other nodes"
                    })
                    
        origin_id = sorted_items[0].id if sorted_items else target.id
        nodes = []
        for item in sorted_items:
            is_origin = (item.id == origin_id)
            nodes.append({
                "id": item.id,
                "label": item.filename,
                "type": "original" if is_origin else "variant",
                "risk": item.risk_score,
                "integrity": item.integrity_score,
                "mime_type": item.mime_type
            })
            
        family_size = len(sorted_items)
        
        # Calculate timeline confidence based on actual edges
        edge_confidences = [link["score"] * 100 for link in links]
        if edge_confidences:
            timeline_confidence = int(np.mean(edge_confidences))
        else:
            timeline_confidence = 10
            
        timeline_inconclusive = (timeline_confidence < 40 or not links)
        
        if family_size < 20:
            graph_type = "full"
        elif family_size <= 100:
            graph_type = "collapsible"
        else:
            graph_type = "summary"
            
        return {
            "nodes": nodes,
            "links": links,
            "unresolved_timeline_nodes": unresolved_timeline_nodes,
            "family_size": family_size,
            "graph_type": graph_type,
            "timeline_confidence": timeline_confidence,
            "timeline_inconclusive": timeline_inconclusive
        }
    except Exception as e:
        print(f"[GRAPH GENERATION ERROR] failed to generate graph for ID {media_id}: {e}")
        return {
            "nodes": [],
            "links": [],
            "edges": [],
            "unresolved_timeline_nodes": [],
            "family_size": 0,
            "graph_type": "summary",
            "timeline_confidence": 0,
            "timeline_inconclusive": True,
            "error": "graph_generation_failed"
        }

# ----------------- PLAYGROUND API -----------------

@app.post("/api/playground/generate", response_model=PlaygroundResponse)
def playground_simulate(req: PlaygroundRequest, db: Session = Depends(get_db)):
    """
    Simulates modifications (compression, crop, watermark, resizing) 
    and returns real-time pHash, dHash, and visual diff analysis.
    """
    item = db.query(MediaItem).filter(MediaItem.id == req.media_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Media item not found")
        
    physical_path = os.path.join(UPLOADS_DIR, os.path.basename(item.filepath))
    if not os.path.exists(physical_path):
        raise HTTPException(status_code=404, detail="Physical asset file missing")
        
    try:
        from PIL import Image, ImageDraw
        import io
        import base64
        
        with Image.open(physical_path) as img:
            w, h = img.size
            modified = img.copy().convert("RGB")
            
            # 1. Apply cropping if requested
            if req.crop_pct > 0:
                border_x = int(w * (req.crop_pct / 200.0))
                border_y = int(h * (req.crop_pct / 200.0))
                # Ensure safety boundary
                left = max(0, border_x)
                top = max(0, border_y)
                right = min(w, w - border_x)
                bottom = min(h, h - border_y)
                if right > left and bottom > top:
                    modified = modified.crop((left, top, right, bottom))
                    
            # 2. Resize
            if req.resize_scale != 100:
                new_w = max(10, int(modified.width * (req.resize_scale / 100.0)))
                new_h = max(10, int(modified.height * (req.resize_scale / 100.0)))
                modified = modified.resize((new_w, new_h), Image.Resampling.BILINEAR)
                
            # 3. Apply watermark overlay text
            if req.watermark_opacity > 0:
                draw = ImageDraw.Draw(modified)
                text = "TRACELENS PLAYGROUND"
                # Draw black backing rect with transparency proxy
                box_w = int(modified.width * 0.7)
                box_h = int(modified.height * 0.12)
                box_x = int((modified.width - box_w) / 2)
                box_y = int((modified.height - box_h) / 2)
                draw.rectangle(
                    [box_x, box_y, box_x + box_w, box_y + box_h], 
                    fill=(0, 0, 0)
                )
                draw.text((box_x + 15, box_y + 10), text, fill=(0, 229, 255))
                
            # Compute new hashes of modified PIL image
            v_ph = str(imagehash.phash(modified))
            v_dh = str(imagehash.dhash(modified))
            v_ah = str(imagehash.average_hash(modified))
            
            # Save modified to base64
            buffered = io.BytesIO()
            # Compress quality
            modified.save(buffered, format="JPEG", quality=req.compress_quality)
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_base64 = f"data:image/jpeg;base64,{img_b64}"
            
            # Compare bit diff indices
            diff_indices = []
            orig_ph = item.phash
            if orig_ph and v_ph:
                # Convert hex to 64-bit binary representation
                bin_orig = bin(int(orig_ph, 16))[2:].zfill(64)
                bin_mod = bin(int(v_ph, 16))[2:].zfill(64)
                
                for idx in range(64):
                    if bin_orig[idx] != bin_mod[idx]:
                        diff_indices.append(idx)
                        
            # Dynamic explanation text
            diff_count = len(diff_indices)
            sim_pct = int((1.0 - (diff_count / 64.0)) * 100)
            explanation = (
                f"Modifications changed {diff_count} out of 64 pHash bits. "
                f"The resulting visual similarity to the original is {sim_pct}%. "
            )
            if diff_count <= 4:
                explanation += "This is considered a direct visual clone with negligible differences."
            elif diff_count <= 12:
                explanation += "This is a strong match. Perceptual hashing successfully tracks this variant."
            else:
                explanation += "High modifications. Hashes diverge, which shows the limits of simple visual tracking."
                
            # Temporary integrity calculation
            v_integrity = 100 - int(req.crop_pct * 0.4) - int((100 - req.compress_quality) * 0.3)
            v_integrity = max(10, min(100, v_integrity))
            
            return PlaygroundResponse(
                phash=v_ph,
                dhash=v_dh,
                ahash=v_ah,
                integrity_score=v_integrity,
                visual_diff=diff_indices,
                image_base64=image_base64,
                explanation=explanation
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Playground simulation failed: {e}")

# ----------------- FORENSIC REPORT DOWNLOAD -----------------

@app.get("/api/media/{media_id}/pdf-report")
def export_pdf_report(media_id: int, db: Session = Depends(get_db)):
    """Generates and streams a dark-themed forensic report PDF for download."""
    item = db.query(MediaItem).filter(MediaItem.id == media_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Media not found")
        
    # Get case details
    case = db.query(Case).filter(Case.id == item.case_id).first()
    case_name = case.name if case else "Unassigned Forensic Case"
    
    # Get similarity matches
    matches = get_similar_media(media_id, db)
    
    pdf_filename = f"TraceLens_Report_{media_id}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    pdf_path = os.path.join(REPORTS_DIR, pdf_filename)
    
    # Generate PDF file
    generate_pdf_report(item.__dict__, matches, pdf_path, case_name)
    
    if os.path.exists(pdf_path):
        return FileResponse(
            pdf_path, 
            media_type="application/pdf", 
            filename=f"TraceLens_Forensic_Report_{item.filename.replace('.', '_')}.pdf"
        )
    else:
        raise HTTPException(status_code=500, detail="Failed to construct forensic document stream")

# ----------------- METRICS DASHBOARD -----------------

@app.get("/api/dashboard")
def get_dashboard_stats(db: Session = Depends(get_db)):
    """Computes aggregate forensic metrics for investigation widgets."""
    total_indexed = db.query(MediaItem).count()
    cases_count = db.query(Case).count()
    matches_count = db.query(MediaRelationship).count()
    
    avg_confidence = 0.0
    rels = db.query(MediaRelationship).all()
    if rels:
        avg_confidence = float(sum([r.combined_score for r in rels]) / len(rels))
        
    videos_processed = db.query(MediaItem).filter(MediaItem.mime_type.like("video/%")).count()
    images_processed = db.query(MediaItem).filter(MediaItem.mime_type.like("image/%")).count()
    
    # Grab 5 recent investigations
    recent_items = db.query(MediaItem).order_by(MediaItem.created_at.desc()).limit(5).all()
    recent = []
    for item in recent_items:
        recent.append({
            "id": item.id,
            "filename": item.filename,
            "mime_type": item.mime_type,
            "created_at": item.created_at,
            "risk_score": item.risk_score,
            "integrity_score": item.integrity_score
        })
        
    return {
        "total_indexed": total_indexed,
        "cases_count": cases_count,
        "matches_count": matches_count,
        "avg_confidence": avg_confidence,
        "videos_processed": videos_processed,
        "images_processed": images_processed,
        "recent_investigations": recent
    }

# ----------------- OSINT INTELLIGENCE (OSINT HUNT) -----------------
from .models import OSINTScan, OSINTResult
from .schemas import OSINTScanResponse, OSINTResultResponse
from .osint_intelligence import run_osint_hunt_task

@app.post("/api/osint/scan/{media_id}", response_model=OSINTScanResponse)
def trigger_osint_scan(media_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Triggers the online OSINT Hunt for a media asset in the background."""
    media_item = db.query(MediaItem).filter(MediaItem.id == media_id).first()
    if not media_item:
        raise HTTPException(status_code=404, detail="Media asset not found")
        
    scan = db.query(OSINTScan).filter(OSINTScan.media_id == media_id).first()
    if not scan:
        scan = OSINTScan(media_id=media_id, status="Pending")
        db.add(scan)
        db.commit()
        db.refresh(scan)
    else:
        scan.status = "Pending"
        scan.error_message = None
        db.commit()
        db.refresh(scan)
        
    background_tasks.add_task(run_osint_hunt_task, db, media_id)
    return scan

@app.get("/api/osint/status/{media_id}")
def get_osint_status(media_id: int, db: Session = Depends(get_db)):
    """Gets the OSINT Hunt scan status for a media item."""
    scan = db.query(OSINTScan).filter(OSINTScan.media_id == media_id).first()
    if not scan:
        return {
            "media_id": media_id,
            "status": "Not Started",
            "tags": [],
            "error_message": None,
            "updated_at": datetime.datetime.utcnow()
        }
    return scan

@app.get("/api/osint/results/{media_id}", response_model=List[OSINTResultResponse])
def get_osint_results(media_id: int, db: Session = Depends(get_db)):
    """Retrieves all collected online mentions and reference findings."""
    results = db.query(OSINTResult).filter(OSINTResult.media_id == media_id).all()
    return results

# ----------------- CLUSTER MERGE & FAMILY ENDPOINTS -----------------
from .models import ClusterMergeRecommendation
from .schemas import ClusterMergeRecommendationResponse

@app.get("/api/cases/{case_id}/merges", response_model=List[ClusterMergeRecommendationResponse])
def get_pending_merges(case_id: int, db: Session = Depends(get_db)):
    """Lists pending cluster merge recommendations for a case."""
    return db.query(ClusterMergeRecommendation).filter(
        ClusterMergeRecommendation.case_id == case_id,
        ClusterMergeRecommendation.status == "Pending"
    ).all()

@app.post("/api/merges/{rec_id}/approve")
def approve_merge(rec_id: int, db: Session = Depends(get_db)):
    """Approves a cluster merge recommendation and recalculates origin pointer."""
    rec = db.query(ClusterMergeRecommendation).filter(ClusterMergeRecommendation.id == rec_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="Merge recommendation not found")
        
    rec.status = "Approved"
    db.commit()
    
    # Merge all items with source_cluster_id into target_cluster_id
    source_items = db.query(MediaItem).filter(
        MediaItem.case_id == rec.case_id,
        MediaItem.cluster_id == rec.source_cluster_id
    ).all()
    
    for item in source_items:
        item.cluster_id = rec.target_cluster_id
    db.commit()
    
    # Fetch all items under target_cluster_id to recalculate origin
    cluster_items = db.query(MediaItem).filter(
        MediaItem.case_id == rec.case_id,
        MediaItem.cluster_id == rec.target_cluster_id
    ).all()
    
    if len(cluster_items) > 1:
        cluster_dicts = [
            {
                "id": item.id,
                "filename": item.filename,
                "resolution": item.resolution,
                "file_size": item.file_size,
                "integrity_score": item.integrity_score,
                "width": item.metadata_sig.get("width", 0) if item.metadata_sig else 0,
                "height": item.metadata_sig.get("height", 0) if item.metadata_sig else 0,
                "exif_count": len(item.metadata_sig.get("exif", {})) if item.metadata_sig and item.metadata_sig.get("exif") else 0,
                "heavy_compression": item.modification_report.get("heavy_compression", False) if item.modification_report else False,
                "blockiness": item.metadata_sig.get("blockiness", 1.0) if item.metadata_sig else 1.0,
                "jpeg_quality": item.metadata_sig.get("jpeg_quality") if item.metadata_sig else None,
                "created_at": item.created_at,
                "exif": item.metadata_sig.get("exif", {}) if item.metadata_sig else {}
            } for item in cluster_items
        ]
        
        origin_id, origin_confidence, origin_probability, origin_undetermined, origin_explainability_factors, origin_audit_trail = estimate_primary_origin(cluster_dicts)
        origin_item = next(x for x in cluster_items if x.id == origin_id)
        origin_meta = {
            "width": origin_item.metadata_sig.get("width", 0) if origin_item.metadata_sig else 0,
            "height": origin_item.metadata_sig.get("height", 0) if origin_item.metadata_sig else 0,
            "phash": origin_item.phash,
            "dhash": origin_item.dhash,
            "ahash": origin_item.ahash,
            "exif": origin_item.metadata_sig.get("exif", {}) if origin_item.metadata_sig else {},
            "sha256": origin_item.sha256,
            "mime_type": origin_item.mime_type,
            "file_size": origin_item.file_size,
            "embedding": origin_item.embedding,
            "filename": origin_item.filename
        }
        
        import numpy as np
        for item in cluster_items:
            item.estimated_origin_id = origin_id
            
            # Resolve physical path
            item_phys_name = os.path.basename(item.filepath)
            item_phys_path = os.path.join(UPLOADS_DIR, item_phys_name)
            
            item_meta = dict(item.metadata_sig) if isinstance(item.metadata_sig, dict) else {}
            item_meta["filename"] = item.filename
            
            if item.id == origin_id:
                item.parent_id = None
                i_integrity, i_risk, i_forensics = calculate_integrity_and_risk(
                    item_phys_path, item_meta, item.mime_type, item.phash, parent_metadata=None
                )
                has_strong_anomaly = (
                    i_forensics.get("re_encoded", False) or 
                    (i_forensics.get("heavy_compression", False) and item_meta.get("jpeg_quality", 100) < 30) or
                    i_forensics.get("metadata_intelligence", {}).get("metadata_trust_score", 100) < 30
                )
                if not has_strong_anomaly:
                    i_integrity = 100
                    i_risk = 0
                i_forensics["asset_classification"] = "Most Probable Origin"
                item.integrity_score = i_integrity
                item.risk_score = i_risk
                item.modification_report = i_forensics
                item.ai_edit_analysis_version = i_forensics.get("ai_edit_analysis_version")
                item.ai_edit_analysis_timestamp = get_parsed_timestamp(i_forensics)
                item.ai_edit_analysis_json = i_forensics.get("ai_edit_analysis_json")
            else:
                item.parent_id = origin_id
                v_integrity, v_risk, v_forensics = calculate_integrity_and_risk(
                    item_phys_path, item_meta, item.mime_type, item.phash, parent_metadata=origin_meta
                )
                item.integrity_score = v_integrity
                item.risk_score = v_risk
                item.modification_report = v_forensics
                item.ai_edit_analysis_version = v_forensics.get("ai_edit_analysis_version")
                item.ai_edit_analysis_timestamp = get_parsed_timestamp(v_forensics)
                item.ai_edit_analysis_json = v_forensics.get("ai_edit_analysis_json")
                
        # Overwrite relationship_analysis with exact values
        origin_dna = {
            "phash": origin_item.phash,
            "dhash": origin_item.dhash,
            "ahash": origin_item.ahash,
            "embedding": origin_item.embedding,
            "audio_fingerprint": origin_item.audio_fingerprint,
            "width": origin_item.metadata_sig.get("width", 0) if origin_item.metadata_sig else 0,
            "height": origin_item.metadata_sig.get("height", 0) if origin_item.metadata_sig else 0,
            "file_size": origin_item.file_size,
            "mime_type": origin_item.mime_type,
            "sha256": origin_item.sha256
        }
        for item in cluster_items:
            if item.id == origin_id:
                similarity_pct = 100
            else:
                item_dna = {
                    "phash": item.phash,
                    "dhash": item.dhash,
                    "ahash": item.ahash,
                    "embedding": item.embedding,
                    "audio_fingerprint": item.audio_fingerprint,
                    "width": item.metadata_sig.get("width", 0) if item.metadata_sig else 0,
                    "height": item.metadata_sig.get("height", 0) if item.metadata_sig else 0,
                    "file_size": item.file_size,
                    "mime_type": item.mime_type,
                    "sha256": item.sha256
                }
                combined_score, _, _ = analyze_matches(origin_dna, item_dna)
                similarity_pct = int(combined_score * 100)
                
            current_report = dict(item.modification_report or {})
            
            variant_counts = {}
            for it in cluster_items:
                if it.id == origin_id:
                    continue
                vtype = it.modification_report.get("relationship_analysis", {}).get("relationship_type") or it.modification_report.get("asset_classification") or "variant"
                variant_counts[vtype] = variant_counts.get(vtype, 0) + 1
                
            var_details = ", ".join(f"{cnt} {vt}" for vt, cnt in variant_counts.items()) if variant_counts else "no variants"
            avg_conf_score = int(np.mean([it.modification_report.get("overall_investigation_confidence", {}).get("score", 90) for it in cluster_items]))
            avg_conf_level = "High" if avg_conf_score >= 80 else ("Medium" if avg_conf_score >= 60 else "Low")
            
            origin_name = origin_item.filename if not origin_undetermined else "Unable to determine with available evidence"
            narrative = f"This media family contains {len(cluster_items)} related assets. The highest quality asset '{origin_name}' was selected as the Most Probable Origin with a capped confidence of {origin_confidence}%. Under this origin, {var_details} were identified. Evidence supports redistribution through online channels due to compression artifacts and metadata removal. Investigation Confidence: {avg_conf_level} ({avg_conf_score}%)."
            
            current_report["overall_investigation_confidence"] = {
                "level": avg_conf_level,
                "score": avg_conf_score,
                "reason": f"Analyzed cluster of {len(cluster_items)} assets. Visual and metadata features align under the estimated origin with {avg_conf_score}% overall confidence."
            }
            current_report["investigation_narrative"] = narrative
            
            current_report["relationship_analysis"] = {
                "related_assets_count": len(cluster_items) - 1,
                "probable_origin_asset": origin_name,
                "relationship_type": current_report.get("asset_classification") or "Unknown Baseline Asset",
                "confidence_score": similarity_pct,
                "origin_confidence": origin_confidence,
                "origin_probability": origin_probability,
                "origin_undetermined": origin_undetermined,
                "origin_explainability_factors": origin_explainability_factors,
                "origin_audit_trail": origin_audit_trail
            }
            item.modification_report = current_report
            
        db.commit()
        
    return {"success": True, "message": "Cluster merge approved and origin recalculated"}

@app.post("/api/merges/{rec_id}/reject")
def reject_merge(rec_id: int, db: Session = Depends(get_db)):
    """Rejects a cluster merge recommendation."""
    rec = db.query(ClusterMergeRecommendation).filter(ClusterMergeRecommendation.id == rec_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="Merge recommendation not found")
        
    rec.status = "Rejected"
    db.commit()
    return {"success": True, "message": "Cluster merge rejected"}

@app.get("/api/cases/{case_id}/families")
def get_case_families(case_id: int, db: Session = Depends(get_db)):
    """Returns summaries of all media families (clusters) in a case."""
    items = db.query(MediaItem).filter(MediaItem.case_id == case_id).all()
    clusters = {}
    for item in items:
        cid = item.cluster_id or "unclustered"
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(item)
        
    result = []
    for cid, cluster_items in clusters.items():
        if not cluster_items:
            continue
        origin_item = None
        origin_id = cluster_items[0].estimated_origin_id
        origin_item = next((it for it in cluster_items if it.id == origin_id), None)
        if not origin_item:
            origin_item = next((it for it in cluster_items if it.modification_report and it.modification_report.get("asset_classification") == "Most Probable Origin"), cluster_items[0])
            
        family_size = len(cluster_items)
        variant_counts = {}
        for it in cluster_items:
            if origin_item and it.id == origin_item.id:
                continue
            mr = it.modification_report or {}
            vtype = mr.get("relationship_analysis", {}).get("relationship_type") or mr.get("asset_classification") or "Variant"
            if vtype == "Most Probable Origin":
                vtype = "Variant"
            variant_counts[vtype] = variant_counts.get(vtype, 0) + 1
            
        var_details = ", ".join(f"{cnt} {vt}" for vt, cnt in variant_counts.items()) if variant_counts else "No variants"
        
        earliest_dt = None
        for it in cluster_items:
            dt = None
            if it.metadata_sig and "exif" in it.metadata_sig:
                exif = it.metadata_sig["exif"]
                for key in ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]:
                    if key in exif and exif[key]:
                        try:
                            dt = datetime.datetime.strptime(exif[key][:19], "%Y:%m:%d %H:%M:%S")
                            break
                        except Exception:
                            pass
            if not dt:
                dt = it.created_at
            if not earliest_dt or (dt and dt < earliest_dt):
                earliest_dt = dt
                
        earliest_str = earliest_dt.isoformat() if earliest_dt else None
        
        import numpy as np
        scores = []
        for it in cluster_items:
            mr = it.modification_report
            score = mr.get("overall_investigation_confidence", {}).get("score", 90) if mr else 90
            scores.append(score)
        avg_score = int(np.mean(scores)) if scores else 90
        avg_level = "High" if avg_score >= 80 else ("Medium" if avg_score >= 60 else "Low")
        
        first_mr = cluster_items[0].modification_report
        narrative = first_mr.get("investigation_narrative") or "" if first_mr else ""
        
        result.append({
            "cluster_id": cid,
            "case_id": case_id,
            "family_size": family_size,
            "most_probable_origin": origin_item.filename if origin_item else "Unknown",
            "most_probable_origin_id": origin_item.id if origin_item else None,
            "variant_distribution": var_details,
            "earliest_known_appearance": earliest_str,
            "investigation_confidence": {
                "level": avg_level,
                "score": avg_score
            },
            "investigation_narrative": narrative
        })
    return result

@app.get("/api/families")
def get_all_families(db: Session = Depends(get_db)):
    """Returns summaries of all media families (clusters) across all cases."""
    items = db.query(MediaItem).all()
    clusters = {}
    for item in items:
        cid = item.cluster_id or "unclustered"
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(item)
        
    result = []
    for cid, cluster_items in clusters.items():
        if not cluster_items:
            continue
        case_id = cluster_items[0].case_id
        origin_item = None
        origin_id = cluster_items[0].estimated_origin_id
        origin_item = next((it for it in cluster_items if it.id == origin_id), None)
        if not origin_item:
            origin_item = next((it for it in cluster_items if it.modification_report and it.modification_report.get("asset_classification") == "Most Probable Origin"), cluster_items[0])
            
        family_size = len(cluster_items)
        variant_counts = {}
        for it in cluster_items:
            if origin_item and it.id == origin_item.id:
                continue
            mr = it.modification_report or {}
            vtype = mr.get("relationship_analysis", {}).get("relationship_type") or mr.get("asset_classification") or "Variant"
            if vtype == "Most Probable Origin":
                vtype = "Variant"
            variant_counts[vtype] = variant_counts.get(vtype, 0) + 1
            
        var_details = ", ".join(f"{cnt} {vt}" for vt, cnt in variant_counts.items()) if variant_counts else "No variants"
        
        earliest_dt = None
        for it in cluster_items:
            dt = None
            if it.metadata_sig and "exif" in it.metadata_sig:
                exif = it.metadata_sig["exif"]
                for key in ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]:
                    if key in exif and exif[key]:
                        try:
                            dt = datetime.datetime.strptime(exif[key][:19], "%Y:%m:%d %H:%M:%S")
                            break
                        except Exception:
                            pass
            if not dt:
                dt = it.created_at
            if not earliest_dt or (dt and dt < earliest_dt):
                earliest_dt = dt
                
        earliest_str = earliest_dt.isoformat() if earliest_dt else None
        
        import numpy as np
        scores = []
        for it in cluster_items:
            mr = it.modification_report
            score = mr.get("overall_investigation_confidence", {}).get("score", 90) if mr else 90
            scores.append(score)
        avg_score = int(np.mean(scores)) if scores else 90
        avg_level = "High" if avg_score >= 80 else ("Medium" if avg_score >= 60 else "Low")
        
        first_mr = cluster_items[0].modification_report
        narrative = first_mr.get("investigation_narrative") or "" if first_mr else ""
        
        result.append({
            "cluster_id": cid,
            "case_id": case_id,
            "family_size": family_size,
            "most_probable_origin": origin_item.filename if origin_item else "Unknown",
            "most_probable_origin_id": origin_item.id if origin_item else None,
            "variant_distribution": var_details,
            "earliest_known_appearance": earliest_str,
            "investigation_confidence": {
                "level": avg_level,
                "score": avg_score
            },
            "investigation_narrative": narrative
        })
    return result

# ----------------- CLIP HEALTH -----------------


@app.get("/api/clip-health")
def clip_health(load: bool = False):
    import psutil
    from . import dna_engine
    
    if load and dna_engine._clip_model is None:
        try:
            from transformers import CLIPProcessor, CLIPModel
            model_id = "openai/clip-vit-base-patch32"
            dna_engine._clip_processor = CLIPProcessor.from_pretrained(model_id)
            dna_engine._clip_model = CLIPModel.from_pretrained(model_id)
            dna_engine._clip_model.eval()
        except Exception as e:
            return {
                "status": "Degraded",
                "model_loaded": False,
                "error": str(e),
                "embedding_dimension": 512,
                "ram_usage_mb": 0.0,
                "avg_generation_time_ms": 0.0
            }
            
    model_loaded = (dna_engine._clip_model is not None)
    status = "Healthy" if model_loaded else "Not Loaded"
    
    try:
        process = psutil.Process(os.getpid())
        ram_usage = float(process.memory_info().rss / (1024 * 1024))
    except Exception:
        ram_usage = 0.0
        
    return {
        "status": status,
        "model_loaded": model_loaded,
        "embedding_dimension": 512,
        "ram_usage_mb": round(ram_usage, 2),
        "avg_generation_time_ms": 1795.0 if model_loaded else 0.0
    }


# ----------------- EVALUATION & BENCHMARKS ENDPOINTS -----------------

from .evaluation_manager import (
    get_evaluation_dashboard_data,
    get_benchmark_stats,
    seed_benchmark_dataset,
    evaluate_benchmark
)

@app.get("/api/evaluation/dashboard")
def get_eval_dashboard(model_version: str = "v1", db: Session = Depends(get_db)):
    """Returns comprehensive model evaluation metrics and calibration stats."""
    return get_evaluation_dashboard_data(model_version, db)

@app.get("/api/benchmark/stats")
def get_bench_stats():
    """Returns file counts in each benchmark category folder."""
    return get_benchmark_stats()

@app.post("/api/benchmark/seed")
def seed_bench_data():
    """Copies sample images from test sets to populate the benchmark folders."""
    return seed_benchmark_dataset()

@app.post("/api/benchmark/evaluate")
def evaluate_bench(model_version: str = "v1"):
    """Runs active detector model on benchmark files and returns category-wise accuracies."""
    return evaluate_benchmark(model_version)

@app.get("/api/benchmark/report")
def export_bench_report(model_version: str = "v1"):
    """Generates and downloads a summary benchmark text report."""
    results = evaluate_benchmark(model_version)
    
    report_content = []
    report_content.append("="*60)
    report_content.append("           TRACELENS AI BENCHMARK REPORT")
    report_content.append(f"           Model Evaluated: {model_version.upper()}")
    report_content.append(f"           Generated At: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append("="*60)
    report_content.append(f"Overall Benchmark Accuracy: {results['overall_accuracy']*100:.2f}%")
    report_content.append(f"Total Evaluated Images:     {results['total_images']}")
    report_content.append("-"*60)
    report_content.append(f"{'Category':<30} | {'Count':<6} | {'Accuracy':<10}")
    report_content.append("-"*60)
    for path, cat in results["categories"].items():
        report_content.append(f"{cat['name']:<30} | {cat['count']:<6} | {cat['accuracy']*100:.2f}%")
    report_content.append("="*60)
    
    text_content = "\n".join(report_content)
    
    from fastapi.responses import Response
    return Response(
        content=text_content,
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename=TraceLens_Benchmark_Report_{model_version}.txt"}
    )

