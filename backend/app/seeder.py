import os
from PIL import Image, ImageDraw
from sqlalchemy.orm import Session
from .models import Case, MediaItem, MediaRelationship
from .dna_engine import compute_sha256, compute_image_hashes, get_clip_embedding, extract_metadata_signature, calculate_integrity_and_risk
from .variant_generator import generate_image_variants
from .similarity_engine import analyze_matches

def seed_data_if_empty(db: Session, uploads_dir: str):
    """
    Checks if database contains data. If empty, generates 3 original files,
    creates 4 modifications of each, computes DNA footprints, and seeds the db.
    """
    if db.query(Case).count() > 0:
        return
        
    print("Database is empty. Initializing TraceLens AI seed dataset...")
    os.makedirs(uploads_dir, exist_ok=True)
    
    # 1. Create a Forensic Case
    default_case = Case(
        name="Case #2026-ALPHA: Intel Leak",
        description="OSINT investigation regarding leaked drone telemetry and satellite imagery logs.",
        status="Active"
    )
    db.add(default_case)
    db.commit()
    db.refresh(default_case)
    
    # 2. Programmatically draw 3 original images using Pillow
    # Original A: Drone Telemetry Overlay
    img1_path = os.path.join(uploads_dir, "drone_telemetry_original.jpg")
    img1 = Image.new("RGB", (800, 600), color=(10, 15, 30))
    draw1 = ImageDraw.Draw(img1)
    draw1.rectangle([50, 50, 750, 550], outline=(0, 229, 255), width=4)
    draw1.line([400, 50, 400, 550], fill=(0, 255, 157), width=1)
    draw1.line([50, 300, 750, 300], fill=(0, 255, 157), width=1)
    draw1.text((80, 80), "DRONE FLIGHT SYSTEM V2.4", fill=(255, 255, 255))
    draw1.text((80, 110), "LAT: 34.0522 N  LON: -118.2437 W", fill=(0, 229, 255))
    draw1.text((80, 140), "ALTITUDE: 1540m  SPEED: 45kts", fill=(0, 255, 157))
    img1.save(img1_path, "JPEG", quality=95)
    
    # Original B: Satellite Recon Asset
    img2_path = os.path.join(uploads_dir, "satellite_recon_original.jpg")
    img2 = Image.new("RGB", (800, 600), color=(20, 20, 20))
    draw2 = ImageDraw.Draw(img2)
    for x in range(0, 800, 80):
        draw2.line([x, 0, x, 600], fill=(40, 40, 40), width=1)
    for y in range(0, 600, 80):
        draw2.line([0, y, 800, y], fill=(40, 40, 40), width=1)
    draw2.ellipse([300, 200, 500, 400], outline=(124, 58, 237), width=5)
    draw2.rectangle([380, 280, 420, 320], fill=(0, 229, 255))
    draw2.text((60, 60), "RECON TARGET COMPLEX // ENCRYPTED", fill=(255, 255, 255))
    img2.save(img2_path, "JPEG", quality=95)

    # Original C: Network Leak Diagram
    img3_path = os.path.join(uploads_dir, "crypto_leak_original.jpg")
    img3 = Image.new("RGB", (800, 600), color=(15, 10, 15))
    draw3 = ImageDraw.Draw(img3)
    draw3.text((320, 50), "SECURE TUNNEL MODEL", fill=(124, 58, 237))
    draw3.rectangle([100, 150, 250, 250], outline=(0, 255, 157), width=3)
    draw3.text((120, 190), "CLIENT HOST", fill=(0, 255, 157))
    draw3.line([250, 200, 550, 200], fill=(0, 229, 255), width=2)
    draw3.rectangle([550, 150, 700, 250], outline=(0, 255, 157), width=3)
    draw3.text((580, 190), "CORE DATACENTER", fill=(0, 255, 157))
    draw3.text((320, 220), "TLS v1.3 VPN", fill=(0, 229, 255))
    img3.save(img3_path, "JPEG", quality=95)
    
    originals = [
        ("drone_telemetry_original.jpg", img1_path),
        ("satellite_recon_original.jpg", img2_path),
        ("crypto_leak_original.jpg", img3_path)
    ]
    
    import uuid
    for filename, filepath in originals:
        sha = compute_sha256(filepath)
        ph, dh, ah = compute_image_hashes(filepath)
        meta = extract_metadata_signature(filepath)
        emb = get_clip_embedding(filepath)
        integrity, risk, forensics = calculate_integrity_and_risk(filepath, meta, "image/jpeg", ph)
        
        c_id = f"cluster_seed_{uuid.uuid4().hex[:8]}"
        
        orig_db = MediaItem(
            case_id=default_case.id,
            filename=filename,
            filepath=f"/media/uploads/{filename}",
            mime_type="image/jpeg",
            sha256=sha,
            phash=ph,
            dhash=dh,
            ahash=ah,
            cluster_id=c_id,
            audio_fingerprint={"has_audio": False, "mean_chroma": [], "temporal_profile": []},
            metadata_sig=meta,
            embedding=emb,
            resolution=meta.get("resolution", "800x600"),
            file_size=meta.get("file_size", 0),
            duration=None,
            risk_score=risk,
            integrity_score=integrity,
            modification_report=forensics
        )
        db.add(orig_db)
        db.commit()
        db.refresh(orig_db)
        
        orig_db.estimated_origin_id = orig_db.id
        db.commit()
        
        # Draw and export mutations
        variants = generate_image_variants(filepath, uploads_dir)
        for var in variants:
            v_path = var["filepath"]
            v_name = var["filename"]
            v_rel = var["relation"]
            
            v_sha = compute_sha256(v_path)
            v_ph, v_dh, v_ah = compute_image_hashes(v_path)
            v_meta = extract_metadata_signature(v_path)
            v_emb = get_clip_embedding(v_path)
            parent_meta = {
                "width": orig_db.metadata_sig.get("width", 0),
                "height": orig_db.metadata_sig.get("height", 0),
                "phash": orig_db.phash,
                "exif": orig_db.metadata_sig.get("exif", {}),
                "sha256": orig_db.sha256,
                "mime_type": orig_db.mime_type,
                "file_size": orig_db.file_size,
                "embedding": orig_db.embedding
            }
            v_integrity, v_risk, v_forensics = calculate_integrity_and_risk(
                v_path, v_meta, "image/jpeg", v_ph, parent_metadata=parent_meta
            )
            
            var_db = MediaItem(
                case_id=default_case.id,
                filename=v_name,
                filepath=f"/media/uploads/{v_name}",
                mime_type="image/jpeg" if not v_name.endswith(".webp") else "image/webp",
                sha256=v_sha,
                phash=v_ph,
                dhash=v_dh,
                ahash=v_ah,
                cluster_id=c_id,
                audio_fingerprint={"has_audio": False, "mean_chroma": [], "temporal_profile": []},
                metadata_sig=v_meta,
                embedding=v_emb,
                resolution=v_meta.get("resolution", "800x600"),
                file_size=v_meta.get("file_size", 0),
                duration=None,
                parent_id=orig_db.id,
                estimated_origin_id=orig_db.id,
                risk_score=v_risk,
                integrity_score=v_integrity,
                modification_report=v_forensics
            )
            db.add(var_db)
            db.commit()
            db.refresh(var_db)
            
            # Relational cross similarity
            src_dna = {
                "phash": orig_db.phash, "dhash": orig_db.dhash, "ahash": orig_db.ahash,
                "embedding": orig_db.embedding, "audio_fingerprint": orig_db.audio_fingerprint,
                "width": orig_db.metadata_sig.get("width"), "height": orig_db.metadata_sig.get("height"),
                "file_size": orig_db.file_size, "mime_type": orig_db.mime_type,
                "sha256": orig_db.sha256
            }
            tgt_dna = {
                "phash": var_db.phash, "dhash": var_db.dhash, "ahash": var_db.ahash,
                "embedding": var_db.embedding, "audio_fingerprint": var_db.audio_fingerprint,
                "width": var_db.metadata_sig.get("width"), "height": var_db.metadata_sig.get("height"),
                "file_size": var_db.file_size, "mime_type": var_db.mime_type,
                "modification_report": v_forensics,
                "sha256": var_db.sha256
            }
            
            combined, level, match_details = analyze_matches(src_dna, tgt_dna)
            
            rel = MediaRelationship(
                source_id=orig_db.id,
                target_id=var_db.id,
                visual_similarity=match_details["visual_similarity"],
                audio_similarity=match_details["audio_similarity"],
                semantic_similarity=match_details["semantic_similarity"],
                combined_score=combined,
                relationship_type=match_details.get("relationship_type", v_rel)
            )
            db.add(rel)
            db.commit()
            
    print("Database seeding finished. 15 variations prepared.")
