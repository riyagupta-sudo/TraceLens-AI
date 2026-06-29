import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from .database import Base

class Case(Base):
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String, default="Active")  # Active, Closed

    media_items = relationship("MediaItem", back_populates="case", cascade="all, delete-orphan")


class MediaItem(Base):
    __tablename__ = "media_items"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    mime_type = Column(String, nullable=False)
    sha256 = Column(String, index=True, nullable=False)
    
    # Perceptual Hashes
    phash = Column(String, index=True, nullable=True)
    dhash = Column(String, nullable=True)
    ahash = Column(String, nullable=True)
    cluster_id = Column(String, index=True, nullable=True)
    
    # DNA Features
    audio_fingerprint = Column(JSON, nullable=True)  # Contains custom spectral peak features
    metadata_sig = Column(JSON, nullable=True)      # Extracted EXIF/stream properties
    embedding = Column(JSON, nullable=True)         # CLIP semantic embedding vector (stored as list)
    
    # Properties
    resolution = Column(String, nullable=True)      # e.g., "1920x1080"
    file_size = Column(Integer, nullable=False)     # bytes
    duration = Column(Float, nullable=True)          # in seconds
    
    # Lineage / History tracking
    parent_id = Column(Integer, ForeignKey("media_items.id"), nullable=True)
    estimated_origin_id = Column(Integer, ForeignKey("media_items.id"), nullable=True)
    
    # Forensic Scores
    risk_score = Column(Integer, default=0)         # 0 - 100
    integrity_score = Column(Integer, default=100)  # 0 - 100
    modification_report = Column(JSON, nullable=True) # Details on compression, cropping, watermark etc.
    
    # Localized AI Editing Forensics Columns
    ai_edit_analysis_version = Column(String, nullable=True)
    ai_edit_analysis_timestamp = Column(DateTime, nullable=True)
    ai_edit_analysis_json = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    case = relationship("Case", back_populates="media_items")
    keyframes = relationship("Keyframe", back_populates="media_item", cascade="all, delete-orphan")
    
    # Self-referential relationships for lineage
    parent = relationship("MediaItem", remote_side=[id], foreign_keys=[parent_id])
    estimated_origin = relationship("MediaItem", remote_side=[id], foreign_keys=[estimated_origin_id])


class Keyframe(Base):
    __tablename__ = "keyframes"

    id = Column(Integer, primary_key=True, index=True)
    media_id = Column(Integer, ForeignKey("media_items.id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(Float, nullable=False)
    filepath = Column(String, nullable=False)
    phash = Column(String, nullable=False)

    media_item = relationship("MediaItem", back_populates="keyframes")


class MediaRelationship(Base):
    __tablename__ = "media_relationships"

    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("media_items.id", ondelete="CASCADE"), nullable=False)
    target_id = Column(Integer, ForeignKey("media_items.id", ondelete="CASCADE"), nullable=False)
    
    # Metrics
    visual_similarity = Column(Float, default=0.0)
    audio_similarity = Column(Float, default=0.0)
    semantic_similarity = Column(Float, default=0.0)
    combined_score = Column(Float, default=0.0)
    
    relationship_type = Column(String, nullable=False)  # original, compressed, cropped, watermarked, re-encoded, etc.

    source = relationship("MediaItem", foreign_keys=[source_id])
    target = relationship("MediaItem", foreign_keys=[target_id])


class OSINTScan(Base):
    __tablename__ = "osint_scans"

    id = Column(Integer, primary_key=True, index=True)
    media_id = Column(Integer, ForeignKey("media_items.id", ondelete="CASCADE"), unique=True, nullable=False)
    status = Column(String, default="Pending")  # "Pending", "Running", "Completed", "Failed"
    tags = Column(JSON, nullable=True)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


class OSINTResult(Base):
    __tablename__ = "osint_results"

    id = Column(Integer, primary_key=True, index=True)
    media_id = Column(Integer, ForeignKey("media_items.id", ondelete="CASCADE"), nullable=False)
    url = Column(String, nullable=False)
    title = Column(String, nullable=True)
    snippet = Column(String, nullable=True)
    publication_date = Column(String, nullable=True)
    source = Column(String, nullable=False)  # 'Reddit', 'Google', 'News'
    confidence = Column(Integer, default=50)  # 0 to 100
    reason = Column(String, nullable=True)  # Reason for match
    source_type = Column(String, nullable=True)  # 'real_provider', 'apify', 'simulated', 'heuristic', 'mock'
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class ClusterMergeRecommendation(Base):
    __tablename__ = "cluster_merge_recommendations"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False)
    source_cluster_id = Column(String, nullable=False)
    target_cluster_id = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    status = Column(String, default="Pending") # "Pending", "Approved", "Rejected"
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


