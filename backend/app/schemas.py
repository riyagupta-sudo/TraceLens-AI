from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# Case Schemas
class CaseBase(BaseModel):
    name: str
    description: Optional[str] = None
    status: Optional[str] = "Active"

class CaseCreate(CaseBase):
    pass

class CaseResponse(CaseBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

# Keyframe Schemas
class KeyframeResponse(BaseModel):
    id: int
    media_id: int
    timestamp: float
    filepath: str
    phash: str

    class Config:
        from_attributes = True

# Media Relationship Schemas
class RelationshipResponse(BaseModel):
    id: int
    source_id: int
    target_id: int
    visual_similarity: float
    audio_similarity: float
    semantic_similarity: float
    combined_score: float
    relationship_type: str

    class Config:
        from_attributes = True

# Media Item Detail Schemas
class MediaItemResponse(BaseModel):
    id: int
    case_id: int
    filename: str
    filepath: str
    mime_type: str
    sha256: str
    phash: Optional[str] = None
    dhash: Optional[str] = None
    ahash: Optional[str] = None
    cluster_id: Optional[str] = None
    audio_fingerprint: Optional[Dict[str, Any]] = None
    metadata_sig: Optional[Dict[str, Any]] = None
    created_at: datetime
    parent_id: Optional[int] = None
    estimated_origin_id: Optional[int] = None
    resolution: Optional[str] = None
    file_size: int
    duration: Optional[float] = None
    risk_score: int
    integrity_score: int
    modification_report: Optional[Dict[str, Any]] = None
    
    # Localized AI Editing Fields
    ai_edit_analysis_version: Optional[str] = None
    ai_edit_analysis_timestamp: Optional[datetime] = None
    ai_edit_analysis_json: Optional[Dict[str, Any]] = None
    
    keyframes: List[KeyframeResponse] = []

    class Config:
        from_attributes = True

class MediaListItem(BaseModel):
    id: int
    case_id: int
    filename: str
    mime_type: str
    sha256: str
    phash: Optional[str] = None
    cluster_id: Optional[str] = None
    created_at: datetime
    parent_id: Optional[int] = None
    estimated_origin_id: Optional[int] = None
    resolution: Optional[str] = None
    file_size: int
    risk_score: int
    integrity_score: int

    class Config:
        from_attributes = True

class ClusterMergeRecommendationResponse(BaseModel):
    id: int
    case_id: int
    source_cluster_id: str
    target_cluster_id: str
    confidence: float
    status: str
    created_at: datetime

    class Config:
        from_attributes = True

# Compare DNA Schemas
class CompareRequest(BaseModel):
    source_id: int
    target_id: int

class CompareResponse(BaseModel):
    source_file: str
    target_file: str
    sha256_match: bool
    phash_distance: int
    dhash_distance: int
    ahash_distance: int
    visual_similarity: float
    audio_similarity: float
    semantic_similarity: float
    confidence: float
    relationship_type: str
    explanation: str
    source_sha256: str
    target_sha256: str
    source_phash: str
    target_phash: str
    source_dhash: str
    target_dhash: str
    source_ahash: str
    target_ahash: str

# Playground Schemas
class PlaygroundRequest(BaseModel):
    media_id: int
    crop_pct: int = Field(0, ge=0, le=80)       # Crop from edges
    watermark_opacity: int = Field(0, ge=0, le=100) # Text watermark
    compress_quality: int = Field(100, ge=1, le=100) # Compression quality
    resize_scale: int = Field(100, ge=10, le=200) # Scale factor (10% - 200%)

class PlaygroundResponse(BaseModel):
    phash: str
    dhash: str
    ahash: str
    integrity_score: int
    visual_diff: List[int] # List of mismatching bits indices (0-63)
    image_base64: str      # Renderable processed image
    explanation: str       # Explain why the hash changed


# OSINT Schemas
class OSINTResultResponse(BaseModel):
    id: int
    media_id: int
    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    publication_date: Optional[str] = None
    source: str
    confidence: int
    reason: Optional[str] = None
    source_type: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class OSINTScanResponse(BaseModel):
    media_id: int
    status: str
    tags: Optional[List[str]] = []
    error_message: Optional[str] = None
    updated_at: datetime

    class Config:
        from_attributes = True

