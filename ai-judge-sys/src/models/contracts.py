from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MediaType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"


class AuditStatus(str, Enum):
    RECEIVED = "received"
    PARSED = "parsed"
    PRELABELED = "prelabeled"
    REVIEWING = "reviewing"
    AUTO_APPROVED = "auto_approved"
    AUTO_REJECTED = "auto_rejected"
    NEEDS_MANUAL = "needs_manual"
    HUMAN_RESOLVED = "human_resolved"
    FAILED = "failed"


class ViolationType(str, Enum):
    NORMAL = "normal"
    ABUSE = "abuse"
    VIOLENCE = "violence"
    PORN = "porn"
    POLITICS = "politics"
    OTHER = "other"


class CaseResultSource(str, Enum):
    HUMAN = "human"
    AI = "ai"


class AuditTask(BaseModel):
    schema_version: str = "1.0.0"
    trace_id: str
    task_id: str
    biz_id: str
    media_type: MediaType
    media_url: str
    content_text: str = ""  # Original text content for text type
    lang: str = "zh"
    priority: int = 5
    source: str = "default"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    retry_count: int = 0
    max_retry: int = 3
    dedupe_key: str = Field(description="Use task_id or business unique key.")


class PreLabelResult(BaseModel):
    violation_candidates: list[ViolationType] = Field(default_factory=list)
    risk_score_pre: float = 0.0
    evidence: list[str] = Field(default_factory=list)
    model_version: str = "prelabel-v1"


class AuditFeature(BaseModel):
    metadata_scalar: dict[str, Any]
    description_desensitized: str
    evidence: str = ""  # Evidence-style description for better retrieval
    embedding_vector: list[float]


class ViolationDetail(BaseModel):
    """Details of a single violation type detection."""
    violation_type: ViolationType
    confidence: float
    evidence: list[str] = Field(default_factory=list)
    reason: str = ""


class AuditDecision(BaseModel):
    schema_version: str = "1.0.0"
    trace_id: str
    task_id: str
    final_label: ViolationType
    confidence: float
    decision_path: list[str] = Field(default_factory=list)
    needs_manual: bool = False
    status: AuditStatus = AuditStatus.REVIEWING
    reason: str = ""
    # Multi-label support: all detected violation types with evidence
    violation_details: list[ViolationDetail] = Field(default_factory=list)


class ManualReviewMessage(BaseModel):
    schema_version: str = "1.0.0"
    trace_id: str
    task_id: str
    biz_id: str
    confidence: float
    final_label: ViolationType
    reason: str
    metadata_scalar: dict[str, Any]
    desensitized_summary: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ManualReviewResultMessage(BaseModel):
    schema_version: str = "1.0.0"
    trace_id: str
    task_id: str
    reviewer: str
    final_label: ViolationType
    comment: str = ""
    approved: bool = False
    reviewed_at: datetime = Field(default_factory=datetime.utcnow)


class ManualAuditCaseMessage(BaseModel):
    """
    Human-reviewed case payload from Spring manual-review system.
    """

    schema_version: str = "1.0.0"
    case_id: str
    trace_id: str
    task_id: str = ""
    media_type: MediaType = MediaType.TEXT
    content_text: str = ""
    desensitized_text: str = ""
    violation_types: list[ViolationType] = Field(default_factory=list)
    review_reason: str = ""
    evidence: list[str] = Field(default_factory=list)
    source: CaseResultSource = CaseResultSource.HUMAN


class AuditResultMessage(BaseModel):
    """
    AI audit result message sent back to the platform.
    """
    schema_version: str = "1.0.0"
    case_id: str
    trace_id: str
    final_label: ViolationType
    confidence: float
    status: AuditStatus
    reason: str = ""
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    source: str = "ai_auto"
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Multi-label support: all detected violation types with evidence
    violation_details: list[ViolationDetail] = Field(default_factory=list)
