from __future__ import annotations

from datetime import datetime

from src.models.contracts import AuditTask, PreLabelResult


class MetadataBuilder:
    def build(self, task: AuditTask, prelabel: PreLabelResult, description: str) -> dict:
        return {
            "task_id": task.task_id,
            "trace_id": task.trace_id,
            "biz_id": task.biz_id,
            "media_type": task.media_type.value,
            "source": task.source,
            "lang": task.lang,
            "priority": task.priority,
            "risk_score_pre": prelabel.risk_score_pre,
            "violation_candidates": [v.value for v in prelabel.violation_candidates],
            "model_version": prelabel.model_version,
            "evidence_size": len(prelabel.evidence),
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
        }
