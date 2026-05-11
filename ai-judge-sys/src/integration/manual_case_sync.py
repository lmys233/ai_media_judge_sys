from __future__ import annotations

import logging
from datetime import datetime
import uuid

from src.feature.desensitize import DesensitizeProcessor
from src.feature.embedder import build_embedding_model
from src.models.contracts import ManualAuditCaseMessage, MediaType, ViolationType
from src.prelabel.text_nb import NaiveBayesTextClassifier
from src.retrieval.milvus_store import MilvusAuditStore

logger = logging.getLogger(__name__)


class ManualCaseSyncService:
    """
    Sync human-reviewed cases into:
    1) NB text training corpus  (original text + label)
    2) Milvus vector database   (desensitized text as embedding)
    """

    def __init__(
        self,
        milvus_store: MilvusAuditStore | None = None,
        nb_model: NaiveBayesTextClassifier | None = None,
        desensitizer: DesensitizeProcessor | None = None,
    ) -> None:
        self.embedder = build_embedding_model()
        self.store = milvus_store or MilvusAuditStore()
        self.nb_model = nb_model or NaiveBayesTextClassifier()
        self.desensitizer = desensitizer or DesensitizeProcessor()

    def connect(self) -> None:
        self.store.connect()

    def ingest_case(self, case: ManualAuditCaseMessage) -> dict:
        nb_update = self._update_nb_from_case(case)
        milvus_update = self._upsert_case_to_milvus(case)
        return {"nb_update": nb_update, "milvus_update": milvus_update}

    # ── NB training ──────────────────────────────────────────────

    def _update_nb_from_case(self, case: ManualAuditCaseMessage) -> dict:
        if case.media_type != MediaType.TEXT:
            return {"status": "skipped", "reason": "non_text_case"}
        if not case.content_text.strip():
            return {"status": "skipped", "reason": "empty_text"}

        label = self._pick_nb_label(case.violation_types).value
        if label not in {"normal", "abuse", "violence", "porn", "politics"}:
            return {"status": "skipped", "reason": "label_not_supported_by_nb", "label": label}

        self.nb_model.append_training_sample(
            text=case.content_text,
            label=label,
            source=case.source.value,
            verified=True,
        )
        retrained = self.nb_model.retrain_from_corpus()
        logger.info("NB训练完成 案例ID=%s 标签=%s 已重训=%s", case.case_id, label, retrained)
        return {"status": "updated", "retrained": retrained, "label": label}

    # ── Milvus vector upsert ─────────────────────────────────────

    def _upsert_case_to_milvus(self, case: ManualAuditCaseMessage) -> dict:
        primary_label = self._pick_primary_violation(case.violation_types)
        risk_score = self._derive_risk_score(case.violation_types)
        label_strings = [v.value for v in case.violation_types]

        desensitized = self._desensitize_for_rag(case.content_text, label_strings)

        vector_text = (
            f"desensitized={desensitized}\n"
            f"labels={','.join(label_strings)}\n"
            f"reason={case.review_reason}\n"
            f"evidence={' ; '.join(case.evidence)}\n"
            f"source={case.source.value}"
        )
        embedding = self.embedder.embed_query(vector_text)
        self.store.upsert_case(
            record_id=str(uuid.uuid4()),
            task_id=case.task_id or case.case_id,
            media_type=case.media_type.value,
            violation_type=primary_label.value,
            risk_score=risk_score,
            source=case.source.value,
            created_at=datetime.utcnow().isoformat(),
            model_version="human-reviewed-case-v1",
            human_verified=True,
            description=desensitized[:3900],
            embedding=embedding,
        )
        logger.info("向量已写入Milvus 案例ID=%s 标签=%s 维度=%d", case.case_id, primary_label.value, len(embedding))
        return {"status": "upserted", "vector_dim": len(embedding), "desensitized_len": len(desensitized)}

    def _desensitize_for_rag(self, text: str, violation_labels: list[str]) -> str:
        if not text.strip():
            return text
        return self.desensitizer.sanitize_for_rag(text, violation_labels)

    def _pick_primary_violation(self, labels: list[ViolationType]) -> ViolationType:
        if not labels:
            return ViolationType.OTHER
        priority = {
            ViolationType.VIOLENCE: 6,
            ViolationType.PORN: 5,
            ViolationType.ABUSE: 4,
            ViolationType.POLITICS: 3,
            ViolationType.OTHER: 2,
            ViolationType.NORMAL: 1,
        }
        return max(labels, key=lambda v: priority.get(v, 0))

    def _pick_nb_label(self, labels: list[ViolationType]) -> ViolationType:
        primary = self._pick_primary_violation(labels)
        if primary in {ViolationType.OTHER}:
            return ViolationType.NORMAL
        return primary

    def _derive_risk_score(self, labels: list[ViolationType]) -> float:
        if not labels:
            return 0.5
        severity = {
            ViolationType.NORMAL: 0.1,
            ViolationType.POLITICS: 0.55,
            ViolationType.ABUSE: 0.65,
            ViolationType.PORN: 0.85,
            ViolationType.VIOLENCE: 0.9,
            ViolationType.OTHER: 0.6,
        }
        max_score = max(severity.get(label, 0.5) for label in labels)
        # multi-violation cases are usually riskier
        bonus = 0.04 * max(0, len(set(labels)) - 1)
        return min(0.99, max_score + bonus)
