from __future__ import annotations

import logging
import uuid
from typing import Any

from src.feature.embedder import build_embedding_model
from src.feature.image_embedder import CLIPImageEmbedder
from src.feature.metadata import MetadataBuilder
from src.models.contracts import AuditFeature, AuditTask, PreLabelResult, ViolationType
from src.retrieval.milvus_store import HybridFilter, MilvusAuditStore
from src.retrieval.reranker import SimpleReranker

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(self, store: MilvusAuditStore | None = None) -> None:
        self.embedding_model = build_embedding_model()
        self.image_embedder = CLIPImageEmbedder()  # 用于图片向量化
        self.store = store or MilvusAuditStore()
        self.reranker = SimpleReranker()
        self.metadata_builder = MetadataBuilder()

    def connect(self) -> None:
        self.store.connect()

    def build_feature(
        self,
        task: AuditTask,
        prelabel: PreLabelResult,
        desensitized_summary: str,
    ) -> AuditFeature:
        metadata = self.metadata_builder.build(task, prelabel, desensitized_summary)
        embedding = self.embedding_model.embed_query(f"{desensitized_summary} {metadata}")
        return AuditFeature(
            metadata_scalar=metadata,
            description_desensitized=desensitized_summary,
            embedding_vector=embedding,
        )

    def build_feature_from_text(
        self,
        task: AuditTask,
        rewritten_text: str,
        nb_label: ViolationType,
        nb_risk_score: float,
        all_violation_labels: list[str] | None = None,
    ) -> AuditFeature:
        """Build feature directly from text for AI auto audit pipeline."""
        # Use all detected violation types for broader matching
        violation_candidates = all_violation_labels if all_violation_labels else [nb_label.value]
        metadata = {
            "media_type": task.media_type.value,
            "violation_candidates": violation_candidates,
            "risk_score_pre": nb_risk_score,
            "model_version": "text-nb-v1",
            "source": task.source,
        }
        embedding = self.embedding_model.embed_query(rewritten_text)
        return AuditFeature(
            metadata_scalar=metadata,
            description_desensitized=rewritten_text,
            embedding_vector=embedding,
        )

    def build_feature_from_image(
        self,
        task: AuditTask,
        image_description: str,
        quick_label_types: list[ViolationType],
        quick_label_score: float,
        image_base64: str | None = None,
        media_url: str | None = None,
        evidence: str = "",
    ) -> AuditFeature:
        """Build feature from image for audit pipeline.

        Args:
            task: 审核任务
            image_description: 图片的脱敏描述文本
            quick_label_types: CLIP 快速打标检测到的违规类型列表
            quick_label_score: 快速打标的风险分数
            image_base64: 可选的 Base64 编码图片（用于 CLIP 向量化）
            media_url: 可选的图片 URL（来自 MinIO）
            evidence: Evidence 风格的详细描述（可选）

        Returns:
            AuditFeature 实例，包含向量和元数据
        """
        violation_candidates = [vtype.value for vtype in quick_label_types]

        metadata = {
            "media_type": task.media_type.value,
            "violation_candidates": violation_candidates,
            "risk_score_pre": quick_label_score,
            "model_version": "image-clip-v1",
            "source": task.source,
            "media_url": media_url or "",
        }

        # 优先使用 CLIP 图像编码器生成向量（语义更准确）
        if image_base64 and self.image_embedder.ready:
            embedding = self.image_embedder.encode_from_base64(image_base64)
        else:
            # 降级：使用文本描述生成向量
            embedding = self.embedding_model.embed_query(image_description)

        return AuditFeature(
            metadata_scalar=metadata,
            description_desensitized=image_description,
            evidence=evidence,
            embedding_vector=embedding,
        )

    def search(self, feature: AuditFeature, top_k: int = 10) -> list[dict[str, Any]]:
        media_type = feature.metadata_scalar.get("media_type", "")
        violation_candidates = feature.metadata_scalar.get("violation_candidates", [])
        risk_pre = float(feature.metadata_scalar.get("risk_score_pre", 0.0))

        hybrid_filter = HybridFilter(media_type=media_type)

        # Broad match: include candidates + normal for comparison
        if violation_candidates:
            filter_labels = list(dict.fromkeys(violation_candidates + ["normal"]))
            hybrid_filter.violation_types = filter_labels

        # Prefer human-verified cases when risk is high
        if risk_pre >= 0.7:
            hybrid_filter.human_verified = True

        # Use OR expression for core conditions (media_type OR violation_type)
        # This improves recall by matching either condition
        filter_expr = hybrid_filter.to_or_expr()
        records = self.store.hybrid_search(
            query_vector=feature.embedding_vector,
            hybrid_filter=hybrid_filter,
            filter_expr=filter_expr,
            limit=top_k * 2,
        )

        # If strict filter yields too few results, fallback to pure vector search (no filter)
        if len(records) < 3:
            records = self.store.hybrid_search(
                query_vector=feature.embedding_vector,
                hybrid_filter=None,
                limit=top_k * 3,
            )
            logger.info("混合检索结果不足，降级为纯向量检索，返回 %d 条记录", len(records))

        # Filter low similarity results (CLIP cross-modal typically: 0.4+ is meaningfully related)
        MIN_VECTOR_SCORE = 0.4
        records = [r for r in records if r.get("score", 0) >= MIN_VECTOR_SCORE]

        target_violation = violation_candidates if violation_candidates else ""
        reranked = self.reranker.rerank(
            records,
            target_risk=risk_pre,
            target_violation=target_violation,
            top_k=top_k,
        )

        # Log matched cases for debugging
        if reranked:
            case_summaries = [
                f"{i+1}. violation_type={r.get('violation_type')}, "
                f"risk_score={r.get('risk_score', 0):.2f}, "
                f"vector_score={r.get('score', 0):.4f}, "
                f"human_verified={r.get('human_verified')}, "
                f"desc={str(r.get('description', ''))[:50]}..."
                for i, r in enumerate(reranked[:5])
            ]
            logger.info(
                "向量检索完成: media_type=%s, candidates=%s, raw_hits=%d, final_hits=%d\n  匹配案例:\n  %s",
                media_type,
                violation_candidates,
                len(records),
                len(reranked),
                "\n  ".join(case_summaries),
            )
        else:
            logger.info(
                "向量检索完成: media_type=%s, candidates=%s, raw_hits=%d, final_hits=0 (无匹配案例)",
                media_type,
                violation_candidates,
                len(records),
            )

        return reranked

    def upsert_case(
        self,
        task: AuditTask,
        feature: AuditFeature,
        violation_type: str,
        risk_score: float,
        model_version: str,
        human_verified: bool,
        created_at: str | None = None,
        media_url: str | None = None,
    ) -> None:
        """Upsert case to vector database.

        Args:
            task: 审核任务
            feature: 审核特征
            violation_type: 违规类型
            risk_score: 风险分数
            model_version: 模型版本
            human_verified: 是否人工验证
            created_at: 可选的创建时间
            media_url: 可选的图片 URL（来自 MinIO）
        """
        # 优先使用 feature 中的 media_url，其次使用参数传入的
        final_media_url = feature.metadata_scalar.get("media_url") or media_url

        self.store.upsert_case(
            record_id=str(uuid.uuid4()),
            task_id=task.task_id,
            media_type=task.media_type.value,
            violation_type=violation_type,
            risk_score=risk_score,
            source=task.source,
            created_at=created_at or feature.metadata_scalar.get("created_at", ""),
            model_version=model_version,
            human_verified=human_verified,
            description=feature.description_desensitized[:3900],
            evidence=feature.evidence[:3900] if feature.evidence else "",
            embedding=feature.embedding_vector,
            media_url=final_media_url,
        )
