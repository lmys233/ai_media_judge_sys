from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, MilvusException, connections, utility

logger = logging.getLogger(__name__)

_HNSW_INDEX = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 256},
}

_SCALAR_INDEX_FIELDS = ["media_type", "violation_type", "source", "human_verified", "risk_score"]

_OUTPUT_FIELDS = [
    "task_id",
    "media_type",
    "violation_type",
    "risk_score",
    "description",
    "evidence",
    "source",
    "model_version",
    "human_verified",
    "created_at",
    "media_url",
]


@dataclass
class HybridFilter:
    """Build a Milvus boolean expression for scalar pre-filtering."""

    media_type: str = ""
    violation_types: list[str] = field(default_factory=list)
    source: str = ""
    human_verified: bool | None = None
    risk_score_min: float | None = None
    risk_score_max: float | None = None

    def to_expr(self) -> str:
        clauses: list[str] = []
        if self.media_type:
            clauses.append(f'media_type == "{self.media_type}"')
        if self.violation_types:
            if len(self.violation_types) == 1:
                clauses.append(f'violation_type == "{self.violation_types[0]}"')
            else:
                in_list = ", ".join(f'"{v}"' for v in self.violation_types)
                clauses.append(f"violation_type in [{in_list}]")
        if self.source:
            clauses.append(f'source == "{self.source}"')
        if self.human_verified is not None:
            clauses.append(f"human_verified == {'true' if self.human_verified else 'false'}")
        if self.risk_score_min is not None:
            clauses.append(f"risk_score >= {self.risk_score_min}")
        if self.risk_score_max is not None:
            clauses.append(f"risk_score <= {self.risk_score_max}")
        return " and ".join(clauses)

    def to_or_expr(self) -> str:
        """Build expression with OR between media_type and violation_types.

        Use this for high-recall retrieval where matching either condition is acceptable.
        Additional filters (source, human_verified, risk_score) are still AND-ed.
        """
        core_clauses: list[str] = []
        other_clauses: list[str] = []

        if self.media_type:
            core_clauses.append(f'media_type == "{self.media_type}"')
        if self.violation_types:
            if len(self.violation_types) == 1:
                core_clauses.append(f'violation_type == "{self.violation_types[0]}"')
            else:
                in_list = ", ".join(f'"{v}"' for v in self.violation_types)
                core_clauses.append(f"violation_type in [{in_list}]")

        if self.source:
            other_clauses.append(f'source == "{self.source}"')
        if self.human_verified is not None:
            other_clauses.append(f"human_verified == {'true' if self.human_verified else 'false'}")
        if self.risk_score_min is not None:
            other_clauses.append(f"risk_score >= {self.risk_score_min}")
        if self.risk_score_max is not None:
            other_clauses.append(f"risk_score <= {self.risk_score_max}")

        # Build final expression
        parts: list[str] = []
        if len(core_clauses) > 1:
            parts.append("(" + " OR ".join(core_clauses) + ")")
        elif len(core_clauses) == 1:
            parts.append(core_clauses[0])
        # else: no core conditions, skip

        parts.extend(other_clauses)

        return " and ".join(parts) if parts else ""


class MilvusAuditStore:
    def __init__(self, collection_name: str = "audit_cases", dim: int = 512) -> None:
        self.collection_name = collection_name
        self.dim = dim
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", "19530")
        self.collection: Collection | None = None

    def connect(self) -> None:
        connections.connect(alias="default", host=self.host, port=self.port)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not utility.has_collection(self.collection_name):
            self._create_collection()
        else:
            self.collection = Collection(self.collection_name)
            self._ensure_indexes()
            self.collection.load()
        logger.info("Milvus集合已就绪 名称=%s", self.collection_name)

    def _create_collection(self) -> None:
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=128, is_primary=True, auto_id=False),
            FieldSchema(name="task_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="media_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="violation_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="risk_score", dtype=DataType.FLOAT),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="model_version", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="human_verified", dtype=DataType.BOOL),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="evidence", dtype=DataType.VARCHAR, max_length=4096, nullable=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            # 图片 URL 字段（可选，用于存储 MinIO 中的原始图片路径）
            FieldSchema(name="media_url", dtype=DataType.VARCHAR, max_length=512, nullable=True),
        ]
        schema = CollectionSchema(fields=fields, description="audit moderation cases")
        self.collection = Collection(name=self.collection_name, schema=schema)
        self._ensure_indexes()
        self.collection.load()

    def _ensure_indexes(self) -> None:
        assert self.collection is not None
        existing = {idx.field_name for idx in self.collection.indexes}

        if "embedding" not in existing:
            self.collection.create_index(field_name="embedding", index_params=_HNSW_INDEX)
            logger.info("已创建 HNSW 向量索引 (embedding)")

        for scalar_field in _SCALAR_INDEX_FIELDS:
            if scalar_field not in existing:
                try:
                    self.collection.create_index(
                        field_name=scalar_field,
                        index_name=f"idx_{scalar_field}",
                        index_params={"index_type": "STL_SORT"} if scalar_field == "risk_score" else {},
                    )
                    logger.info("已创建标量索引: %s", scalar_field)
                except MilvusException as exc:
                    logger.warning("标量索引创建跳过 字段=%s: %s", scalar_field, exc)

    # ── Write ────────────────────────────────────────────────────

    def upsert_case(
        self,
        record_id: str,
        task_id: str,
        media_type: str,
        violation_type: str,
        risk_score: float,
        source: str,
        created_at: str,
        model_version: str,
        human_verified: bool,
        description: str,
        embedding: list[float],
        media_url: str | None = None,
        evidence: str = "",
    ) -> None:
        if self.collection is None:
            raise RuntimeError("Milvus collection not initialized")
        data = [
            [record_id],
            [task_id],
            [media_type],
            [violation_type],
            [risk_score],
            [source],
            [created_at],
            [model_version],
            [human_verified],
            [description],
            [evidence] if evidence else [""],
            [embedding],
            [media_url] if media_url else [""],
        ]
        self.collection.insert(data)
        self.collection.flush()

    def batch_insert(
        self,
        records: list[dict],
        flush: bool = True,
    ) -> int:
        """批量插入记录，提高写入效率"""
        if self.collection is None:
            raise RuntimeError("Milvus collection not initialized")
        if not records:
            return 0

        ids = [r["record_id"] for r in records]
        task_ids = [r["task_id"] for r in records]
        media_types = [r["media_type"] for r in records]
        violation_types = [r["violation_type"] for r in records]
        risk_scores = [r["risk_score"] for r in records]
        sources = [r["source"] for r in records]
        created_ats = [r["created_at"] for r in records]
        model_versions = [r["model_version"] for r in records]
        human_verifieds = [r["human_verified"] for r in records]
        descriptions = [r["description"] for r in records]
        evidences = [r.get("evidence", "") or "" for r in records]
        embeddings = [r["embedding"] for r in records]
        media_urls = [r.get("media_url", "") or "" for r in records]

        data = [
            ids,
            task_ids,
            media_types,
            violation_types,
            risk_scores,
            sources,
            created_ats,
            model_versions,
            human_verifieds,
            descriptions,
            evidences,
            embeddings,
            media_urls,
        ]
        self.collection.insert(data)
        if flush:
            self.collection.flush()
        return len(records)

    # ── Hybrid search ────────────────────────────────────────────

    def hybrid_search(
        self,
        query_vector: list[float],
        hybrid_filter: HybridFilter | None = None,
        filter_expr: str | None = None,
        limit: int = 10,
        ef: int = 128,
    ) -> list[dict[str, Any]]:
        if self.collection is None:
            return []
        # Use custom filter_expr if provided, otherwise fall back to HybridFilter.to_expr()
        expr = filter_expr if filter_expr is not None else (hybrid_filter.to_expr() if hybrid_filter else "")
        try:
            result = self.collection.search(
                data=[query_vector],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"ef": ef}},
                expr=expr,
                limit=limit,
                output_fields=_OUTPUT_FIELDS,
            )
        except MilvusException:
            logger.exception("Milvus 混合检索失败 过滤条件=%s", expr)
            return []
        return self._hits_to_dicts(result)

    def search(self, query_vector: list[float], filters: str = "", limit: int = 10) -> list[dict[str, Any]]:
        """Legacy search kept for backward compatibility."""
        if self.collection is None:
            return []
        try:
            result = self.collection.search(
                data=[query_vector],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                expr=filters,
                limit=limit,
                output_fields=_OUTPUT_FIELDS,
            )
        except MilvusException:
            return []
        return self._hits_to_dicts(result)

    @staticmethod
    def _hits_to_dicts(result) -> list[dict[str, Any]]:  # noqa: ANN001
        records: list[dict[str, Any]] = []
        output_fields = [
            "task_id",
            "media_type",
            "violation_type",
            "risk_score",
            "description",
            "source",
            "model_version",
            "human_verified",
            "created_at",
            "media_url",
        ]
        for hit in result[0]:
            row: dict[str, Any] = {}
            entity = hit.entity
            if entity:
                for field in output_fields:
                    try:
                        row[field] = entity.get(field)
                    except Exception:  # noqa: BLE001
                        row[field] = None
            row["score"] = float(hit.distance)
            records.append(row)
        return records

    # ── Stats ────────────────────────────────────────────────────

    def count(self) -> int:
        if self.collection is None:
            return 0
        return self.collection.num_entities
