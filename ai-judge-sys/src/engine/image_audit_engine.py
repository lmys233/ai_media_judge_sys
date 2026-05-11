"""图片审核引擎模块

编排图片审核的完整流程：
1. 图片预处理（resize + Base64）
2. CLIP 快速打标（多标签分类）
3. VL 生成图片描述（脱敏重写）
4. 向量检索 + 重排序
5. VL 最终审核判断
6. 置信度路由
7. 结果反馈（存向量库 + NB 增量训练）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.decision.gateway import ConfidenceGateway
from src.feature.image_desensitize import ImageDesensitizeProcessor, get_image_desensitize_processor
from src.feature.image_embedder import CLIPImageEmbedder
from src.models.contracts import (
    AuditDecision,
    AuditStatus,
    AuditTask,
    ViolationDetail,
    ViolationType,
)
from src.prelabel.image_quick_label import CLIPViolationClassifier, get_image_classifier
from src.reasoning.image_judge import ImageJudgeLLM, get_image_judge_llm
from src.retrieval.service import RetrievalService
from src.tool.image_utils import preprocess_image

logger = logging.getLogger(__name__)


@dataclass
class ImageAuditResult:
    """图片审核结果

    Attributes:
        decision: 审核决策
        quick_label_result: 快速打标结果
        image_description: 脱敏后的图片描述
        processed_base64: 处理后的 Base64 图片
    """

    decision: AuditDecision
    quick_label_types: list[ViolationType]
    quick_label_score: float
    image_description: str
    processed_base64: str
    image_url: str | None


class ImageAuditEngine:
    """图片审核引擎

    负责处理图片审核请求，协调各组件完成完整审核流程。
    """

    def __init__(self) -> None:
        """初始化图片审核引擎"""
        # 图片编码器
        self.image_embedder = CLIPImageEmbedder()

        # CLIP 快速打标
        self.quick_classifier = get_image_classifier()

        # 图片描述生成与脱敏
        self.image_desensitize = get_image_desensitize_processor()

        # VL 最终审核
        self.image_judge = get_image_judge_llm()

        # 向量检索服务
        self.retrieval = RetrievalService()
        self.retrieval.connect()

        # 置信度路由
        self.gateway = ConfidenceGateway()

        # 弱标签阈值（高置信度自动入库）
        self.weak_label_threshold = 0.9

        logger.info("图片审核引擎初始化完成")

    def _download_image(self, image_url: str, task_id: str) -> str | None:
        """从 URL 下载图片到本地临时文件

        Args:
            image_url: 图片 URL（支持 http://, https://, 或本地路径）
            task_id: 任务 ID（用于临时文件名）

        Returns:
            本地图片路径，下载失败返回 None
        """
        import os
        import tempfile
        from pathlib import Path
        from urllib.parse import urlparse

        try:
            import requests

            suffix = ".jpg"
            if image_url.lower().endswith(".png"):
                suffix = ".png"
            elif image_url.lower().endswith(".webp"):
                suffix = ".webp"

            target_dir = Path(tempfile.gettempdir()) / "audit_engine" / task_id
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_dir / f"source{suffix}"

            if image_url.startswith("http://") or image_url.startswith("https://"):
                # 公共读桶直接匿名下载，不需要认证
                resp = requests.get(image_url, timeout=30)

                if resp.status_code != 200:
                    logger.error("图片下载失败: url=%s, status=%d", image_url, resp.status_code)
                    return None

                target_file.write_bytes(resp.content)
            else:
                # 本地路径
                local_path = Path(image_url)
                if not local_path.exists():
                    logger.error("本地图片不存在: %s", image_url)
                    return None
                target_file.write_bytes(local_path.read_bytes())

            logger.info("图片下载成功: url=%s, path=%s", image_url, target_file)
            return str(target_file)

        except Exception as e:
            logger.error("图片下载异常: url=%s, error=%s", image_url, e)
            return None

    def audit_image(
        self,
        task: AuditTask,
        image_path: str | None = None,
        image_base64: str | None = None,
        image_url: str | None = None,
    ) -> ImageAuditResult:
        """执行图片审核

        完整流程：
        1. 图片预处理
        2. CLIP 快速打标
        3. VL 生成描述 + 脱敏
        4. 构建特征 + 向量检索
        5. VL 最终审核
        6. 置信度路由
        7. 结果反馈

        Args:
            task: 审核任务
            image_path: 图片文件路径（优先使用）
            image_base64: Base64 编码的图片
            image_url: 图片 URL（来自 MinIO）

        Returns:
            ImageAuditResult，包含审核决策和中间结果
        """
        try:
            # ========== 步骤1: 图片预处理 ==========
            if image_path:
                _, processed_base64 = preprocess_image(image_path)
            elif image_base64:
                processed_base64 = image_base64
            elif image_url:
                # 从 URL 下载图片
                downloaded_path = self._download_image(image_url, task.task_id)
                if not downloaded_path:
                    raise ValueError(f"图片下载失败: {image_url}")
                _, processed_base64 = preprocess_image(downloaded_path)
            else:
                raise ValueError("必须提供 image_path、image_base64 或 image_url")

            # ========== 步骤2: CLIP 快速打标 ==========
            quick_label = self.quick_classifier.predict_from_base64(processed_base64)
            quick_label_types = quick_label.violation_types
            quick_label_score = quick_label.risk_score

            logger.info(
                "图片快速打标完成: task_id=%s, types=%s, score=%.2f, source=%s",
                task.task_id,
                [v.value for v in quick_label_types],
                quick_label_score,
                quick_label.decision_source,
            )

            # ========== 步骤2.5: VL 生成描述 + Evidence（提前生成供 NB 分类） ==========
            storage_info = self.image_desensitize.describe_for_storage(processed_base64)
            image_description = storage_info["description"]
            image_evidence = storage_info["evidence"]

            logger.info(
                "图片描述生成完成: task_id=%s, desc_len=%d, evidence_len=%d",
                task.task_id,
                len(image_description),
                len(image_evidence),
            )

            # ========== 步骤2.6: NB 文本分类器辅助判断（基于 evidence 描述） ==========
            from src.prelabel.text_nb import NaiveBayesTextClassifier
            nb_classifier = NaiveBayesTextClassifier()
            nb_prediction = nb_classifier.predict(image_evidence)
            nb_label = nb_prediction.label
            nb_score = nb_prediction.risk_score

            # NB 也可能检测到违规
            nb_detected_types = [nb_label] if nb_label != ViolationType.NORMAL else []

            # 合并 CLIP 和 NB 的结果：如果两者不一致，取并集并降低置信度
            merged_types = list(set(quick_label_types + nb_detected_types))
            if nb_detected_types and nb_label.value not in [v.value for v in quick_label_types]:
                # CLIP 和 NB 结果不一致，NB 检出了 CLIP 没检出的类型
                quick_label_score = min(quick_label_score, nb_score * 0.8)  # 降低置信度
                logger.info(
                    "CLIP/NB 结果不一致，已合并: clip=%s, nb=%s, merged=%s",
                    [v.value for v in quick_label_types],
                    nb_label.value,
                    [v.value for v in merged_types],
                )

            # 使用合并后的结果
            quick_label_types = merged_types

            logger.info(
                "NB 辅助判断完成: task_id=%s, nb_label=%s, nb_score=%.2f, final_types=%s",
                task.task_id,
                nb_label.value,
                nb_score,
                [v.value for v in quick_label_types],
            )

            # ========== 步骤4: 构建特征 + 向量检索 ==========
            feature = self.retrieval.build_feature_from_image(
                task=task,
                image_description=image_description,
                quick_label_types=quick_label_types,
                quick_label_score=quick_label_score,
                image_base64=processed_base64,
                media_url=image_url,
                evidence=image_evidence,
            )

            retrieved_cases = self.retrieval.search(feature, top_k=20)

            # 批量获取案例详情
            if retrieved_cases:
                case_ids = [c.get("task_id") for c in retrieved_cases if c.get("task_id")]
                if case_ids:
                    from src.integration.judge_case_api import JudgeCaseApiClient

                    case_api = JudgeCaseApiClient()
                    full_cases = case_api.batch_get_cases(case_ids)
                    for case in retrieved_cases:
                        task_id = case.get("task_id")
                        if task_id and task_id in full_cases:
                            case["_full_data"] = full_cases[task_id]

            logger.info(
                "向量检索完成: task_id=%s, retrieved=%d",
                task.task_id,
                len(retrieved_cases),
            )

            # ========== 步骤5: VL 最终审核 ==========
            judge_result = self.image_judge.judge_with_image(
                task_id=task.task_id,
                trace_id=task.trace_id,
                image_base64=processed_base64,
                retrieved_cases=retrieved_cases,
                metadata=feature.metadata_scalar,
            )

            # 构建审核决策
            final_label_str = judge_result.get("final_label", "other")
            try:
                final_label = ViolationType(final_label_str)
            except ValueError:
                final_label = ViolationType.OTHER

            confidence = float(judge_result.get("confidence", 0.0))
            reason = judge_result.get("reason", "")

            # 构建违规详情
            violation_details = []
            for vd in judge_result.get("violation_details", []):
                try:
                    vtype = ViolationType(vd.get("violation_type", "other"))
                except ValueError:
                    vtype = ViolationType.OTHER

                violation_details.append(
                    ViolationDetail(
                        violation_type=vtype,
                        confidence=float(vd.get("confidence", 0.0)),
                        evidence=vd.get("evidence", []),
                        reason=vd.get("reason", ""),
                    )
                )

            decision = AuditDecision(
                trace_id=task.trace_id,
                task_id=task.task_id,
                final_label=final_label,
                confidence=confidence,
                status=AuditStatus.REVIEWING,
                reason=reason,
                decision_path=["image_preprocess", "clip_quick_label", "vl_describe", "vector_search", "vl_judge"],
                needs_manual=False,
                violation_details=violation_details,
            )

            # ========== 步骤6: 置信度路由 ==========
            route = self.gateway.route(decision.confidence)

            logger.info(
                "图片审核完成: task_id=%s, final_label=%s, confidence=%.2f, route=%s",
                task.task_id,
                final_label.value,
                confidence,
                route,
            )

            if decision.confidence >= 0.8:
                if final_label == ViolationType.NORMAL:
                    decision.status = AuditStatus.AUTO_APPROVED
                else:
                    decision.status = AuditStatus.AUTO_REJECTED
                decision.needs_manual = False
                decision.decision_path.append("auto_decision")

                # 存入向量库
                self._upsert_image_case(task, feature, decision, image_url)

                # 追加到 NB 训练语料（使用 evidence 详细描述，让 NB 学习更具体的特征）
                self._append_nb_sample(image_evidence, final_label.value, source="image_audit")

            elif decision.confidence <= 0.2:
                decision.status = AuditStatus.NEEDS_MANUAL
                decision.needs_manual = True
                decision.decision_path.append("low_confidence_manual")

            else:
                decision.status = AuditStatus.NEEDS_MANUAL
                decision.needs_manual = True
                decision.decision_path.append("medium_confidence_manual")

            return ImageAuditResult(
                decision=decision,
                quick_label_types=quick_label_types,
                quick_label_score=quick_label_score,
                image_description=image_description,
                processed_base64=processed_base64,
                image_url=image_url,
            )

        except Exception as e:
            logger.error("图片审核异常: task_id=%s, error=%s", task.task_id, e)
            decision = AuditDecision(
                trace_id=task.trace_id,
                task_id=task.task_id,
                final_label=ViolationType.OTHER,
                confidence=0.0,
                status=AuditStatus.FAILED,
                reason=f"审核异常: {str(e)[:100]}",
                decision_path=["image_audit_error"],
                needs_manual=True,
                violation_details=[],
            )
            return ImageAuditResult(
                decision=decision,
                quick_label_types=[ViolationType.OTHER],
                quick_label_score=0.5,
                image_description="",
                processed_base64=image_base64 or "",
                image_url=image_url,
            )

    def _upsert_image_case(
        self,
        task: AuditTask,
        feature: Any,
        decision: AuditDecision,
        image_url: str | None,
    ) -> None:
        """将图片审核结果存入向量数据库"""
        try:
            from datetime import datetime

            created_at = datetime.utcnow().isoformat()

            if decision.violation_details:
                for vd in decision.violation_details:
                    self.retrieval.upsert_case(
                        task=task,
                        feature=feature,
                        violation_type=vd.violation_type.value,
                        risk_score=vd.confidence,
                        model_version="image-audit-v1",
                        human_verified=False,
                        created_at=created_at,
                        media_url=image_url,
                    )
                    logger.info(
                        "图片案例已存入向量库: task_id=%s, violation=%s, score=%.2f",
                        task.task_id,
                        vd.violation_type.value,
                        vd.confidence,
                    )
            else:
                self.retrieval.upsert_case(
                    task=task,
                    feature=feature,
                    violation_type=decision.final_label.value,
                    risk_score=decision.confidence,
                    model_version="image-audit-v1",
                    human_verified=False,
                    created_at=created_at,
                    media_url=image_url,
                )

        except Exception as e:
            logger.error("图片案例存入向量库失败: task_id=%s, error=%s", task.task_id, e)

    def _append_nb_sample(
        self,
        text: str,
        label: str,
        source: str = "image_audit",
    ) -> None:
        """将图片描述追加到 NB 训练语料

        图片审核完成后，将图片描述作为文本样本追加到 NB 分类器的训练语料中，
        使 NB 能够从图片审核中学习到"某种图片描述对应某种违规类型"。
        """
        try:
            from src.prelabel.text_nb import NaiveBayesTextClassifier

            nb_classifier = NaiveBayesTextClassifier()
            nb_classifier.append_training_sample(
                text=text,
                label=label,
                source=source,
                verified=False,
            )
            logger.info("NB 训练样本已追加: label=%s, source=%s", label, source)

        except Exception as e:
            logger.error("NB 样本追加失败: label=%s, error=%s", label, e)
