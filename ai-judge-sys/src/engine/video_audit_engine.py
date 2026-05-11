"""视频审核引擎模块

编排视频审核的完整流程：
1. 关键帧提取（场景检测 + 信息量评分）
2. CLIP 快速打标（每帧独立分类）
3. 视频帧汇总（多帧结果汇总）
4. VL 生成视频帧描述（每帧描述拼接）
5. 向量检索 + 重排序
6. LLM 最终审核判断
7. 置信度路由
8. 结果反馈（存向量库 + NB 增量训练）

复用图片审核链路的核心组件。
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
from src.prelabel.image_mobilenet import MobileNetImageClassifier
from src.prelabel.video_summary import VideoFrameSummarizer
from src.reasoning.judge import JudgeLLM, build_default_judge_llm
from src.retrieval.service import RetrievalService
from src.tool.image_utils import preprocess_image
from src.tool.keyframe_extractor import KeyframeExtractor

logger = logging.getLogger(__name__)


@dataclass
class VideoAuditResult:
    """视频审核结果"""
    decision: AuditDecision
    quick_label_types: list[ViolationType]
    quick_label_score: float
    frame_descriptions: list[str]
    keyframe_paths: list[str]


class VideoAuditEngine:
    """视频审核引擎

    负责处理视频审核请求，协调各组件完成完整审核流程。
    """

    def __init__(self) -> None:
        """初始化视频审核引擎"""
        # 关键帧提取器
        self.keyframe_extractor = KeyframeExtractor(max_keyframes=16)

        # 图片编码器（用于向量检索）
        self.image_embedder = CLIPImageEmbedder()

        # CLIP 快速打标（每帧独立分类）
        self.image_classifier = MobileNetImageClassifier()

        # 视频帧汇总
        self.frame_summarizer = VideoFrameSummarizer()

        # 图片描述生成与脱敏（VL模型）
        self.image_desensitize = get_image_desensitize_processor()

        # LLM 判决
        self.judge_llm = build_default_judge_llm()
        self.judge = JudgeLLM(self.judge_llm)

        # 向量检索服务
        self.retrieval = RetrievalService()
        self.retrieval.connect()

        # 置信度路由
        self.gateway = ConfidenceGateway()

        logger.info("视频审核引擎初始化完成")

    def _download_video(self, video_url: str, task_id: str) -> str | None:
        """从 URL 下载视频到本地临时文件

        Args:
            video_url: 视频 URL（支持 http://, https://, 或本地路径）
            task_id: 任务 ID（用于临时文件名）

        Returns:
            本地视频路径，下载失败返回 None
        """
        import tempfile
        from pathlib import Path
        import requests

        try:
            suffix = ".mp4"
            if video_url.lower().endswith(".avi"):
                suffix = ".avi"
            elif video_url.lower().endswith(".mov"):
                suffix = ".mov"

            target_dir = Path(tempfile.gettempdir()) / "audit_engine" / task_id
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_dir / f"source{suffix}"

            if video_url.startswith("http://") or video_url.startswith("https://"):
                resp = requests.get(video_url, timeout=60)
                if resp.status_code != 200:
                    logger.error("视频下载失败: url=%s, status=%d", video_url, resp.status_code)
                    return None
                target_file.write_bytes(resp.content)
            else:
                # 本地路径
                local_path = Path(video_url)
                if not local_path.exists():
                    logger.error("本地视频不存在: %s", video_url)
                    return None
                target_file.write_bytes(local_path.read_bytes())

            logger.info("视频下载成功: url=%s, path=%s", video_url, target_file)
            return str(target_file)

        except Exception as e:
            logger.error("视频下载异常: url=%s, error=%s", video_url, e)
            return None

    def audit_video(
        self,
        task: AuditTask,
        video_path: str | None = None,
        video_url: str | None = None,
    ) -> VideoAuditResult:
        """执行视频审核

        完整流程：
        1. 视频下载/定位
        2. 关键帧提取
        3. 每帧 CLIP 快速打标
        4. 视频帧汇总
        5. VL 生成帧描述 + 脱敏
        6. 构建特征 + 向量检索
        7. LLM 最终审核
        8. 置信度路由
        9. 结果反馈

        Args:
            task: 审核任务
            video_path: 视频文件路径（优先使用）
            video_url: 视频 URL

        Returns:
            VideoAuditResult，包含审核决策和中间结果
        """
        try:
            # ========== 步骤1: 获取视频路径 ==========
            if video_path is None and video_url:
                video_path = self._download_video(video_url, task.task_id)
                if not video_path:
                    raise ValueError(f"视频下载失败: {video_url}")

            if not video_path:
                raise ValueError("必须提供 video_path 或 video_url")

            # ========== 步骤2: 关键帧提取 ==========
            keyframe_paths = self.keyframe_extractor.extract_keyframes(video_path, task.task_id)
            if not keyframe_paths:
                raise ValueError("关键帧提取失败，视频可能损坏或无法读取")

            logger.info(
                "关键帧提取完成: task_id=%s, frames=%d",
                task.task_id,
                len(keyframe_paths),
            )

            # ========== 步骤3: 每帧 CLIP 快速打标 ==========
            frame_labels: list[ViolationType] = []
            frame_scores: list[float] = []
            frame_evidence: list[str] = []

            for frame_path in keyframe_paths:
                pred = self.image_classifier.predict(frame_path)
                frame_labels.append(pred.label)
                frame_scores.append(pred.risk_score)
                frame_evidence.extend(pred.evidence)

            # 汇总多帧结果
            final_label, final_score = self.frame_summarizer.infer_label(frame_labels, frame_scores)

            # 合并所有帧的违规类型
            quick_label_types = list(set(frame_labels))
            quick_label_score = final_score

            logger.info(
                "视频帧快速打标完成: task_id=%s, types=%s, score=%.2f",
                task.task_id,
                [v.value for v in quick_label_types],
                quick_label_score,
            )

            # ========== 步骤4: VL 生成帧描述 + 脱敏 ==========
            frame_descriptions: list[str] = []
            for frame_path in keyframe_paths[:8]:  # 最多处理8帧，避免token超限
                try:
                    _, base64 = preprocess_image(frame_path)
                    desc_info = self.image_desensitize.describe_for_storage(base64)
                    frame_descriptions.append(desc_info["description"])
                except Exception as e:
                    logger.warning("帧描述生成失败: path=%s, error=%s", frame_path, e)
                    frame_descriptions.append("无法识别帧内容")

            # 拼接所有帧描述
            combined_description = "; ".join(frame_descriptions) if frame_descriptions else "无帧描述"
            evidence_text = "; ".join(frame_evidence[:20]) if frame_evidence else combined_description

            logger.info(
                "视频帧描述生成完成: task_id=%s, frames=%d, desc_len=%d",
                task.task_id,
                len(frame_descriptions),
                len(combined_description),
            )

            # ========== 步骤5: NB 文本分类器辅助判断 ==========
            from src.prelabel.text_nb import NaiveBayesTextClassifier
            nb_classifier = NaiveBayesTextClassifier()
            nb_prediction = nb_classifier.predict(evidence_text)
            nb_label = nb_prediction.label
            nb_score = nb_prediction.risk_score

            nb_detected_types = [nb_label] if nb_label != ViolationType.NORMAL else []

            # 合并 CLIP 和 NB 的结果
            merged_types = list(set(quick_label_types + nb_detected_types))
            if nb_detected_types and nb_label.value not in [v.value for v in quick_label_types]:
                quick_label_score = min(quick_label_score, nb_score * 0.8)
                logger.info(
                    "CLIP/NB 结果不一致，已合并: clip=%s, nb=%s",
                    [v.value for v in quick_label_types],
                    nb_label.value,
                )

            quick_label_types = merged_types

            # ========== 步骤6: 构建特征 + 向量检索 ==========
            # 使用首帧图片做向量检索
            first_frame_base64 = None
            if keyframe_paths:
                try:
                    _, first_frame_base64 = preprocess_image(keyframe_paths[0])
                except Exception as e:
                    logger.warning("首帧预处理失败: error=%s", e)

            feature = self.retrieval.build_feature_from_image(
                task=task,
                image_description=combined_description,
                quick_label_types=quick_label_types,
                quick_label_score=quick_label_score,
                image_base64=first_frame_base64,
                media_url=video_url,
                evidence=evidence_text,
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

            # ========== 步骤7: LLM 最终审核 ==========
            decision = self.judge.judge(
                task_id=task.task_id,
                trace_id=task.trace_id,
                metadata=feature.metadata_scalar,
                prelabel_summary=combined_description,
                retrieved_cases=retrieved_cases,
            )

            # 构建审核决策
            try:
                final_label = ViolationType(decision.final_label.value)
            except ValueError:
                final_label = ViolationType.OTHER

            audit_decision = AuditDecision(
                trace_id=task.trace_id,
                task_id=task.task_id,
                final_label=final_label,
                confidence=decision.confidence,
                status=AuditStatus.REVIEWING,
                reason=decision.reason,
                decision_path=[
                    "video_keyframe_extract",
                    "clip_quick_label",
                    "frame_summarize",
                    "vl_describe",
                    "vector_search",
                    "llm_judge",
                ],
                needs_manual=False,
                violation_details=decision.violation_details or [],
            )

            # ========== 步骤8: 置信度路由 ==========
            route = self.gateway.route(audit_decision.confidence)

            logger.info(
                "视频审核完成: task_id=%s, final_label=%s, confidence=%.2f, route=%s",
                task.task_id,
                final_label.value,
                audit_decision.confidence,
                route,
            )

            if audit_decision.confidence >= 0.8:
                if final_label == ViolationType.NORMAL:
                    audit_decision.status = AuditStatus.AUTO_APPROVED
                else:
                    audit_decision.status = AuditStatus.AUTO_REJECTED
                audit_decision.needs_manual = False
                audit_decision.decision_path.append("auto_decision")

                # 存入向量库
                self._upsert_video_case(task, feature, audit_decision, video_url)

                # 追加到 NB 训练语料
                self._append_nb_sample(evidence_text, final_label.value, source="video_audit")

            elif audit_decision.confidence <= 0.2:
                audit_decision.status = AuditStatus.NEEDS_MANUAL
                audit_decision.needs_manual = True
                audit_decision.decision_path.append("low_confidence_manual")

            else:
                audit_decision.status = AuditStatus.NEEDS_MANUAL
                audit_decision.needs_manual = True
                audit_decision.decision_path.append("medium_confidence_manual")

            return VideoAuditResult(
                decision=audit_decision,
                quick_label_types=quick_label_types,
                quick_label_score=quick_label_score,
                frame_descriptions=frame_descriptions,
                keyframe_paths=keyframe_paths,
            )

        except Exception as e:
            logger.error("视频审核异常: task_id=%s, error=%s", task.task_id, e)
            decision = AuditDecision(
                trace_id=task.trace_id,
                task_id=task.task_id,
                final_label=ViolationType.OTHER,
                confidence=0.0,
                status=AuditStatus.FAILED,
                reason=f"视频审核异常: {str(e)[:100]}",
                decision_path=["video_audit_error"],
                needs_manual=True,
                violation_details=[],
            )
            return VideoAuditResult(
                decision=decision,
                quick_label_types=[ViolationType.OTHER],
                quick_label_score=0.5,
                frame_descriptions=[],
                keyframe_paths=[],
            )

    def _upsert_video_case(
        self,
        task: AuditTask,
        feature: Any,
        decision: AuditDecision,
        video_url: str | None,
    ) -> None:
        """将视频审核结果存入向量数据库"""
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
                        model_version="video-audit-v1",
                        human_verified=False,
                        created_at=created_at,
                        media_url=video_url,
                    )
            else:
                self.retrieval.upsert_case(
                    task=task,
                    feature=feature,
                    violation_type=decision.final_label.value,
                    risk_score=decision.confidence,
                    model_version="video-audit-v1",
                    human_verified=False,
                    created_at=created_at,
                    media_url=video_url,
                )

            logger.info(
                "视频案例已存入向量库: task_id=%s, violation=%s, score=%.2f",
                task.task_id,
                decision.final_label.value,
                decision.confidence,
            )

        except Exception as e:
            logger.error("视频案例存入向量库失败: task_id=%s, error=%s", task.task_id, e)

    def _append_nb_sample(
        self,
        text: str,
        label: str,
        source: str = "video_audit",
    ) -> None:
        """将视频描述追加到 NB 训练语料"""
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
