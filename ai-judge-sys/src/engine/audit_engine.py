from __future__ import annotations

import logging
import os
from typing import Any

from src.decision.extended_gateway import ExtendedConfidenceGateway
from src.feature.desensitize import DesensitizeProcessor
from src.integration.judge_case_api import JudgeCaseApiClient
from src.integration.manual_bridge import ManualReviewBridge
from src.models.contracts import (
    AuditDecision,
    AuditResultMessage,
    AuditStatus,
    AuditTask,
    ViolationType,
)
from src.prelabel.pipeline import PreLabelPipeline
from src.prelabel.text_nb import MultiLabelPrediction
from src.reasoning.judge import JudgeLLM, build_default_judge_llm
from src.reasoning.multi_agent.decision_maker import ReActDecisionMaker
from src.reasoning.multi_agent.evidence_assistant import CoTEvidenceAssistant
from src.reasoning.multi_agent.logic_assistant import CoTLogicAssistant
from src.reasoning.multi_agent.reentrancy import ReentrancyManager, ReviewSessionStorage
from src.reasoning.multi_agent.supervisor import MetaReviewSupervisor
from src.reasoning.react_reviewer import CoTAuditor, ReActReviewer
from src.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)


class AuditEngine:
    def __init__(self, manual_bridge: ManualReviewBridge | None = None) -> None:
        self.prelabel = PreLabelPipeline()
        # Create shared LLM instance for both rewrite and judge
        shared_llm = build_default_judge_llm()
        self.desensitize = DesensitizeProcessor(llm=shared_llm)
        self.retrieval = RetrievalService()
        # Extended gateway with 3-layer routing
        self.gateway = ExtendedConfidenceGateway()
        self.judge = JudgeLLM(shared_llm)

        # Legacy review (kept for backward compatibility)
        self.review = ReActReviewer(
            react_llm=None,
            logic_auditor=CoTAuditor(focus="logic"),
            evidence_auditor=CoTAuditor(focus="evidence"),
            max_retry=3,
        )

        # Multi-agent review components
        self._setup_multi_agent_review(shared_llm)

        self.manual_bridge = manual_bridge
        self._media_tool = None
        self.case_api = JudgeCaseApiClient()
        # 图片审核引擎
        self._image_engine = None
        # 视频审核引擎
        self._video_engine = None
        # 弱标签阈值
        self.weak_label_threshold = float(os.getenv("WEAK_LABEL_THRESHOLD", "0.9"))
        # 待人工处理的上下文
        self._pending_manual_context: dict[str, tuple[AuditTask, Any]] = {}

    def _setup_multi_agent_review(self, shared_llm: Any) -> None:
        """Setup multi-agent review components"""
        # Decision maker (ReAct)
        self.decision_maker = ReActDecisionMaker(
            llm=shared_llm,
            max_retries=3,
        )
        # Logic assistant (CoT)
        self.logic_assistant = CoTLogicAssistant(llm=shared_llm)
        # Evidence assistant (CoT)
        self.evidence_assistant = CoTEvidenceAssistant(llm=shared_llm)
        # Supervisor
        self.supervisor = MetaReviewSupervisor(
            decision_maker=self.decision_maker,
            logic_assistant=self.logic_assistant,
            evidence_assistant=self.evidence_assistant,
            gateway=self.gateway,
        )
        # Reentrancy manager
        self.reentrancy_manager = ReentrancyManager(
            storage=ReviewSessionStorage(),
        )
        logger.info("Multi-agent review components initialized")

    @property
    def media_tool(self):
        """Lazy load media tool to avoid cv2 import failure."""
        if self._media_tool is None:
            from src.tool.MediaParseTool import MediaParseTool
            self._media_tool = MediaParseTool()
        return self._media_tool

    def bootstrap(self) -> None:
        self.retrieval.connect()

    def handle_task(self, task: AuditTask) -> AuditDecision:
        parsed = self.media_tool._run(task.media_url, task.media_type.value, task.task_id)
        if parsed.get("status") != "success":
            return AuditDecision(
                trace_id=task.trace_id,
                task_id=task.task_id,
                final_label=ViolationType.OTHER,
                confidence=0.0,
                status=AuditStatus.FAILED,
                reason=parsed.get("msg", "parse_failed"),
                decision_path=["parse_failed"],
                needs_manual=True,
            )

        prelabel_result, summary = self.prelabel.run(parsed)
        desensitized_summary = self.desensitize.sanitize(summary)
        feature = self.retrieval.build_feature(task, prelabel_result, desensitized_summary)
        retrieved_cases = self.retrieval.search(feature)

        decision = self.judge.judge(
            task_id=task.task_id,
            trace_id=task.trace_id,
            metadata=feature.metadata_scalar,
            prelabel_summary=desensitized_summary,
            retrieved_cases=retrieved_cases,
        )

        route = self.gateway.route(decision.confidence)
        if route == "review":
            decision = self.review.review(decision, feature.metadata_scalar)
            route = self.gateway.route(decision.confidence)

        if route == "manual" or decision.needs_manual:
            decision.needs_manual = True
            decision.status = AuditStatus.NEEDS_MANUAL
            decision.decision_path.append("to_manual_queue")
            self._pending_manual_context[task.task_id] = (task, feature)
            if self.manual_bridge is not None:
                self.manual_bridge.send_to_manual(task, decision, feature)
            return decision

        if decision.final_label.value == "normal":
            decision.status = AuditStatus.AUTO_APPROVED
        else:
            decision.status = AuditStatus.AUTO_REJECTED
        decision.needs_manual = False
        decision.decision_path.append("auto_decision")

        # cold-start bootstrap policy: admit highly confident cases
        if decision.confidence >= self.weak_label_threshold:
            self.retrieval.upsert_case(
                task=task,
                feature=feature,
                violation_type=decision.final_label.value,
                risk_score=decision.confidence,
                model_version=feature.metadata_scalar.get("model_version", "judge-v1"),
                human_verified=False,
            )
            decision.decision_path.append("weak_label_upserted")

        return decision

    def handle_manual_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.manual_bridge is None:
            return {"status": "ignored", "reason": "manual_bridge_disabled"}
        result = self.manual_bridge.parse_manual_result(payload)
        if result.task_id in self._pending_manual_context:
            task, feature = self._pending_manual_context.pop(result.task_id)
            self.retrieval.upsert_case(
                task=task,
                feature=feature,
                violation_type=result.final_label.value,
                risk_score=1.0 if result.approved else 0.5,
                model_version="human-review-v1",
                human_verified=True,
            )
        return {
            "status": "ok",
            "task_id": result.task_id,
            "trace_id": result.trace_id,
            "final_label": result.final_label.value,
            "human_verified": True,
        }

    def handle_ai_auto_task(self, task: AuditTask, original_text: str) -> AuditDecision:
        """Handle AI auto audit task without manual review.

        Flow: NB multi-label quick label -> Text rewrite -> Vector retrieval -> Rerank -> LLM judge -> Route
        """
        # 1. NB multi-label quick labeling (no media parsing needed for text)
        nb_multi = self.prelabel.text_classifier.predict_multi(original_text)

        # Log all detected violation types
        all_labels = [p.label.value for p in nb_multi.predictions] if nb_multi.predictions else []
        all_scores = {p.label.value: p.risk_score for p in nb_multi.predictions} if nb_multi.predictions else {}
        logger.info(
            "NB multi-label quick label: task_id=%s, labels=%s, scores=%s, source=%s",
            task.task_id,
            all_labels,
            all_scores,
            nb_multi.decision_source,
        )

        # Get top violation type for rewrite hint
        top_label = nb_multi.top_label
        top_score = nb_multi.overall_risk_score

        # 2. Text rewrite: rule replacement + LLM semantic rewrite
        # For multi-label, we rewrite the original text and keep all violation types
        rule_replaced = self.desensitize.sanitize(original_text)
        rewritten = self.desensitize.sanitize_for_rag(rule_replaced, all_labels)
        logger.info("Text rewritten: task_id=%s, original_len=%d, rewritten_len=%d, violation_types=%s",
                    task.task_id, len(original_text), len(rewritten), all_labels)

        # 3. Build feature for retrieval (using top violation type)
        feature = self.retrieval.build_feature_from_text(
            task, rewritten, top_label, top_score, all_labels
        )

        # 4. Vector retrieval
        retrieved_cases = self.retrieval.search(feature, top_k=20)

        # 4.1 Batch fetch full case details (including review_reason) from backend
        retrieved_context = {}
        if retrieved_cases:
            case_ids = [c.get("task_id") for c in retrieved_cases if c.get("task_id")]
            if case_ids:
                full_cases = self.case_api.batch_get_cases(case_ids)
                # Merge full case data into retrieved_cases
                for case in retrieved_cases:
                    task_id = case.get("task_id")
                    if task_id and task_id in full_cases:
                        case["_full_data"] = full_cases[task_id]
                logger.info(
                    "Batch fetched %d case details for task_id=%s",
                    len(full_cases),
                    task.task_id,
                )

        # Build context for multi-agent review
        context = {
            "retrieved_cases": retrieved_cases,
            "metadata": feature.metadata_scalar,
            "rewritten_text": rewritten,
            "all_labels": all_labels,
        }

        # 5. LLM audit judgment with multi-label context
        decision = self.judge.judge(
            task_id=task.task_id,
            trace_id=task.trace_id,
            metadata=feature.metadata_scalar,
            prelabel_summary=rewritten,
            retrieved_cases=retrieved_cases,
        )
        decision.decision_path.append("nb_multi_quick")
        decision.decision_path.append("rewrite")
        decision.decision_path.append("vector_search")
        decision.decision_path.append("rerank")
        decision.decision_path.append("llm_judge")

        # 6. Extended confidence routing with multi-agent review
        route = self.gateway.route_initial(decision.confidence)
        logger.info(
            "AI auto audit: task_id=%s, final_label=%s, confidence=%.2f, route=%s, violation_details=%s",
            task.task_id,
            decision.final_label.value,
            decision.confidence,
            route,
            [(v.violation_type.value, v.confidence) for v in decision.violation_details],
        )

        if route == "auto":
            # High confidence (>= 0.8): direct auto pass/reject
            return self._handle_auto_decision(task, feature, decision, original_text)

        elif route == "manual":
            # Low confidence (< 0.4): direct to manual queue
            return self._route_to_manual_queue(task, decision)

        else:  # route == "review"
            # Medium confidence (0.4-0.8): multi-agent review
            return self._execute_multi_agent_review(task, feature, decision, context, original_text)

    def _handle_auto_decision(
        self,
        task: AuditTask,
        feature: Any,
        decision: AuditDecision,
        original_text: str,
    ) -> AuditDecision:
        """Handle high confidence auto decision (>= 0.8)"""
        if decision.final_label.value == "normal":
            decision.status = AuditStatus.AUTO_APPROVED
        else:
            decision.status = AuditStatus.AUTO_REJECTED

        decision.needs_manual = False
        decision.decision_path.append("auto_decision")

        # Upsert to Milvus with source="ai"
        self._upsert_ai_case(task, feature, decision)
        # Append to NB training corpus for each violation type
        for vd in decision.violation_details:
            self._append_nb_sample(original_text, vd.violation_type.value, source="ai_auto")

        logger.info(
            "High confidence auto decision: task_id=%s, label=%s, confidence=%.2f, violation_types=%s, upserted to Milvus",
            task.task_id,
            decision.final_label.value,
            decision.confidence,
            [v.violation_type.value for v in decision.violation_details],
        )
        return decision

    def _route_to_manual_queue(
        self,
        task: AuditTask,
        decision: AuditDecision,
    ) -> AuditDecision:
        """Route low confidence decision to manual queue"""
        decision.status = AuditStatus.NEEDS_MANUAL
        decision.needs_manual = True
        decision.decision_path.append("low_confidence_manual")
        self._pending_manual_context[task.task_id] = (task, None)
        if self.manual_bridge is not None:
            self.manual_bridge.send_to_manual(task, decision, None)
        logger.info(
            "Low confidence routing to manual: task_id=%s, confidence=%.2f",
            task.task_id,
            decision.confidence,
        )
        return decision

    def _execute_multi_agent_review(
        self,
        task: AuditTask,
        feature: Any,
        decision: AuditDecision,
        context: dict[str, Any],
        original_text: str,
    ) -> AuditDecision:
        """Execute multi-agent review for medium confidence decisions (0.4-0.8)"""
        # Create review session
        session = self.reentrancy_manager.create_session(
            task_id=task.task_id,
            trace_id=task.trace_id,
            original_decision=decision,
        )

        logger.info(
            "Starting multi-agent review: task_id=%s, session_id=%s, confidence=%.2f",
            task.task_id,
            session.session_id,
            decision.confidence,
        )

        # Execute multi-agent review
        reviewed_decision = self.supervisor.review(
            decision=decision,
            context=context,
            session=session,
        )

        # Route based on review result
        review_route = self.gateway.route_review_result(reviewed_decision.confidence)

        if review_route == "auto_pass":
            # Review passed: auto pass and upsert
            if reviewed_decision.final_label.value == "normal":
                reviewed_decision.status = AuditStatus.AUTO_APPROVED
            else:
                reviewed_decision.status = AuditStatus.AUTO_REJECTED
            reviewed_decision.needs_manual = False
            reviewed_decision.decision_path.append("multi_agent_review_pass")

            self._upsert_ai_case(task, feature, reviewed_decision)
            for vd in reviewed_decision.violation_details:
                self._append_nb_sample(original_text, vd.violation_type.value, source="ai_auto_review")

            logger.info(
                "Multi-agent review passed: task_id=%s, confidence=%.2f, label=%s",
                task.task_id,
                reviewed_decision.confidence,
                reviewed_decision.final_label.value,
            )
            return reviewed_decision

        else:  # review_route == "dlq_manual"
            # Review failed: route to manual queue
            reviewed_decision.status = AuditStatus.NEEDS_MANUAL
            reviewed_decision.needs_manual = True
            reviewed_decision.decision_path.append("multi_agent_review_dlq")
            self._pending_manual_context[task.task_id] = (task, feature)
            if self.manual_bridge is not None:
                self.manual_bridge.send_to_manual(task, reviewed_decision, feature)

            logger.info(
                "Multi-agent review dlq: task_id=%s, confidence=%.2f, errors=%d",
                task.task_id,
                reviewed_decision.confidence,
                session.total_errors_found,
            )
            return reviewed_decision

    def _upsert_ai_case(
        self,
        task: AuditTask,
        feature: Any,
        decision: AuditDecision,
    ) -> None:
        """Upsert AI decision case to Milvus with source='ai'.

        For multi-label scenarios, upserts each violation type separately.
        """
        try:
            from datetime import datetime
            created_at = datetime.utcnow().isoformat()

            # If we have multiple violation details, upsert each one
            if decision.violation_details:
                for vd in decision.violation_details:
                    self.retrieval.upsert_case(
                        task=task,
                        feature=feature,
                        violation_type=vd.violation_type.value,
                        risk_score=vd.confidence,
                        model_version="ai-auto-v1",
                        human_verified=False,
                        created_at=created_at,
                    )
                    logger.info(
                        "AI case upserted: task_id=%s, violation_type=%s, risk_score=%.2f",
                        task.task_id,
                        vd.violation_type.value,
                        vd.confidence,
                    )
            else:
                # Fallback to single violation type
                self.retrieval.upsert_case(
                    task=task,
                    feature=feature,
                    violation_type=decision.final_label.value,
                    risk_score=decision.confidence,
                    model_version="ai-auto-v1",
                    human_verified=False,
                    created_at=created_at,
                )
                logger.info(
                    "AI case upserted: task_id=%s, violation_type=%s, risk_score=%.2f, violation_details=%s",
                    task.task_id,
                    decision.final_label.value,
                    decision.confidence,
                    [(v.violation_type.value, v.confidence) for v in decision.violation_details],
                )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to upsert AI case: task_id=%s", task.task_id)

    def _append_nb_sample(
        self,
        text: str,
        label: str,
        source: str = "ai_auto",
    ) -> None:
        """Append AI decision sample to NB training corpus."""
        try:
            self.prelabel.text_classifier.append_training_sample(
                text=text,
                label=label,
                source=source,
                verified=False,
            )
            logger.info("NB sample appended: label=%s, source=%s", label, source)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to append NB sample: label=%s", label)

    def build_result_message(
        self,
        task: AuditTask,
        decision: AuditDecision,
    ) -> AuditResultMessage:
        """Build result message for MQ response to platform."""
        return AuditResultMessage(
            case_id=task.task_id,
            trace_id=task.trace_id,
            final_label=decision.final_label,
            confidence=decision.confidence,
            status=decision.status,
            reason=decision.reason,
            source="ai_auto",
            metadata={
                "decision_path": decision.decision_path,
            },
            violation_details=decision.violation_details,
        )

    @property
    def image_engine(self):
        """Lazy load image audit engine."""
        if self._image_engine is None:
            from src.engine.image_audit_engine import ImageAuditEngine

            self._image_engine = ImageAuditEngine()
        return self._image_engine

    @property
    def video_engine(self):
        """Lazy load video audit engine."""
        if self._video_engine is None:
            from src.engine.video_audit_engine import VideoAuditEngine

            self._video_engine = VideoAuditEngine()
        return self._video_engine

    def handle_image_audit_task(
        self,
        task: AuditTask,
        image_path: str | None = None,
        image_base64: str | None = None,
        image_url: str | None = None,
    ) -> AuditDecision:
        """Handle image audit task.

        Args:
            task: 审核任务
            image_path: 图片文件路径（优先使用）
            image_base64: Base64 编码的图片
            image_url: 图片 URL（来自 MinIO）

        Returns:
            AuditDecision 审核决策
        """
        logger.info(
            "开始图片审核: task_id=%s, media_type=%s, image_url=%s",
            task.task_id,
            task.media_type.value,
            image_url,
        )

        try:
            # 调用图片审核引擎执行审核
            result = self.image_engine.audit_image(
                task=task,
                image_path=image_path,
                image_base64=image_base64,
                image_url=image_url,
            )

            decision = result.decision

            logger.info(
                "图片审核完成: task_id=%s, final_label=%s, confidence=%.2f, status=%s",
                task.task_id,
                decision.final_label.value,
                decision.confidence,
                decision.status.value,
            )

            # Extended confidence routing with multi-agent review
            route = self.gateway.route_initial(decision.confidence)

            if route == "auto":
                # High confidence (>= 0.8): direct auto pass/reject
                # Already handled by image_audit_engine, just log
                decision.decision_path.append("image_auto_decision")
                logger.info(
                    "Image auto decision: task_id=%s, label=%s, confidence=%.2f",
                    task.task_id,
                    decision.final_label.value,
                    decision.confidence,
                )
                return decision

            elif route == "manual":
                # Low confidence (< 0.4): direct to manual queue
                decision.status = AuditStatus.NEEDS_MANUAL
                decision.needs_manual = True
                decision.decision_path.append("image_low_confidence_manual")
                self._pending_manual_context[task.task_id] = (task, None)
                if self.manual_bridge is not None:
                    self.manual_bridge.send_to_manual(task, decision, None)
                logger.info(
                    "Image low confidence routing to manual: task_id=%s, confidence=%.2f",
                    task.task_id,
                    decision.confidence,
                )
                return decision

            else:  # route == "review"
                # Medium confidence (0.4-0.8): multi-agent review
                # Build context for review using image description
                context = {
                    "retrieved_cases": [],
                    "metadata": {"media_type": "image", "source": task.source},
                    "image_description": result.image_description,
                    "quick_label_types": [v.value for v in result.quick_label_types],
                    "quick_label_score": result.quick_label_score,
                }

                # Create review session
                session = self.reentrancy_manager.create_session(
                    task_id=task.task_id,
                    trace_id=task.trace_id,
                    original_decision=decision,
                )

                logger.info(
                    "Starting image multi-agent review: task_id=%s, session_id=%s, confidence=%.2f",
                    task.task_id,
                    session.session_id,
                    decision.confidence,
                )

                # Execute multi-agent review
                reviewed_decision = self.supervisor.review(
                    decision=decision,
                    context=context,
                    session=session,
                )

                # Route based on review result
                review_route = self.gateway.route_review_result(reviewed_decision.confidence)

                if review_route == "auto_pass":
                    # Review passed: auto pass
                    if reviewed_decision.final_label.value == "normal":
                        reviewed_decision.status = AuditStatus.AUTO_APPROVED
                    else:
                        reviewed_decision.status = AuditStatus.AUTO_REJECTED
                    reviewed_decision.needs_manual = False
                    reviewed_decision.decision_path.append("image_multi_agent_review_pass")

                    logger.info(
                        "Image multi-agent review passed: task_id=%s, confidence=%.2f, label=%s",
                        task.task_id,
                        reviewed_decision.confidence,
                        reviewed_decision.final_label.value,
                    )
                    return reviewed_decision

                else:  # review_route == "dlq_manual"
                    # Review failed: route to manual queue
                    reviewed_decision.status = AuditStatus.NEEDS_MANUAL
                    reviewed_decision.needs_manual = True
                    reviewed_decision.decision_path.append("image_multi_agent_review_dlq")
                    self._pending_manual_context[task.task_id] = (task, None)
                    if self.manual_bridge is not None:
                        self.manual_bridge.send_to_manual(task, reviewed_decision, None)

                    logger.info(
                        "Image multi-agent review dlq: task_id=%s, confidence=%.2f, errors=%d",
                        task.task_id,
                        reviewed_decision.confidence,
                        session.total_errors_found,
                    )
                    return reviewed_decision

        except Exception as e:
            logger.error("图片审核异常: task_id=%s, error=%s", task.task_id, e)
            return AuditDecision(
                trace_id=task.trace_id,
                task_id=task.task_id,
                final_label=ViolationType.OTHER,
                confidence=0.0,
                status=AuditStatus.FAILED,
                reason=f"图片审核异常: {str(e)[:100]}",
                decision_path=["image_audit_error"],
                needs_manual=True,
                violation_details=[],
            )

    def handle_video_audit_task(
        self,
        task: AuditTask,
        video_path: str | None = None,
        video_url: str | None = None,
    ) -> AuditDecision:
        """Handle video audit task.

        Args:
            task: 审核任务
            video_path: 视频文件路径（优先使用）
            video_url: 视频 URL（来自 MinIO）

        Returns:
            AuditDecision 审核决策
        """
        logger.info(
            "开始视频审核: task_id=%s, media_type=%s, video_url=%s",
            task.task_id,
            task.media_type.value,
            video_url,
        )

        try:
            # 调用视频审核引擎执行审核
            result = self.video_engine.audit_video(
                task=task,
                video_path=video_path,
                video_url=video_url,
            )

            decision = result.decision

            logger.info(
                "视频审核完成: task_id=%s, final_label=%s, confidence=%.2f, status=%s",
                task.task_id,
                decision.final_label.value,
                decision.confidence,
                decision.status.value,
            )

            # Extended confidence routing with multi-agent review
            route = self.gateway.route_initial(decision.confidence)

            if route == "auto":
                # High confidence (>= 0.8): direct auto pass/reject
                decision.decision_path.append("video_auto_decision")
                logger.info(
                    "Video auto decision: task_id=%s, label=%s, confidence=%.2f",
                    task.task_id,
                    decision.final_label.value,
                    decision.confidence,
                )
                return decision

            elif route == "manual":
                # Low confidence (< 0.4): direct to manual queue
                decision.status = AuditStatus.NEEDS_MANUAL
                decision.needs_manual = True
                decision.decision_path.append("video_low_confidence_manual")
                self._pending_manual_context[task.task_id] = (task, None)
                if self.manual_bridge is not None:
                    self.manual_bridge.send_to_manual(task, decision, None)
                logger.info(
                    "Video low confidence routing to manual: task_id=%s, confidence=%.2f",
                    task.task_id,
                    decision.confidence,
                )
                return decision

            else:  # route == "review"
                # Medium confidence (0.4-0.8): multi-agent review
                context = {
                    "retrieved_cases": [],
                    "metadata": {"media_type": "video", "source": task.source},
                    "video_description": "; ".join(result.frame_descriptions[:3]),
                    "quick_label_types": [v.value for v in result.quick_label_types],
                    "quick_label_score": result.quick_label_score,
                }

                session = self.reentrancy_manager.create_session(
                    task_id=task.task_id,
                    trace_id=task.trace_id,
                    original_decision=decision,
                )

                logger.info(
                    "Starting video multi-agent review: task_id=%s, session_id=%s, confidence=%.2f",
                    task.task_id,
                    session.session_id,
                    decision.confidence,
                )

                reviewed_decision = self.supervisor.review(
                    decision=decision,
                    context=context,
                    session=session,
                )

                review_route = self.gateway.route_review_result(reviewed_decision.confidence)

                if review_route == "auto_pass":
                    if reviewed_decision.final_label.value == "normal":
                        reviewed_decision.status = AuditStatus.AUTO_APPROVED
                    else:
                        reviewed_decision.status = AuditStatus.AUTO_REJECTED
                    reviewed_decision.needs_manual = False
                    reviewed_decision.decision_path.append("video_multi_agent_review_pass")

                    logger.info(
                        "Video multi-agent review passed: task_id=%s, confidence=%.2f, label=%s",
                        task.task_id,
                        reviewed_decision.confidence,
                        reviewed_decision.final_label.value,
                    )
                    return reviewed_decision

                else:  # review_route == "dlq_manual"
                    reviewed_decision.status = AuditStatus.NEEDS_MANUAL
                    reviewed_decision.needs_manual = True
                    reviewed_decision.decision_path.append("video_multi_agent_review_dlq")
                    self._pending_manual_context[task.task_id] = (task, None)
                    if self.manual_bridge is not None:
                        self.manual_bridge.send_to_manual(task, reviewed_decision, None)

                    logger.info(
                        "Video multi-agent review dlq: task_id=%s, confidence=%.2f, errors=%d",
                        task.task_id,
                        reviewed_decision.confidence,
                        session.total_errors_found,
                    )
                    return reviewed_decision

        except Exception as e:
            logger.error("视频审核异常: task_id=%s, error=%s", task.task_id, e)
            return AuditDecision(
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
