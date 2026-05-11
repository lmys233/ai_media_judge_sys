"""元复核监督器模块

协调三智能体（1个 ReAct 决策者 + 2个 CoT 辅助者）的复核工作流程：
- 管理复核会话生命周期
- 协调决策者和辅助者的交互
- 跟踪错误计数实现自适应机制
- 决定最终路由（通过或死信队列）
"""

from __future__ import annotations

import logging
from typing import Any

from src.decision.extended_gateway import ExtendedConfidenceGateway
from src.models.contracts import AuditDecision, AuditStatus
from src.reasoning.multi_agent.contracts import ReviewSession, ReviewStatus
from src.reasoning.multi_agent.decision_maker import ReActDecisionMaker
from src.reasoning.multi_agent.evidence_assistant import CoTEvidenceAssistant
from src.reasoning.multi_agent.logic_assistant import CoTLogicAssistant

logger = logging.getLogger(__name__)


class MetaReviewSupervisor:
    """元复核监督器

    协调三智能体复核工作流程：

    流程：
    1. 决策者(ReAct) 基于原始结论 + 案例上下文输出 thinking chain
    2. 逻辑辅助者(CoT) 分析 thinking chain 的逻辑漏洞
    3. 证据辅助者(CoT) 分析证据充分性
    4. 如果辅助者发现错误，反馈给决策者进行 rerethink
    5. 重复步骤 1-4，直到：
       - 辅助者累计发现 3+ 错误 -> 死信队列
       - 达到最大轮次 -> 根据置信度决定
       - 辅助者无异议 -> 根据最终置信度决定

    阈值：
    - MAX_ERRORS_DLQ = 3: 辅助者发现 3 次错误则死信队列
    - REVIEW_CONFIDENCE_PASS = 0.65: 复核后置信度 >= 0.65 则通过
    """

    MAX_ERRORS_DLQ = 3
    REVIEW_CONFIDENCE_PASS = 0.65

    def __init__(
        self,
        decision_maker: ReActDecisionMaker,
        logic_assistant: CoTLogicAssistant,
        evidence_assistant: CoTEvidenceAssistant,
        gateway: ExtendedConfidenceGateway,
    ) -> None:
        self.decision_maker = decision_maker
        self.logic_assistant = logic_assistant
        self.evidence_assistant = evidence_assistant
        self.gateway = gateway

    def review(
        self,
        decision: AuditDecision,
        context: dict[str, Any],
        session: ReviewSession,
    ) -> AuditDecision:
        """执行多智能体复核

        Args:
            decision: 原始审核决策
            context: 案例上下文（检索到的案例、元数据等）
            session: 复核会话

        Returns:
            复核后的 AuditDecision
        """
        logger.info(
            "开始多智能体复核: task_id=%s, original_confidence=%.2f, session_id=%s",
            decision.task_id,
            decision.confidence,
            session.session_id,
        )

        try:
            # 执行复核轮次
            session = self._execute_review_rounds(session, context)

            # 根据复核结果决定路由
            if self._should_route_to_dlq(session):
                session.status = ReviewStatus.DLQ_MANUAL
                return self._build_dlq_decision(session, "max_errors_or_threshold")
            else:
                session.status = ReviewStatus.PASSED
                return self._build_pass_decision(session)

        except Exception as e:
            logger.error("复核过程异常: task_id=%s, error=%s", decision.task_id, e)
            session.status = ReviewStatus.ERROR
            return self._build_error_decision(session, str(e))

    def _execute_review_rounds(
        self,
        session: ReviewSession,
        context: dict[str, Any],
    ) -> ReviewSession:
        """执行多轮复核"""
        # 收集之前的质疑（如果有）
        all_objections = list(session.logic_objections) + list(session.evidence_objections)

        # 增加轮次
        session.increment_round()
        round_num = session.review_round

        logger.info(
            "执行复核轮次: task_id=%s, round=%d, total_errors=%d",
            session.task_id,
            round_num,
            session.total_errors_found,
        )

        # 步骤 1: 决策者决策
        dm_output = self.decision_maker.decide(
            original_decision=session.original_decision,
            context=context,
            assistant_objections=all_objections if all_objections else None,
            round_num=round_num,
        )

        # 更新会话状态
        session.current_decision = dm_output.decision
        session.thinking_chain.extend(dm_output.thinking_chain)
        session.decision_path.append(f"dm_round_{round_num}")
        session.decision_path.append(f"dm_confidence_{dm_output.confidence:.2f}")

        # 步骤 2: 逻辑辅助者分析
        logic_objection = self.logic_assistant.analyze(
            thinking_chain=dm_output.thinking_chain,
            original_decision=session.original_decision,
            context=context,
        )
        session.add_objection(logic_objection)

        if logic_objection.has_objection:
            session.decision_path.append(f"logic_objection_{len(logic_objection.objection_points)}")

        # 步骤 3: 证据辅助者分析
        evidence_objection = self.evidence_assistant.analyze(
            thinking_chain=dm_output.thinking_chain,
            key_evidence=dm_output.key_evidence,
            original_decision=session.original_decision,
            context=context,
        )
        session.add_objection(evidence_objection)

        if evidence_objection.has_objection:
            session.decision_path.append(f"evidence_objection_{len(evidence_objection.objection_points)}")

        logger.info(
            "复核轮次完成: task_id=%s, round=%d, logic_objection=%s, evidence_objection=%s, "
            "total_errors=%d, current_confidence=%.2f",
            session.task_id,
            round_num,
            logic_objection.has_objection,
            evidence_objection.has_objection,
            session.total_errors_found,
            session.current_decision.confidence,
        )

        # 检查是否应该继续
        if session.should_stop():
            logger.info(
                "复核停止条件触发: task_id=%s, should_stop=%s",
                session.task_id,
                session.should_stop(),
            )
            return session

        # 如果有质疑，决策者需要在下一轮重新思考
        has_objection = logic_objection.has_objection or evidence_objection.has_objection
        if has_objection:
            session.decision_path.append("has_objection_continue")
            # 继续下一轮（递归调用）
            return self._execute_review_rounds(session, context)

        return session

    def _should_route_to_dlq(self, session: ReviewSession) -> bool:
        """判断是否应该路由到死信队列"""
        # 错误阈值达到
        if session.total_errors_found >= self.MAX_ERRORS_DLQ:
            logger.info(
                "错误阈值触发: task_id=%s, errors=%d >= %d",
                session.task_id,
                session.total_errors_found,
                self.MAX_ERRORS_DLQ,
            )
            return True

        # 最大轮次达到
        if session.review_round >= session.max_rounds:
            logger.info(
                "最大轮次触发: task_id=%s, round=%d >= %d",
                session.task_id,
                session.review_round,
                session.max_rounds,
            )
            return True

        # 置信度低于阈值
        if session.current_decision.confidence < self.REVIEW_CONFIDENCE_PASS:
            logger.info(
                "置信度阈值触发: task_id=%s, confidence=%.2f < %.2f",
                session.task_id,
                session.current_decision.confidence,
                self.REVIEW_CONFIDENCE_PASS,
            )
            return True

        return False

    def _build_dlq_decision(
        self,
        session: ReviewSession,
        reason: str,
    ) -> AuditDecision:
        """构建死信队列决策"""
        decision = session.current_decision
        decision.status = AuditStatus.NEEDS_MANUAL
        decision.needs_manual = True
        decision.decision_path.append(f"dlq_manual_reason_{reason}")
        decision.decision_path.append(f"dlq_total_errors_{session.total_errors_found}")
        decision.decision_path.append(f"dlq_final_confidence_{decision.confidence:.2f}")

        logger.info(
            "复核结果-死信队列: task_id=%s, confidence=%.2f, errors=%d, reason=%s",
            session.task_id,
            decision.confidence,
            session.total_errors_found,
            reason,
        )

        return decision

    def _build_pass_decision(self, session: ReviewSession) -> AuditDecision:
        """构建通过决策"""
        decision = session.current_decision
        decision.decision_path.append(f"review_pass_confidence_{decision.confidence:.2f}")
        decision.decision_path.append(f"review_rounds_{session.review_round}")

        logger.info(
            "复核结果-通过: task_id=%s, confidence=%.2f, rounds=%d",
            session.task_id,
            decision.confidence,
            session.review_round,
        )

        return decision

    def _build_error_decision(
        self,
        session: ReviewSession,
        error_message: str,
    ) -> AuditDecision:
        """构建错误决策"""
        decision = session.current_decision
        decision.status = AuditStatus.FAILED
        decision.needs_manual = True
        decision.decision_path.append(f"review_error_{error_message[:50]}")

        return decision
