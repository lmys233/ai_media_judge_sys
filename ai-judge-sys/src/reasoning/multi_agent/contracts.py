"""多智能体复核模块数据结构和枚举

定义多智能体复核系统使用的数据结构：
- ThinkingChain: 决策者的推理链
- AssistantObjection: 辅助者的质疑
- ReviewSession: 复核会话
- ReviewStatus: 会话状态
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ReviewStatus(str, Enum):
    """复核会话状态"""

    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    DLQ_MANUAL = "dlq_manual"
    MAX_RETRIES = "max_retries"
    ERROR = "error"


@dataclass(frozen=True)
class ThinkingChain:
    """ReAct 决策者的推理链

    Attributes:
        step: 推理步骤编号
        observation: 观察到的内容
        reasoning: 推理过程
        action: 采取的行动
        conclusion: 得出的结论
    """

    step: int
    observation: str
    reasoning: str
    action: str
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "observation": self.observation,
            "reasoning": self.reasoning,
            "action": self.action,
            "conclusion": self.conclusion,
        }


@dataclass(frozen=True)
class AssistantObjection:
    """CoT 辅助者的质疑

    Attributes:
        assistant_type: 辅助者类型 ("logic" 或 "evidence")
        has_objection: 是否有质疑
        objection_points: 质疑点列表
        confidence_impact: 置信度影响 (-0.1 到 0.1)
        reasoning_chain: 质疑理由链
    """

    assistant_type: str
    has_objection: bool
    objection_points: list[str] = field(default_factory=list)
    confidence_impact: float = 0.0
    reasoning_chain: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "assistant_type": self.assistant_type,
            "has_objection": self.has_objection,
            "objection_points": self.objection_points,
            "confidence_impact": self.confidence_impact,
            "reasoning_chain": self.reasoning_chain,
        }


@dataclass
class ReviewSession:
    """复核会话

    支持同一案例多次进入复核流程（可重入设计）。

    Attributes:
        session_id: 会话唯一 ID (UUID)
        task_id: 任务 ID
        trace_id: 追踪 ID
        original_decision: 原始审核决策
        current_decision: 当前审核决策
        thinking_chain: 推理链列表
        logic_objections: 逻辑辅助者质疑列表
        evidence_objections: 证据辅助者质疑列表
        total_errors_found: 累计发现的错误数
        review_round: 当前复核轮次
        max_rounds: 最大复核轮次
        status: 会话状态
    """

    session_id: str
    task_id: str
    trace_id: str
    original_decision: Any  # AuditDecision
    current_decision: Any  # AuditDecision
    thinking_chain: list[ThinkingChain] = field(default_factory=list)
    logic_objections: list[AssistantObjection] = field(default_factory=list)
    evidence_objections: list[AssistantObjection] = field(default_factory=list)
    total_errors_found: int = 0
    review_round: int = 0
    max_rounds: int = 3
    status: ReviewStatus = ReviewStatus.IN_PROGRESS

    def add_thinking(self, thinking: ThinkingChain) -> None:
        """添加推理链"""
        self.thinking_chain.append(thinking)

    def add_objection(self, objection: AssistantObjection) -> None:
        """添加辅助者质疑"""
        if objection.assistant_type == "logic":
            self.logic_objections.append(objection)
        else:
            self.evidence_objections.append(objection)

        if objection.has_objection:
            self.total_errors_found += 1

    def increment_round(self) -> None:
        """增加复核轮次"""
        self.review_round += 1

    def should_stop(self) -> bool:
        """判断是否应该停止复核"""
        if self.total_errors_found >= 3:
            return True
        if self.review_round >= self.max_rounds:
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "trace_id": self.trace_id,
            "original_decision": self.original_decision.model_dump()
            if hasattr(self.original_decision, "model_dump")
            else str(self.original_decision),
            "current_decision": self.current_decision.model_dump()
            if hasattr(self.current_decision, "model_dump")
            else str(self.current_decision),
            "thinking_chain": [t.to_dict() for t in self.thinking_chain],
            "logic_objections": [o.to_dict() for o in self.logic_objections],
            "evidence_objections": [o.to_dict() for o in self.evidence_objections],
            "total_errors_found": self.total_errors_found,
            "review_round": self.review_round,
            "max_rounds": self.max_rounds,
            "status": self.status.value,
        }
