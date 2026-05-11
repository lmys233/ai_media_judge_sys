"""可重入管理器模块

管理复核会话的生命周期，支持同一案例多次进入复核流程：
- 为每次进入创建新的 session_id
- 追踪同一 task_id 的所有复核尝试
- 累积错误计数
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from src.models.contracts import AuditDecision
from src.reasoning.multi_agent.contracts import ReviewSession, ReviewStatus

logger = logging.getLogger(__name__)


class ReviewSessionStorage:
    """复核会话存储接口（内存实现）"""

    def __init__(self) -> None:
        self._sessions: dict[str, ReviewSession] = {}
        self._task_sessions: dict[str, list[str]] = {}

    def save(self, session: ReviewSession) -> None:
        """保存会话"""
        self._sessions[session.session_id] = session
        if session.task_id not in self._task_sessions:
            self._task_sessions[session.task_id] = []
        if session.session_id not in self._task_sessions[session.task_id]:
            self._task_sessions[session.task_id].append(session.session_id)

    def load(self, session_id: str) -> ReviewSession | None:
        """加载会话"""
        return self._sessions.get(session_id)

    def load_by_task_id(self, task_id: str) -> list[ReviewSession]:
        """根据 task_id 加载所有会话"""
        session_ids = self._task_sessions.get(task_id, [])
        return [self._sessions[sid] for sid in session_ids if sid in self._sessions]

    def link(self, session_id: str, parent_id: str | None) -> None:
        """链接会话到父会话（用于审计追踪）"""
        if session_id in self._sessions:
            self._sessions[session_id].decision_path.append(f"linked_to_parent_{parent_id}")

    def get_cumulative_errors(self, task_id: str) -> int:
        """获取同一 task_id 的累计错误数"""
        sessions = self.load_by_task_id(task_id)
        return sum(s.total_errors_found for s in sessions)


class ReentrancyManager:
    """可重入管理器

    支持同一案例多次进入复核流程的设计：

    原则：
    - 同一 task 可以多次进入复核流程
    - 每次进入创建新的 session_id
    - 通过 task_id 链接所有尝试
    - 错误计数跨尝试累积

    使用场景：
    - 置信度 0.4-0.8 的案例进入复核
    - 复核后置信度变化，需要再次复核
    - 同一案例在不同时间点被重新送审
    """

    def __init__(self, storage: ReviewSessionStorage | None = None) -> None:
        self.storage = storage or ReviewSessionStorage()

    def create_session(
        self,
        task_id: str,
        trace_id: str,
        original_decision: AuditDecision,
        parent_session_id: str | None = None,
    ) -> ReviewSession:
        """为任务创建新的复核会话

        Args:
            task_id: 任务 ID
            trace_id: 追踪 ID
            original_decision: 原始审核决策
            parent_session_id: 可选的父会话 ID（用于链接审计追踪）

        Returns:
            新的 ReviewSession 实例
        """
        session_id = str(uuid.uuid4())
        session = ReviewSession(
            session_id=session_id,
            task_id=task_id,
            trace_id=trace_id,
            original_decision=original_decision,
            current_decision=original_decision,
            thinking_chain=[],
            logic_objections=[],
            evidence_objections=[],
            total_errors_found=0,
            review_round=0,
            max_rounds=3,
            status=ReviewStatus.IN_PROGRESS,
        )

        # 如果有父会话，链接起来
        if parent_session_id:
            session.decision_path.append(f"child_of_session_{parent_session_id}")
            # 获取父会话的累计错误
            parent_errors = self.storage.get_cumulative_errors(task_id)
            session.total_errors_found = parent_errors
            session.decision_path.append(f"inherited_errors_{parent_errors}")

        self.storage.save(session)

        logger.info(
            "创建复核会话: task_id=%s, session_id=%s, parent=%s, initial_errors=%d",
            task_id,
            session_id,
            parent_session_id,
            session.total_errors_found,
        )

        return session

    def resume_session(
        self,
        session_id: str,
        updated_decision: AuditDecision,
    ) -> ReviewSession | None:
        """恢复并更新已有会话

        Args:
            session_id: 会话 ID
            updated_decision: 更新后的决策

        Returns:
            更新后的 ReviewSession 或 None
        """
        session = self.storage.load(session_id)
        if session:
            session.current_decision = updated_decision
            self.storage.save(session)
            logger.info(
                "更新复核会话: session_id=%s, new_confidence=%.2f",
                session_id,
                updated_decision.confidence,
            )
        return session

    def get_session(self, session_id: str) -> ReviewSession | None:
        """获取会话"""
        return self.storage.load(session_id)

    def get_task_sessions(self, task_id: str) -> list[ReviewSession]:
        """获取同一任务的所有会话"""
        return self.storage.load_by_task_id(task_id)

    def get_cumulative_errors(self, task_id: str) -> int:
        """获取同一任务的累计错误数"""
        return self.storage.get_cumulative_errors(task_id)

    def is_first_review(self, task_id: str) -> bool:
        """判断是否为首次复核"""
        return len(self.storage.load_by_task_id(task_id)) == 0
