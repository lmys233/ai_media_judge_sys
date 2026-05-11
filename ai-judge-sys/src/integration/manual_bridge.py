from __future__ import annotations

from src.models.contracts import (
    AuditDecision,
    AuditFeature,
    AuditTask,
    ManualReviewMessage,
    ManualReviewResultMessage,
)
from src.mq.rabbitmq_client import QueueNames, RabbitMQClient


class ManualReviewBridge:
    def __init__(self, mq_client: RabbitMQClient, queue_names: QueueNames | None = None) -> None:
        self.mq_client = mq_client
        self.queue_names = queue_names or QueueNames()

    def send_to_manual(self, task: AuditTask, decision: AuditDecision, feature: AuditFeature) -> None:
        payload = ManualReviewMessage(
            trace_id=task.trace_id,
            task_id=task.task_id,
            biz_id=task.biz_id,
            confidence=decision.confidence,
            final_label=decision.final_label,
            reason=decision.reason,
            metadata_scalar=feature.metadata_scalar,
            desensitized_summary=feature.description_desensitized,
        ).model_dump(mode="json")
        self.mq_client.publish(self.queue_names.manual, payload)

    def parse_manual_result(self, payload: dict) -> ManualReviewResultMessage:
        return ManualReviewResultMessage.model_validate(payload)
