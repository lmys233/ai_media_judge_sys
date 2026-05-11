from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from src.integration.manual_case_sync import ManualCaseSyncService
from src.models.contracts import ManualAuditCaseMessage
from src.mq.rabbitmq_client import QueueNames, RabbitMQClient

logger = logging.getLogger(__name__)


class ManualCaseConsumer:
    def __init__(self, mq_client: RabbitMQClient, sync_service: ManualCaseSyncService, queue_names: QueueNames | None = None) -> None:
        self.mq_client = mq_client
        self.sync_service = sync_service
        self.queue_names = queue_names or QueueNames()

    def _callback(self, channel, method, properties, body) -> None:  # noqa: ANN001
        try:
            payload = json.loads(body.decode("utf-8"))
            case = ManualAuditCaseMessage.model_validate(payload)
            result = self.sync_service.ingest_case(case)
            logger.info("人工案例已入库 案例ID=%s 结果=%s", case.case_id, result)
            channel.basic_ack(delivery_tag=method.delivery_tag)
        except ValidationError as exc:
            logger.exception("人工案例消息格式校验失败: %s", exc)
            self.mq_client.publish(self.queue_names.dlq, {"error": "manual_case_validation_error", "body": body.decode("utf-8")})
            channel.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as exc:  # noqa: BLE001
            logger.exception("人工案例入库处理失败: %s", exc)
            self.mq_client.publish(self.queue_names.dlq, {"error": "manual_case_ingest_error", "body": body.decode("utf-8")})
            channel.basic_ack(delivery_tag=method.delivery_tag)

    def start(self) -> None:
        self.mq_client.consume(self.queue_names.manual_case, self._callback)
