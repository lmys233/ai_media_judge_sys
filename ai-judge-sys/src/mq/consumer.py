from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable

from pydantic import ValidationError

from src.models.contracts import AuditTask
from src.mq.rabbitmq_client import QueueNames, RabbitMQClient

logger = logging.getLogger(__name__)


class InMemoryDedupeStore:
    """PoC idempotency store, replace with Redis in production."""

    def __init__(self, ttl_seconds: int = 600) -> None:
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, float] = {}

    def seen(self, key: str) -> bool:
        now = time.time()
        self._cache = {k: v for k, v in self._cache.items() if v > now}
        if key in self._cache:
            return True
        self._cache[key] = now + self.ttl_seconds
        return False

    def clear(self, key: str) -> None:
        """清除dedupe记录，允许任务重新被处理（用于重试场景）"""
        self._cache.pop(key, None)


class AuditTaskConsumer:
    def __init__(
        self,
        mq_client: RabbitMQClient,
        task_handler: Callable[[AuditTask], None],
        dedupe_store: InMemoryDedupeStore | None = None,
        queue_names: QueueNames | None = None,
    ) -> None:
        self.mq_client = mq_client
        self.task_handler = task_handler
        self.dedupe_store = dedupe_store or InMemoryDedupeStore()
        self.queue_names = queue_names or QueueNames()

    def _process_message(self, channel, method, properties, body) -> None:  # noqa: ANN001
        try:
            payload = json.loads(body.decode("utf-8"))
            task = AuditTask.model_validate(payload)
            if self.dedupe_store.seen(task.dedupe_key):
                logger.info("跳过重复任务 task=%s 去重键=%s", task.task_id, task.dedupe_key)
                channel.basic_ack(delivery_tag=method.delivery_tag)
                return

            self.task_handler(task)
            channel.basic_ack(delivery_tag=method.delivery_tag)
        except ValidationError as exc:
            logger.exception("任务消息格式无效，已发送到死信队列: %s", exc)
            self.mq_client.publish(self.queue_names.dlq, {"error": "validation_error", "body": body.decode("utf-8")})
            channel.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as exc:  # noqa: BLE001
            payload = json.loads(body.decode("utf-8"))
            retry_count = int(payload.get("retry_count", 0))
            max_retry = int(payload.get("max_retry", 3))
            if retry_count >= max_retry:
                logger.exception("任务重试耗尽，发送到失败队列: %s", exc)
                self.mq_client.publish(self.queue_names.failure, payload)
                channel.basic_ack(delivery_tag=method.delivery_tag)
                return

            payload["retry_count"] = retry_count + 1
            # 清除dedupe缓存，让重试任务能正常处理
            self.dedupe_store.clear(task.dedupe_key)
            self.mq_client.publish(self.queue_names.task, payload)
            channel.basic_ack(delivery_tag=method.delivery_tag)

    def start(self) -> None:
        self.mq_client.consume(self.queue_names.task, self._process_message)
