from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable

import pika


@dataclass
class QueueNames:
    task: str = "audit.task"
    manual: str = "audit.manual"
    manual_result: str = "audit.manual.result"
    manual_case: str = "audit.manual.case"
    dlq: str = "audit.task.dlq"
    failure: str = "audit.failure"


class RabbitMQClient:
    def __init__(self, queue_names: QueueNames | None = None) -> None:
        self.queue_names = queue_names or QueueNames()
        self.host = os.getenv("RABBITMQ_HOST", "localhost")
        self.port = int(os.getenv("RABBITMQ_PORT", "5672"))
        self.username = os.getenv("RABBITMQ_USERNAME", "admin")
        self.password = os.getenv("RABBITMQ_PASSWORD", "admin123")
        self.vhost = os.getenv("RABBITMQ_VHOST", "/")
        self._connection: pika.BlockingConnection | None = None
        self._channel: pika.channel.Channel | None = None

    def connect(self, declare_queues: list[str] | None = None) -> None:
        credentials = pika.PlainCredentials(self.username, self.password)
        params = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host=self.vhost,
            credentials=credentials,
            heartbeat=30,
        )
        self._connection = pika.BlockingConnection(params)
        self._channel = self._connection.channel()
        if declare_queues is None:
            self._declare_queues()
        else:
            self._declare_selected_queues(declare_queues)

    def _declare_selected_queues(self, queue_names: list[str]) -> None:
        assert self._channel is not None
        for q in queue_names:
            self._channel.queue_declare(queue=q, durable=True)

    def _declare_queues(self) -> None:
        assert self._channel is not None
        self._channel.queue_declare(
            queue=self.queue_names.dlq,
            durable=True,
        )
        task_args = {
            "x-dead-letter-exchange": "",
            "x-dead-letter-routing-key": self.queue_names.dlq,
        }
        self._channel.queue_declare(queue=self.queue_names.task, durable=True, arguments=task_args)
        self._channel.queue_declare(queue=self.queue_names.manual, durable=True)
        self._channel.queue_declare(queue=self.queue_names.manual_result, durable=True)
        self._channel.queue_declare(queue=self.queue_names.manual_case, durable=True)
        self._channel.queue_declare(queue=self.queue_names.failure, durable=True)

    def publish(self, queue_name: str, payload: dict) -> None:
        assert self._channel is not None
        self._channel.basic_publish(
            exchange="",
            routing_key=queue_name,
            body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            properties=pika.BasicProperties(
                delivery_mode=2,
                content_type="application/json",
            ),
        )

    def consume(self, queue_name: str, callback: Callable) -> None:
        assert self._channel is not None
        self._channel.basic_qos(prefetch_count=1)
        self._channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=False)
        self._channel.start_consuming()

    def stop_consuming(self) -> None:
        """优雅停止consuming，让start_consuming()正常返回"""
        if self._channel and self._channel.is_open:
            self._channel.stop_consuming()

    def close(self) -> None:
        if self._connection and self._connection.is_open:
            self._connection.close()
