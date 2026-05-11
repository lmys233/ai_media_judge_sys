from __future__ import annotations

import json
import logging
import os
import time

import pika

logger = logging.getLogger(__name__)


class AuditResultProducer:
    """Producer for sending AI audit results back to the platform."""

    def __init__(self) -> None:
        self.host = os.getenv("RABBITMQ_HOST", "localhost")
        self.port = int(os.getenv("RABBITMQ_PORT", "5672"))
        self.username = os.getenv("RABBITMQ_USERNAME", "admin")
        self.password = os.getenv("RABBITMQ_PASSWORD", "admin123")
        self.vhost = os.getenv("RABBITMQ_VHOST", "/")
        self.queue_name = "audit.result"
        self._connection: pika.BlockingConnection | None = None
        self._channel: pika.channel.Channel | None = None
        self._max_retries = 3
        self._retry_delay = 2  # seconds

    def connect(self) -> None:
        credentials = pika.PlainCredentials(self.username, self.password)
        params = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host=self.vhost,
            credentials=credentials,
            heartbeat=60,
            blocked_connection_timeout=120,
        )
        self._connection = pika.BlockingConnection(params)
        self._channel = self._connection.channel()
        self._channel.queue_declare(queue=self.queue_name, durable=True)
        logger.info("AuditResultProducer connected to RabbitMQ, queue=%s", self.queue_name)

    def _ensure_connection(self) -> None:
        """Ensure connection is alive, reconnect if needed."""
        if self._connection is None or self._connection.is_closed:
            logger.info("RabbitMQ connection lost, reconnecting...")
            self.connect()
            return

        if self._channel is None or self._channel.is_closed:
            logger.info("RabbitMQ channel closed, recreating...")
            self._channel = self._connection.channel()
            self._channel.queue_declare(queue=self.queue_name, durable=True)

    def send_result(self, result_message: dict) -> None:
        """Send audit result message to audit.result queue with retry logic."""
        last_error = None

        for attempt in range(self._max_retries):
            try:
                self._ensure_connection()

                self._channel.basic_publish(
                    exchange="",
                    routing_key=self.queue_name,
                    body=json.dumps(result_message, ensure_ascii=False).encode("utf-8"),
                    properties=pika.BasicProperties(
                        delivery_mode=2,
                        content_type="application/json",
                    ),
                )
                logger.info(
                    "Audit result sent: case_id=%s, label=%s, confidence=%.2f, status=%s",
                    result_message.get("case_id"),
                    result_message.get("final_label"),
                    result_message.get("confidence"),
                    result_message.get("status"),
                )
                return
            except (pika.exceptions.StreamLostError, pika.exceptions.ConnectionClosedByBroker, pika.exceptions.AMQPConnectionError) as e:
                last_error = e
                logger.warning(
                    "RabbitMQ send failed (attempt %d/%d): %s, reconnecting...",
                    attempt + 1,
                    self._max_retries,
                    e,
                )
                self._connection = None
                self._channel = None
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to send audit result: case_id=%s", result_message.get("case_id"))
                raise

        # All retries failed
        logger.error(
            "Failed to send audit result after %d attempts: case_id=%s, error=%s",
            self._max_retries,
            result_message.get("case_id"),
            last_error,
        )
        raise last_error

    def close(self) -> None:
        if self._connection and self._connection.is_open:
            self._connection.close()
            logger.info("AuditResultProducer connection closed")
