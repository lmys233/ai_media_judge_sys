from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

# 全局变量，供 signal handler 使用
_shutdown_flag = False


def _setup_signal_handler(mq_clients: list) -> None:
    """注册 SIGINT/SIGTERM handler，优雅退出"""

    def _handler(signum, frame):
        global _shutdown_flag
        if _shutdown_flag:
            logger.warning("正在强制退出 ...")
            sys.exit(1)
        _shutdown_flag = True
        logger.info("收到退出信号，正在优雅停止 ...")
        for mc in mq_clients:
            try:
                mc.stop_consuming()
            except Exception:
                pass

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def run_task_worker() -> None:
    """消费 audit.task 队列，运行完整审核引擎（需要 torch / cv2）。"""
    from src.engine.audit_engine import AuditEngine
    from src.integration.manual_bridge import ManualReviewBridge
    from src.models.contracts import AuditTask
    from src.mq.consumer import AuditTaskConsumer
    from src.mq.rabbitmq_client import RabbitMQClient

    logger.info("启动审核任务消费者 (队列: audit.task) ...")
    mq_client = RabbitMQClient()
    mq_client.connect()
    manual_bridge = ManualReviewBridge(mq_client)
    engine = AuditEngine(manual_bridge=manual_bridge)
    engine.bootstrap()

    _setup_signal_handler([mq_client])

    def _handler(task: AuditTask) -> None:
        decision = engine.handle_task(task)
        logger.info("任务完成 task=%s 判定=%s 置信度=%.3f", task.task_id, decision.final_label.value, decision.confidence)

    consumer = AuditTaskConsumer(mq_client=mq_client, task_handler=_handler)
    try:
        consumer.start()
    except (KeyboardInterrupt, InterruptedError):
        logger.info("收到中断信号，正在停止 ...")
    finally:
        mq_client.stop_consuming()
        mq_client.close()
        logger.info("审核任务Worker已退出")


def run_ai_auto_worker() -> None:
    """AI自动审核Worker：消费audit.task队列，执行自动审核->结果回传。

    支持文本和图片审核：
    - text: NB快速打标->文本重写->向量检索->LLM判断
    - image: CLIP快速打标->VL描述生成->向量检索->VL判断

    失败重试机制：
    - 审核失败后自动重试，最多3次
    - 3次都失败则发送到失败队列，通知平台保存到失败表
    """
    from src.engine.audit_engine import AuditEngine
    from src.models.contracts import AuditTask, MediaType
    from src.mq.consumer import AuditTaskConsumer
    from src.mq.result_producer import AuditResultProducer
    from src.mq.rabbitmq_client import RabbitMQClient

    logger.info("启动AI自动审核Worker (队列: audit.task) ...")
    mq_client = RabbitMQClient()
    mq_client.connect()
    engine = AuditEngine()
    engine.bootstrap()

    result_producer = AuditResultProducer()
    result_producer.connect()

    # 注册信号处理器
    _setup_signal_handler([mq_client])

    # 失败队列名称
    failure_queue_name = mq_client.queue_names.failure

    def _send_to_failure_queue(task: AuditTask, error_message: str) -> None:
        from datetime import datetime
        """发送失败任务到失败队列"""
        failure_payload = {
            "schema_version": "1.0.0",
            "task_id": task.task_id,
            "trace_id": task.trace_id,
            "biz_id": task.biz_id,
            "media_type": task.media_type.value,
            "media_url": task.media_url,
            "content_text": task.content_text,
            "error_message": error_message,
            "failed_at": str(datetime.utcnow()),
        }
        mq_client.publish(failure_queue_name, failure_payload)
        logger.warning("任务已发送到失败队列: task_id=%s, error=%s", task.task_id, error_message)

    def _handler(task: AuditTask) -> None:
        max_retry = 3
        retry_count = getattr(task, 'retry_count', 0)

        try:
            # 根据媒体类型选择审核流程
            if task.media_type == MediaType.IMAGE:
                # 图片审核流程
                logger.info("开始图片审核 task=%s media_url=%s", task.task_id, task.media_url)
                decision = engine.handle_image_audit_task(
                    task=task,
                    image_url=task.media_url,
                )
            elif task.media_type == MediaType.TEXT:
                # 文本审核流程
                original_text = task.content_text or ""
                if not original_text:
                    logger.warning("task=%s 缺少content_text，跳过", task.task_id)
                    return
                decision = engine.handle_ai_auto_task(task, original_text)
            elif task.media_type == MediaType.VIDEO:
                # 视频审核流程
                logger.info("开始视频审核 task=%s media_url=%s", task.task_id, task.media_url)
                decision = engine.handle_video_audit_task(
                    task=task,
                    video_url=task.media_url,
                )
            else:
                # 暂不支持的媒体类型
                logger.warning("task=%s 不支持的媒体类型: %s，跳过", task.task_id, task.media_type)
                return

            # 检查是否是失败状态
            if decision.status.value == "failed" or decision.confidence == 0.0:
                # 审核本身失败，抛出异常让 consumer 处理重试
                raise RuntimeError(f"审核失败: task_id={task.task_id}, reason={decision.reason}")

            # 发送结果回平台
            result_msg = engine.build_result_message(task, decision)
            result_producer.send_result(result_msg.model_dump(mode="json"))

            logger.info(
                "AI自动审核完成 task=%s media_type=%s 判定=%s 置信度=%.3f 状态=%s",
                task.task_id,
                task.media_type.value,
                decision.final_label.value,
                decision.confidence,
                decision.status.value,
            )

        except Exception as e:
            # 捕获异常，判断是否重试
            error_msg = str(e)
            logger.error("审核异常: task_id=%s, error=%s", task.task_id, error_msg)

            if retry_count >= max_retry:
                # 重试次数用尽，发送到失败队列，然后抛出让 consumer 不再重试
                logger.error("审核异常次数耗尽: task_id=%s, retry=%d", task.task_id, retry_count)
                _send_to_failure_queue(task, error_msg)
                return  # 不再抛异常，consumer 不会重试

            # retry_count < max_retry：抛出异常让 consumer 处理重试（consumer 会清除 dedupe）
            raise

    consumer = AuditTaskConsumer(mq_client=mq_client, task_handler=_handler)
    logger.info("AI自动审核Worker正在监听 audit.task 队列 ... (Ctrl+C 停止)")
    try:
        consumer.start()
    except (KeyboardInterrupt, InterruptedError):
        logger.info("收到中断信号，正在停止 ...")
    finally:
        mq_client.stop_consuming()
        mq_client.close()
        result_producer.close()
        logger.info("AI自动审核Worker已退出")


def run_manual_result_worker() -> None:
    """消费 audit.manual.result 队列，处理人工复核回传结果。"""
    from src.engine.audit_engine import AuditEngine
    from src.integration.manual_bridge import ManualReviewBridge
    from src.mq.rabbitmq_client import QueueNames, RabbitMQClient

    logger.info("启动人工复核结果消费者 (队列: audit.manual.result) ...")
    mq_client = RabbitMQClient()
    mq_client.connect()
    queue_names = QueueNames()
    engine = AuditEngine(manual_bridge=ManualReviewBridge(mq_client))
    engine.bootstrap()

    _setup_signal_handler([mq_client])

    def _callback(channel, method, properties, body) -> None:  # noqa: ANN001
        payload = json.loads(body.decode("utf-8"))
        output = engine.handle_manual_result(payload)
        logger.info("人工复核结果已处理: %s", output)
        channel.basic_ack(delivery_tag=method.delivery_tag)

    try:
        mq_client.consume(queue_names.manual_result, _callback)
    except (KeyboardInterrupt, InterruptedError):
        logger.info("收到中断信号，正在停止 ...")
    finally:
        mq_client.stop_consuming()
        mq_client.close()
        logger.info("人工复核结果Worker已退出")


def run_manual_case_worker() -> None:
    """消费 audit.manual.case 队列，同步人工审核案例到 NB 训练 + Milvus 向量库。仅需轻量依赖。"""
    from src.integration.manual_case_sync import ManualCaseSyncService
    from src.mq.manual_case_consumer import ManualCaseConsumer
    from src.mq.rabbitmq_client import RabbitMQClient

    logger.info("启动人工案例同步消费者 (队列: audit.manual.case) ...")
    logger.info("  -> NB 训练语料: data/text_nb/train_samples.jsonl")
    logger.info("  -> Milvus 向量写入: collection=audit_cases")
    mq_client = RabbitMQClient()
    mq_client.connect(declare_queues=["audit.manual.case", "audit.task.dlq"])
    sync_service = ManualCaseSyncService()
    sync_service.connect()
    consumer = ManualCaseConsumer(mq_client=mq_client, sync_service=sync_service)
    _setup_signal_handler([mq_client])
    logger.info("正在监听人工审核案例消息 ... (Ctrl+C 停止)")
    try:
        consumer.start()
    except (KeyboardInterrupt, InterruptedError):
        logger.info("收到中断信号，正在停止 ...")
    finally:
        mq_client.stop_consuming()
        mq_client.close()
        logger.info("人工案例同步Worker已退出")


def _load_config_yaml() -> None:
    """从 config.yaml 加载配置并设置环境变量"""
    import os
    from pathlib import Path

    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        return

    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        return

    # LLM 配置
    llm = config.get("llm", {})
    if llm.get("provider"):
        os.environ.setdefault("JUDGE_PROVIDER", llm["provider"])
        os.environ.setdefault("DESENSITIZE_LLM_PROVIDER", llm["provider"])
    if llm.get("api_key"):
        os.environ.setdefault("JUDGE_API_KEY", llm["api_key"])
        os.environ.setdefault("DESENSITIZE_LLM_API_KEY", llm["api_key"])
        os.environ.setdefault("LLM_API_KEY", llm["api_key"])
        os.environ.setdefault("DASHSCOPE_API_KEY", llm["api_key"])  # VL模型与主LLM共用同一key
    if llm.get("model"):
        os.environ.setdefault("JUDGE_MODEL", llm["model"])
        os.environ.setdefault("DESENSITIZE_LLM_MODEL", llm["model"])
    if llm.get("base_url"):
        os.environ.setdefault("JUDGE_BASE_URL", llm["base_url"])
        os.environ.setdefault("DESENSITIZE_LLM_BASE_URL", llm["base_url"])

    # MinIO 配置
    minio = config.get("minio", {})
    if minio.get("endpoint"):
        os.environ.setdefault("MINIO_ENDPOINT", minio["endpoint"])
    if minio.get("access_key"):
        os.environ.setdefault("MINIO_ACCESS_KEY", minio["access_key"])
    if minio.get("secret_key"):
        os.environ.setdefault("MINIO_SECRET_KEY", minio["secret_key"])
    if minio.get("bucket"):
        os.environ.setdefault("MINIO_BUCKET", minio["bucket"])

    # RabbitMQ 配置
    rabbitmq = config.get("rabbitmq", {})
    if rabbitmq.get("host"):
        os.environ.setdefault("RABBITMQ_HOST", str(rabbitmq["host"]))
    if rabbitmq.get("port"):
        os.environ.setdefault("RABBITMQ_PORT", str(rabbitmq["port"]))
    if rabbitmq.get("username"):
        os.environ.setdefault("RABBITMQ_USERNAME", rabbitmq["username"])
    if rabbitmq.get("password"):
        os.environ.setdefault("RABBITMQ_PASSWORD", rabbitmq["password"])
    if rabbitmq.get("vhost"):
        os.environ.setdefault("RABBITMQ_VHOST", rabbitmq["vhost"])

    # Milvus 配置
    milvus = config.get("milvus", {})
    if milvus.get("host"):
        os.environ.setdefault("MILVUS_HOST", str(milvus["host"]))
    if milvus.get("port"):
        os.environ.setdefault("MILVUS_PORT", str(milvus["port"]))

    logger.info("配置文件已加载: %s", config_path)


WORKERS = {
    "task": run_task_worker,
    "manual_result": run_manual_result_worker,
    "manual_case": run_manual_case_worker,
    "ai-auto": run_ai_auto_worker,
}


def main() -> None:
    # 加载 config.yaml 配置（命令行参数会覆盖这些值）
    _load_config_yaml()

    parser = argparse.ArgumentParser(description="AI Judge System - Worker Launcher")
    parser.add_argument(
        "--worker",
        choices=list(WORKERS.keys()),
        default="manual_case",
        help=(
            "Which worker to start. "
            "task: full audit engine (needs media parsing); "
            "ai-auto: AI auto audit for text+image (NB/CLIP+rewrite+vector+LLM); "
            "manual_result: handle human review callbacks; "
            "manual_case: sync human cases to NB + Milvus (default)"
        ),
    )
    # LLM配置参数
    parser.add_argument(
        "--llm-provider",
        choices=["qwen", "openai", "anthropic"],
        default=None,
        help="LLM provider: qwen (default), openai, anthropic",
    )
    parser.add_argument(
        "--llm-api-key",
        default=None,
        help="LLM API key (or set LLM_API_KEY environment variable)",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="LLM model name (optional, has defaults per provider)",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="LLM API base URL (optional, for proxies/custom endpoints)",
    )

    args = parser.parse_args()

    # 命令行参数覆盖 config.yaml 的设置
    if args.llm_provider:
        os.environ["JUDGE_PROVIDER"] = args.llm_provider
        os.environ["DESENSITIZE_LLM_PROVIDER"] = args.llm_provider
    if args.llm_api_key:
        os.environ["JUDGE_API_KEY"] = args.llm_api_key
        os.environ["DESENSITIZE_LLM_API_KEY"] = args.llm_api_key
        os.environ["LLM_API_KEY"] = args.llm_api_key
        os.environ["DASHSCOPE_API_KEY"] = args.llm_api_key  # VL模型与主LLM共用同一key
    if args.llm_model:
        os.environ["JUDGE_MODEL"] = args.llm_model
        os.environ["DESENSITIZE_LLM_MODEL"] = args.llm_model
    if args.llm_base_url:
        os.environ["JUDGE_BASE_URL"] = args.llm_base_url
        os.environ["DESENSITIZE_LLM_BASE_URL"] = args.llm_base_url

    # 打印LLM配置信息
    provider = args.llm_provider or os.getenv("JUDGE_PROVIDER", "qwen (default)")
    model = args.llm_model or os.getenv("JUDGE_MODEL", "qwen-plus (default)")
    logger.info("=" * 50)
    logger.info("智能审核系统正在启动工作器: %s", args.worker)
    logger.info("LLM配置: provider=%s, model=%s", provider, model)
    logger.info("=" * 50)
    WORKERS[args.worker]()


if __name__ == "__main__":
    main()
