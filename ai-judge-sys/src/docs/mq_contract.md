# RabbitMQ 消息契约

本项目使用以下队列：

- `audit.task`：待审核任务入站队列
- `audit.manual`：下发给人工审核系统（Spring 消费）
- `audit.manual.result`：人工审核结果回传队列

通用消息字段：

- `schema_version`：消息结构版本
- `trace_id`：全链路追踪 ID
- `task_id`：任务唯一 ID
- `dedupe_key`：幂等键，默认可与 `task_id` 相同

重试与幂等策略：

1. 消费端使用手动 ACK。
2. 任务处理失败时，增加 `retry_count` 并重新投递。
3. 若 `retry_count >= max_retry`，投递到 `audit.task.dlq`。
4. PoC 阶段使用内存去重，避免重复执行。
5. 生产环境建议改为 Redis 去重。
