# 人工审核案例契约（Spring -> 审核引擎）

队列名称：

- `audit.manual.case`

用途：

- Spring Web 人工审核项目发布已确认的审核案例。
- 审核引擎消费后同步更新：
  - NB 文本训练语料（`data/text_nb/train_samples.jsonl`）
  - Milvus 向量库（`audit_cases`）

## JSON 消息结构

```json
{
  "schema_version": "1.0.0",
  "trace_id": "uuid-xxxx",
  "case_id": "case-20260320-0001",
  "media_type": "text",
  "content_text": "你这个人真垃圾，滚开",
  "violation_types": ["abuse", "violence"],
  "review_reason": "存在辱骂并带有威胁语义",
  "evidence": ["垃圾", "滚开"],
  "source": "human"
}
```

## 字段约束

- `schema_version`：消息结构版本，当前为 `1.0.0`
- `trace_id`：全链路追踪 ID（UUID）
- `media_type`：`text | image | video`
- `violation_types`：违规类型集合，元素取值 `normal | abuse | violence | porn | politics | other`
- `review_reason`：人工审核给出的简短理由描述
- `source`：仅支持 `human | ai`
- `content_text`：当 `media_type=text` 时，NB 更新必须提供文本内容

## 处理流程

1. 使用 `ManualAuditCaseMessage` 校验消息结构。
2. 若 `media_type=text` 且标签属于 NB 支持范围：
   - 原始文本 + 标签追加到 `data/text_nb/train_samples.jsonl`
   - 在进程内重训 NB（仅在 sklearn 可用时）
3. 文本去敏感化（用于 RAG 向量存储）：
   - 优先使用 LLM 语义重写（`sanitize_for_rag`）：同义改写去除违禁词，辱骂替换为 `[违规言语]` 等
   - 未配置 LLM 时回退到基于正则的关键词替换
   - 环境变量：`DESENSITIZE_LLM_PROVIDER=openai`，`DESENSITIZE_LLM_MODEL=gpt-4.1-mini`
4. 构造向量化文本（仅包含去敏感化内容）并调用 embedding 模型：
   - 推荐：`text-embedding-3-large`（OpenAI）
   - 未配置 API 时：回退到 hash embedding
5. 审核引擎根据 `violation_types` 自动计算风险分并写入 Milvus，同时标记 `human_verified=true`。
6. Milvus `description` 字段仅存储去敏感化后的文本，不存储原始违规内容。
