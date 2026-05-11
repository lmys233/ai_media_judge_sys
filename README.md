# AI 智能审核系统

多模态内容安全审核平台，支持文本、图片、视频的自动审核与人工复核协同工作流。

## 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│                      审核管理平台 (Spring Boot)                │
├──────────────────────────────────────────────────────────────┤
│   RabbitMQ  ←→  AI 审核引擎 (Python)  ←→  Milvus 向量库     │
│                                                    │        │
│                                               MinIO 对象存储  │
└──────────────────────────────────────────────────────────────┘
```

**核心链路：**

- **文本审核：** NB 朴素贝叶斯预打标 → 脱敏重写 → 向量检索 → LLM 判定 → 置信度路由
- **图片审核：** CLIP Zero-shot 预打标 → VL 描述生成 → 向量检索 → VL 判定 → 置信度路由
- **视频审核：** 关键帧提取 → 逐帧 CLIP 打标 → VL 描述 → 向量检索 → LLM 判定

**多智能体复核（中置信度 0.4~0.8）：** ReAct 决策者 + CoT 逻辑/证据辅助者协同复核，自适应路由。

## 环境要求

| 依赖 | 版本要求 | 说明 |
|------|---------|------|
| Python | >= 3.14 | 推荐 3.14 |
| JDK | >= 17 | 仅审核管理平台需要 |
| Docker & Docker Compose | latest | 运行基础设施服务 |
| 显存/内存 | >= 8GB | CLIP 模型 + LLM 推理 |

## 基础设施（Docker）

```bash
cd judge-plaform/judge-plaform
docker-compose up -d
```

启动的服务：

| 服务 | 端口 | 说明 |
|------|------|------|
| RabbitMQ | 5672 / 15672 | 消息队列 / 管理 UI |
| MinIO | 9000 / 9001 | 对象存储 / 控制台 |
| Milvus | 19530 / 9091 | 向量数据库 / WebUI |
| etcd | 2379 | Milvus 依赖的分布式存储 |
| Attu | 8000 | Milvus 可视化工具 |

> 首次启动后，进入 MinIO 控制台 (http://localhost:9001) 创建一个名为 `judge-media` 的 bucket。

## 数据库初始化

### MySQL（审核管理平台）

脚本位于 `judge-plaform/judge-plaform/src/main/resources/sql/`：

| 文件 | 说明 |
|------|------|
| `judge_case.sql` | 审核案例主表（含 Snowflake ID、媒体类型、违规标签、审核状态流转等） |
| `failure_case.sql` | 审核失败记录表（记录重试耗尽后的失败任务） |

按顺序执行：

```bash
# 先建主表
mysql -u root -p judge_platform < judge-plaform/src/main/resources/sql/judge_case.sql

# 建失败记录表
mysql -u root -p judge_platform < judge-plaform/src/main/resources/sql/failure_case.sql
```

> `alter_is_viewed.sql` 是表结构迁移脚本，需在 `judge_case` 表已存在且有 is_viewed 字段需求时执行。

#### judge_case 表结构

| 字段 | 类型 | 说明 |
|------|------|------|
| case_id | BIGINT (PK, Snowflake) | 案例 ID |
| media_id | BIGINT | 业务媒体 ID |
| content_text | TEXT | 文本内容（media_type=text 时） |
| media_type | VARCHAR(16) | text / image / video |
| media_url | VARCHAR(512) | MinIO 文件 URL |
| violation_types | JSON | 违规类型数组 |
| review_reason | VARCHAR(50) | 审核理由 |
| evidence | JSON | 证据数组 |
| source | VARCHAR(16) | human / ai |
| reviewer_id | BIGINT | 审核员 ID |
| review_time | DATETIME | 审核时间 |
| audit_status | VARCHAR(32) | 审核状态流转 |
| ai_result | JSON | AI 审核结果快照 |
| ai_processed_at | DATETIME | AI 处理时间 |
| retry_count | INT | 重试次数 |
| is_viewed | TINYINT(1) | 是否已查看 |

### Milvus 向量数据库

Collection 名称为 `audit_cases`，由 AI 审核引擎启动时自动创建（`milvus_store.py:_create_collection()`），无需手动执行 SQL。

#### Collection Schema

| 字段 | 类型 | 维度/长度 | 说明 |
|------|------|-----------|------|
| id | VARCHAR (PK) | 128 | UUID |
| task_id | VARCHAR | 128 | 任务 ID |
| media_type | VARCHAR | 32 | 媒体类型 |
| violation_type | VARCHAR | 32 | 违规类型 |
| risk_score | FLOAT | - | 风险分数 |
| source | VARCHAR | 64 | 数据来源 |
| created_at | VARCHAR | 64 | 创建时间 |
| model_version | VARCHAR | 64 | 模型版本 |
| human_verified | BOOL | - | 是否人工验证 |
| description | VARCHAR | 4096 | 脱敏描述 |
| evidence | VARCHAR | 4096 | 详细证据 |
| embedding | FLOAT_VECTOR | 512 | CLIP 向量 |
| media_url | VARCHAR | 512 | 媒体文件 URL |

#### 索引

| 字段 | 索引类型 | 参数 |
|------|---------|------|
| embedding | HNSW | COSINE 距离, M=16, efConstruction=256 |
| risk_score | STL_SORT | - |
| media_type | 默认 | - |
| violation_type | 默认 | - |
| source | 默认 | - |
| human_verified | 默认 | - |

## 快速启动

### 1. 克隆并安装

```bash
git clone <repo-url>
cd ai-judge-sys

# 创建虚拟环境
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置文件

复制配置文件并填入实际值：

```bash
cp config.yaml.example config.yaml
```

编辑 `config.yaml`：

```yaml
llm:
  provider: qwen                  # qwen | openai | anthropic
  api_key: "your-api-key-here"    # 必填
  model: ""                       # 可选，默认 qwen-plus
  base_url: ""                    # 可选，用于代理/自定义端点

minio:
  endpoint: "http://localhost:9000"
  access_key: "minioadmin"
  secret_key: "minioadmin123"
  bucket: "judge-media"

rabbitmq:
  host: "localhost"
  port: 5672
  username: "admin"
  password: "admin123"
  vhost: "/"

milvus:
  host: "localhost"
  port: 19530
```

支持的环境变量覆盖（优先级高于 config.yaml）：

```bash
# LLM
export JUDGE_PROVIDER=qwen
export JUDGE_API_KEY=sk-xxx
export JUDGE_MODEL=qwen-plus
export DESENSITIZE_LLM_API_KEY=sk-xxx  # 脱敏专用（可独立配置）
export DASHSCOPE_API_KEY=sk-xxx        # VL 模型（图片审核）

# 基础设施
export RABBITMQ_HOST=localhost
export MILVUS_HOST=localhost
export MINIO_ENDPOINT=http://localhost:9000
```

### 3. 初始化种子数据

```bash
python scripts/init_seed_data.py --csv data/init_review_data.csv
```

### 4. 启动 Worker

```bash
# AI 自动审核 Worker（核心，文本 + 图片）
python -m src.app --worker ai-auto --llm-provider qwen --llm-api-key sk-xxx

# 人工案例同步 Worker
python -m src.app --worker manual_case

# 人工复核结果处理 Worker
python -m src.app --worker manual_result

# 完整审核引擎 Worker（含媒体解析，需 torch/cv2）
python -m src.app --worker task
```

## 模块说明

### 预打标引擎 (PreLabel)

| 模块 | 文件 | 技术 | 功能 |
|------|------|------|------|
| 文本分类器 | `text_nb.py` | 朴素贝叶斯 + 关键词规则 | 多标签：hard 关键词短路、soft 关键词加分、NB 概率预测 |
| 图片 CLIP 分类器 | `image_quick_label.py` | CLIP Zero-shot 图文相似度 | 多标签分类，5 类违规文本描述，阈值 > 0.25 |
| 图片 MobileNet | `image_mobilenet.py` | MobileNet V3 Small | 传统分类器（旧链路降级兜底） |
| 预打标管道 | `pipeline.py` | 策略模式 | 按媒体类型路由打标器 |

### 脱敏处理器 (Desensitize)

三级降级架构：

1. **Hard 规则层** — 代码内置敏感词正则替换（如 "操你妈" → `[违规言语]`）
2. **Soft 规则层** — YAML 配置热加载关键词替换（`text_keyword_overrides.yaml`）
3. **LLM 语义层** — LLM 语义等价重写，保留意图但不含违禁词（入库用）

LLM 不可用时自动降级到规则层。

### 向量检索 (Retrieval)

- **编码模型：** CLIP ViT-B/16，512 维统一语义空间
- **向量库：** Milvus 2.6 + HNSW 索引（余弦距离）
- **检索策略：** 混合检索（媒体类型 OR 违规类型）→ 不足 3 条时降级纯向量检索
- **重排序：** 4 信号加权（违规类型匹配 + 人工验证加分 + 向量分 ×0.6 + 风险分接近度 ×0.3）

### 审核判定 (Reasoning)

| 模块 | 技术 | 说明 |
|------|------|------|
| 文本判定器 | LLM（Qwen/OpenAI/Anthropic） | 原文 + 相似案例，结构化 JSON 输出 |
| 图片判定器 | Qwen-VL-Max | 多模态输入直接审核图片 |
| 多智能体复核 | ReAct + CoT × 2 + Meta 监督器 | 中置信度场景三智能体协同 |
| 置信度路由 | 三层路由 | >= 0.8 自动 / 0.4~0.8 复核 / < 0.4 人工 |

### 消息队列 (RabbitMQ)

| 队列 | 用途 |
|------|------|
| `audit.task` | 审核任务（支持重试，最多 3 次） |
| `audit.task.dlq` | 死信队列 |
| `audit.failure` | 重试耗尽的任务 |
| `audit.manual` | 人工审核任务 |
| `audit.manual.result` | 人工审核结果回传 |
| `audit.manual.case` | 人工案例同步 |
| `audit.result` | AI 审核结果 |

### 数据飞轮闭环

```
高置信度 AI 审核结果 ──→ Milvus 入库（扩充向量库）
                      ──→ NB 增量训练（提升预打标准确率）
人工审核结果           ──→ Milvus（黄金案例）
                      ──→ NB 重训练（verified=true）
```

## 冷启动

系统上线初期训练数据不足时：

- NB 分类器内置 10 条种子样本，支持增量追加
- 弱标签策略：置信度 >= 0.9 才自动入库，保证冷启动阶段数据质量
- CLIP Zero-shot 无需训练数据，开箱即用
- 支持通过 `init_review_data.csv` 批量导入种子案例

## 项目结构

```
ai-judge-sys/
├── config.yaml              # 主配置文件
├── config.yaml.example      # 配置模板
├── requirements.txt         # Python 依赖
├── src/
│   ├── app.py               # 入口，Worker 启动器
│   ├── engine/              # 审核引擎（文本/图片/视频）
│   ├── prelabel/            # 预打标引擎（NB/CLIP/MobileNet）
│   ├── feature/             # 特征工程（脱敏/编码）
│   ├── retrieval/           # 向量检索（Milvus/Reranker）
│   │   └── milvus_store.py  # Milvus collection 创建与 CRUD
│   ├── reasoning/           # 审核判定 + 多智能体复核
│   │   ├── multi_agent/     # ReAct + CoT 协同复核
│   │   └── image_judge.py   # VL 图片审核
│   ├── decision/            # 置信度路由
│   ├── mq/                  # 消息队列（RabbitMQ）
│   ├── integration/         # 外部系统集成
│   ├── models/              # 数据模型 + LLM 工厂
│   ├── tool/                # 工具类（图片处理/视频抽帧）
│   └── learning/            # 主动学习
├── scripts/
│   ├── init_seed_data.py    # 种子数据初始化
│   └── run_text_only_demo.py
├── data/                    # 训练数据
├── policy/
│   └── text_keyword_overrides.yaml  # 关键词热加载配置
└── judge-plaform/           # Spring Boot 审核管理平台
    └── src/main/resources/sql/
        ├── judge_case.sql          # 审核案例表
        ├── failure_case.sql        # 失败记录表
        └── alter_is_viewed.sql     # [迁移] 添加 is_viewed 字段
```

## LLM 配置

支持三种 LLM 提供商：

```bash
# 通义千问（默认）
--llm-provider qwen --llm-api-key sk-xxx

# OpenAI
--llm-provider openai --llm-api-key sk-xxx

# Anthropic
--llm-provider anthropic --llm-api-key sk-ant-xxx
```

多模型职责分离：

| 用途 | 环境变量 | 默认模型 | 温度 |
|------|---------|---------|------|
| 审核判定 | `JUDGE_*` | qwen-plus | 0.0 |
| 文本脱敏重写 | `DESENSITIZE_LLM_*` | qwen-plus | 0.3 |
| 图片/VL 判定 | `DASHSCOPE_API_KEY` | qwen-vl-max | 0.0 |

## 数据文件参考

```
data/
├── text_nb/                    # NB 分类器数据
│   ├── train_samples.jsonl     # 训练样本
│   └── model_hash.txt          # 模型哈希校验
├── init_review_data.csv        # 种子数据
├── error_memory.jsonl          # 纠偏案例记录 [TBD]
└── reference_errors.yaml       # 生效的纠偏案例配置 [TBD]
```
