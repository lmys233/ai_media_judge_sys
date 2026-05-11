"""
初始化种子数据到Milvus向量数据库和NB分类器

Usage:
    python scripts/init_seed_data.py [--csv data/init_review_data.csv]

CSV格式（text,label,source,verified）:
    text,label,source,verified
    今天天气真好,normal,human,true
    暴力内容示例,violence,human,true
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature.embedder import build_embedding_model
from src.prelabel.text_nb import NaiveBayesTextClassifier
from src.retrieval.milvus_store import MilvusAuditStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_csv_data(csv_path: Path) -> list[dict]:
    """从CSV文件加载种子数据"""
    records = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "text": row["text"].strip(),
                "label": row["label"].strip(),
                "source": row.get("source", "seed").strip(),
                "verified": row.get("verified", "true").lower() == "true",
            })
    return records


def init_milvus(records: list[dict], embedder) -> int:
    """将数据写入Milvus向量数据库"""
    logger.info("连接Milvus...")
    store = MilvusAuditStore(collection_name="audit_cases")
    store.connect()

    existing_count = store.count()
    logger.info("Milvus现有记录数: %d", existing_count)

    # 生成embeddings
    texts = [r["text"] for r in records]
    logger.info("正在生成 %d 条文本的embedding向量...", len(texts))
    embeddings = embedder.embed_documents(texts)

    # 批量构建记录
    batch_records = []
    for i, record in enumerate(records):
        record_id = str(uuid.uuid4())
        task_id = f"seed_{i:04d}"
        created_at = datetime.utcnow().isoformat()
        label = record["label"]
        risk_score = 0.0 if label == "normal" else 0.85

        batch_records.append({
            "record_id": record_id,
            "task_id": task_id,
            "media_type": "text",
            "violation_type": label,
            "risk_score": risk_score,
            "source": record["source"],
            "created_at": created_at,
            "model_version": "seed-v1",
            "human_verified": record["verified"],
            "description": record["text"],
            "embedding": embeddings[i],
            "media_url": "",  # 文本数据无 media_url，留空
        })

    # 批量写入
    logger.info("正在批量写入 %d 条记录到Milvus...", len(batch_records))
    success_count = store.batch_insert(batch_records, flush=True)
    logger.info("Milvus写入完成: %d 条记录", success_count)
    logger.info("Milvus总记录数: %d", store.count())
    return success_count


def init_nb_classifier(records: list[dict]) -> int:
    """训练NB分类器"""
    logger.info("初始化NB分类器...")
    classifier = NaiveBayesTextClassifier()

    # 添加训练样本
    valid_labels = {"normal", "abuse", "violence", "porn", "politics"}
    count = 0
    for record in records:
        if record["label"] in valid_labels:
            classifier.append_training_sample(
                text=record["text"],
                label=record["label"],
                source=record["source"],
                verified=record["verified"],
            )
            count += 1

    # 重新训练模型
    logger.info("正在重新训练NB模型（%d 条样本）...", count)
    retrained = classifier.retrain_from_corpus()
    logger.info("NB模型训练完成: %s", "成功" if retrained else "失败")

    return count


def verify_milvus_search(store: MilvusAuditStore, embedder, query_texts: list[str]) -> None:
    """验证Milvus检索功能"""
    logger.info("\n验证Milvus检索功能...")
    for query in query_texts:
        vector = embedder.embed_query(query)
        results = store.search(query_vector=vector, limit=3)
        logger.info("查询: %s", query[:50])
        for r in results:
            logger.info(
                "  - [score=%.4f] %s: %s",
                r.get("score", 0),
                r.get("violation_type", "?"),
                str(r.get("description", ""))[:40],
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="初始化种子数据到审核引擎")
    parser.add_argument(
        "--csv",
        type=str,
        default="data/init_review_data.csv",
        help="CSV数据文件路径",
    )
    parser.add_argument(
        "--skip-milvus",
        action="store_true",
        help="跳过Milvus写入",
    )
    parser.add_argument(
        "--skip-nb",
        action="store_true",
        help="跳过NB训练",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="训练后验证检索功能",
    )
    args = parser.parse_args()

    csv_path = ROOT / args.csv
    if not csv_path.exists():
        logger.error("CSV文件不存在: %s", csv_path)
        sys.exit(1)

    # 加载数据
    logger.info("从 %s 加载数据...", csv_path)
    records = load_csv_data(csv_path)
    logger.info("加载了 %d 条记录", len(records))

    # 统计标签分布
    label_counts = {}
    for r in records:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
    logger.info("标签分布: %s", json.dumps(label_counts, ensure_ascii=False))

    # 构建embedding模型
    embedder = build_embedding_model()

    # 写入Milvus
    milvus_count = 0
    if not args.skip_milvus:
        milvus_count = init_milvus(records, embedder)

    # 训练NB
    nb_count = 0
    if not args.skip_nb:
        nb_count = init_nb_classifier(records)

    # 验证
    if args.verify and not args.skip_milvus:
        store = MilvusAuditStore(collection_name="audit_cases")
        store.connect()
        verify_milvus_search(
            store, embedder,
            ["今天天气真好", "暴力威胁内容", "色情视频"],
        )

    logger.info("\n初始化完成!")
    logger.info("  - Milvus写入: %d 条", milvus_count)
    logger.info("  - NB训练样本: %d 条", nb_count)


if __name__ == "__main__":
    main()
