"""Milvus Collection 迁移脚本

用途：将旧 collection（缺少 image_url 字段）迁移到新 schema，同时保留向量数据

使用方式：
1. 先停掉所有使用 Milvus 的 Worker
2. 运行脚本：python scripts/migrate_milvus_collection.py
3. 脚本会自动：导出 -> 删除旧 collection -> 重连让程序重建 -> 重新导入
4. 重启 Worker
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 确保 src 目录在 path 里
sys.path.insert(0, str(Path(__file__).parent.parent))

from pymilvus import connections, utility, Collection


def get_milvus_client():
    host = os.getenv("MILVUS_HOST", "localhost")
    port = int(os.getenv("MILVUS_PORT", "19530"))
    connections.connect(host=host, port=port)
    return f"{host}:{port}"


def export_collection(collection_name: str) -> list[dict]:
    """导出 collection 中所有数据（含向量）"""
    if not utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' 不存在，无需迁移")
        return []

    collection = Collection(collection_name)
    collection.load()

    # 用 query 查所有字段（包括 embedding 向量），Milvus 支持 output_fields=["*"]
    all_data = []
    batch_size = 500
    offset = 0

    while True:
        results = collection.query(
            expr="task_id != ''",
            output_fields=["*"],  # 查询所有字段，包括向量
            limit=batch_size,
            offset=offset
        )
        if not results:
            break
        all_data.extend(results)
        offset += batch_size
        print(f"  已导出 {len(all_data)} 条 ...")

    collection.release()
    print(f"共导出 {len(all_data)} 条记录")
    if all_data:
        print(f"  字段: {list(all_data[0].keys())}")
    return all_data


def drop_collection(collection_name: str) -> None:
    """删除旧 collection"""
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"已删除旧 collection: {collection_name}")


def reconnect_and_recreate() -> None:
    """让 MilvusStore 重新创建 collection"""
    from src.retrieval.milvus_store import MilvusAuditStore
    store = MilvusAuditStore()
    store.connect()
    print("新 collection 已创建（含 image_url 字段）")


def import_data(collection_name: str, data: list[dict]) -> None:
    """重新导入数据"""
    if not data:
        print("无数据需要导入")
        return

    collection = Collection(collection_name)
    collection.load()

    # 按新 schema 字段顺序组织数据
    entities = []
    for record in data:
        entities.append({
            "id": record["id"],
            "task_id": record["task_id"],
            "media_type": record["media_type"],
            "violation_type": record["violation_type"],
            "risk_score": float(record["risk_score"]),
            "source": record["source"],
            "created_at": record["created_at"],
            "model_version": record["model_version"],
            "human_verified": bool(record["human_verified"]),
            "description": record["description"],
            "embedding": record["embedding"],
            "media_url": record.get("media_url") or record.get("image_url") or "",  # 兼容旧字段名
        })

    # 分批插入
    batch_size = 100
    for i in range(0, len(entities), batch_size):
        batch = entities[i:i + batch_size]
        collection.insert(batch)
        print(f"  已导入 {min(i + batch_size, len(entities))}/{len(entities)} 条 ...")

    collection.flush()
    collection.release()
    print(f"导入完成，共 {len(entities)} 条（向量数据已保留）")


def main():
    collection_name = "audit_cases"

    print("=" * 50)
    print("Milvus Collection 迁移脚本")
    print("=" * 50)

    print("\n[1/4] 连接 Milvus ...")
    get_milvus_client()
    print("连接成功")

    print(f"\n[2/4] 导出 '{collection_name}' 中的数据 ...")
    data = export_collection(collection_name)

    print(f"\n[3/4] 删除旧 collection ...")
    drop_collection(collection_name)

    print(f"\n[4/5] 重建 collection（让 MilvusStore 创建新 schema）...")
    reconnect_and_recreate()

    print(f"\n[5/5] 导入数据 ...")
    import_data(collection_name, data)

    print("\n" + "=" * 50)
    print("迁移完成！向量数据已保留。")
    print("现在可以重启 Worker，程序会使用新的 12 字段 schema（含 image_url）")


if __name__ == "__main__":
    main()
