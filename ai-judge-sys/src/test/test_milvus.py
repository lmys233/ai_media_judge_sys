from src.retrieval.milvus_store import MilvusAuditStore


def getDataCount():
    store = MilvusAuditStore(collection_name="audit_cases")
    store.connect()

    existing_count = store.count()
    print("Milvus现有记录数: %d" % existing_count)


if __name__ == "__main__":
    getDataCount()