from pathlib import Path
import json
import chromadb
from chromadb.utils import embedding_functions

JSONL_PATH = Path(r"C:\Users\91951\OneDrive\Desktop\pythonProject\leetcode\Ai-ML-Projects\credit-card-rag\data\processed\all_card_snippets_enriched.jsonl")
DB_DIR = Path("vectordb/chroma_db")
COLLECTION_NAME = "credit_card_chunks"


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def safe_meta(meta: dict) -> dict:
    clean = {}
    for k, v in meta.items():
        if v is None:
            clean[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


def main():
    DB_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(DB_DIR))

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    existing = collection.get()
    if existing.get("ids"):
        collection.delete(ids=existing["ids"])

    ids = []
    docs = []
    metas = []

    for obj in load_jsonl(JSONL_PATH):
        ids.append(obj["id"])
        docs.append(obj["document"])
        metas.append(safe_meta(obj["metadata"]))

    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=docs[i:i+batch_size],
            metadatas=metas[i:i+batch_size],
        )

    print(f"Stored {len(ids)} chunks in ChromaDB")
    print(f"DB path: {DB_DIR}")
    print(f"Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()