from src.lib.openai import get_embeddings
from src.lib.mongodb import collection, setup_vector_index
from pathlib import Path
import json

PROJECT_ROOT = (Path(__file__).resolve().parent).parent
JSONL_FILEPATH = PROJECT_ROOT / "data" / \
    "processed" / "rwanda_law_corpus.jsonl"


def load_jsonl_to_dict(filepath: Path) -> dict:
    """Reads JSONL into a dictionary where the key is the document 'id'."""
    data_dict = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                data_dict[doc['id']] = doc
    return data_dict


def process_and_upload(data_dict: dict, batch_size: int = 100):
    """Batches the text, fetches embeddings, and uploads to MongoDB."""

    # Setup the vector index & clear the collection
    setup_vector_index()
    collection.delete_many({})

    doc_ids = list(data_dict.keys())
    total_docs = len(doc_ids)

    print(f"Loaded {total_docs} documents. Starting embedding process...")

    for i in range(0, total_docs, batch_size):
        batch_ids = doc_ids[i:i + batch_size]
        batch_texts = [data_dict[doc_id]['content'] for doc_id in batch_ids]

        print(f"Fetching embeddings for batch {i} to {i + len(batch_ids)}...")

        embeddings = get_embeddings(batch_texts)

        mongodb_payload = []
        for doc_id, embedding_vector in zip(batch_ids, embeddings):
            doc = data_dict[doc_id]
            doc['_id'] = doc.pop('id')
            doc['embedding'] = embedding_vector
            mongodb_payload.append(doc)

        collection.insert_many(mongodb_payload)
        print(f"Inserted batch into MongoDB.")


if __name__ == "__main__":
    raw_data_dict = load_jsonl_to_dict(JSONL_FILEPATH)
    process_and_upload(raw_data_dict, batch_size=100)

    print("Ingestion complete!")
