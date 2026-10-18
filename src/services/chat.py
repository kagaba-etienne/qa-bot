from src.lib.mongodb import collection
from src.lib.openai import get_embeddings
from src.lib.qwen import FINETUNED_MODEL, build_messages, qwen


def _retrieve_context(question: str, limit: int = 4) -> str:
    """Return the top matching legal chunks formatted for the model prompt."""
    embedding = get_embeddings([question])[0]
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": 50,
                "limit": limit,
            }
        },
        {"$project": {"content": 1, "metadata.citation": 1}},
    ]
    docs = list(collection.aggregate(pipeline))
    parts = []
    for index, doc in enumerate(docs, start=1):
        citation = (doc.get("metadata") or {}).get("citation")
        content = doc.get("content") or ""
        if citation:
            header = f"Document [{index}] (Citation: {citation}):"
        else:
            header = f"Document [{index}]:"
        parts.append(f"{header}\n{content}")
    return "\n\n".join(parts)


def chat(user_prompt: str) -> str:
    """Embed a question, attach the top 4 articles, and answer with the finetuned model."""
    context = _retrieve_context(user_prompt)
    return qwen(build_messages(user_prompt, context), model=FINETUNED_MODEL)
