from openai import OpenAI
from src.config.env import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Takes a list of strings and returns a list of embedding vectors.
    Defaults to 1536 dimensions for text-embedding-3-small.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]
