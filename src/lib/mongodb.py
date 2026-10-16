from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from src.config.env import MONGO_URI, DB_NAME, COLLECTION_NAME

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]


def setup_vector_index():
    """
    Creates an Atlas Vector Search index on the 'embedding' field.
    """
    search_index_model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 1536,
                    "similarity": "dotProduct"
                }
            ]
        },
        name="vector_index",
        type="vectorSearch"
    )

    try:
        collection.create_search_index(model=search_index_model)
        print("Successfully initiated vector index creation.")
        print("Note: It may take a few minutes for MongoDB Atlas to build the index in the background.")
    except Exception as e:
        print(f"Index creation skipped or failed (it may already exist): {e}")
