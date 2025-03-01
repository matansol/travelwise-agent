# from langchain.vectorstores import Qdrant
import os
import sys
import warnings

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from src.data_collector import (
    create_data_lists,
    generate_activity_data,
    generate_hotel_data,
)
from src.env_variables import get_qdrant_credentials

warnings.simplefilter("ignore")

# Initialize the Qdrant database client
qdrant_url, qdrant_api_key = get_qdrant_credentials()
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)


def clean_collection(collection_name, vector_dim=384):
    """
    Clean the specified Qdrant collection by deleting all its vectors.
    :param collection_name: The name of the collection to clean.
    :param vector_dim: The dimensionality of the vectors in the collection.
    """
    qdrant_client.delete_collection(collection_name=collection_name)
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )
    print(f"Collection '{collection_name}' has been cleaned and recreated.")


def delete_collection(collection_name):
    """
    Delete the specified Qdrant collection.
    :param collection_name: The name of the collection to delete.
    """
    qdrant_client.delete_collection(collection_name=collection_name)
    print(f"Collection '{collection_name}' has been deleted.")


def delete_all_collections():
    """
    Delete all Qdrant collections.
    """
    collections = qdrant_client.get_collections()
    for collection in collections.collections:
        collection_name = collection.name
        qdrant_client.delete_collection(collection_name=collection.name)
        print(f"Collection '{collection_name}' has been deleted.")


# create the collections and the syntetic data
def create_collection_if_not_exists(collection_name: str, vector_dim: int = 5):
    """
    Create a collection in Qdrant if it does not already exist.
    :param collection_name: The name of the collection to create.
    :param vector_dim: The dimensionality of the vectors in the collection.
    """
    try:
        # Try to retrieve the collection information.
        collection_info = qdrant_client.get_collection(collection_name=collection_name)
        if collection_info:
            print(f"Collection '{collection_name}' already exists.")
            return
    except Exception as e:
        # If an exception occurs, assume the collection does not exist.
        print(f"Collection '{collection_name}' not found. Proceeding to create it.")

    # Create the collection.
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )
    print(f"Collection '{collection_name}' created.")


def insert_data_into_qdrant(collection_name: str, records: list):
    """ Insert the provided records into the specified Qdrant collection.
        :param collection_name: The name of the collection to insert the records into.
        :param records: The list of records to insert into the collection."""
    points = []
    for record in records:
        record_vector = record["vector"].tolist()
        # Create a point where the vector and payload are set.
        point = PointStruct(
            id=record["id"],
            vector=record_vector,
            payload={key: value for key, value in record.items() if key not in ["id", "vector"]},
        )
        points.append(point)
    # Upsert the points into the collection.
    try:
        response = qdrant_client.upload_points(collection_name=collection_name, points=points)
        print(f"Inserted {len(points)} points into '{collection_name}' collection.")
    except Exception as e:
        print(f"ERROR: Failed to upload points - {e}")


def create_and_load_collections():
    """ Create the collections in Qdrant and load the synthetic data into them. """
    # Intialize embeddings model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # ('multi-qa-mpnet-base-dot-v1') - stronget model
    VECTOR_DIM = embedding_model.get_sentence_embedding_dimension()

    # Create collections for flights, hotels, and activities.
    create_collection_if_not_exists("flights", VECTOR_DIM)
    create_collection_if_not_exists("hotels", VECTOR_DIM)
    create_collection_if_not_exists("activities", VECTOR_DIM)

    hotels, activities, flights = create_data_lists(embedding_model)

    # Insert the synthetic data into their respective Qdrant collections.
    insert_data_into_qdrant("flights", flights)
    insert_data_into_qdrant("hotels", hotels)
    insert_data_into_qdrant("activities", activities)

    # shows the collections
    collections = qdrant_client.get_collections()
    print(f"Available collections: {collections.collections}")

    # Fetch and print an example from each collection
    for collection in collections.collections:
        points = qdrant_client.scroll(collection_name=collection.name, limit=1)
        print(f"Example from collection '{collection.name}': {points}")



def check_collections():
    """ Check the collections in Qdrant and print an example from each collection. """
    collections = qdrant_client.get_collections()
    print(f"Available collections: {collections.collections}")

    # Fetch and print an example from each collection
    for collection in collections.collections:
        points = qdrant_client.scroll(collection_name=collection.name, limit=1, with_vectors=True)
        print(f"Example from collection '{collection.name}': {points}")

    collection_info = qdrant_client.get_collection(collection_name="flights")
    print(collection_info)


def check_vector_db():
    """ Check the vector database by querying it with a sample text query. """
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_text = "luxury hotel in Sydney"
    query_vector = embedding_model.encode(query_text)

    results = qdrant_client.search(
        collection_name="flights",
        query_vector=query_vector,
        limit=5,
    )

    # Process and print the results
    for point in results:
        print(f"ID: {point.id}")
        print(f"Payload: {point.payload}")
        print(f"Simillarity Score: {point.score}")  # if Qdrant returns the distance score
        print("-----")


if __name__ == "__main__":
    # delete_all_collections()
    # create_and_load_collections()
    # check_collections()
    check_vector_db()
