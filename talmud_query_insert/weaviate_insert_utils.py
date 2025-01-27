from embed_utils import get_nvidia_embeddings
import requests
import json
import os
from tqdm import tqdm

# Replace with your Weaviate endpoint and API keys
WEAVIATE_URL = "https://1exconarfknip3xtkwcg.c0.us-east1.gcp.weaviate.cloud"
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

SCHEMA_API_URL = f"{WEAVIATE_URL}/v1/schema"
BATCH_API_URL = f"{WEAVIATE_URL}/v1/batch/objects"
BATCH_SIZE = 100

def insert_data_into_weaviate(chunks_sample):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {WEAVIATE_API_KEY}",
        "X-OpenAI-Api-Key": OPENAI_API_KEY
    }

    def send_data(data):
        response = requests.post(BATCH_API_URL, headers=headers, json=data)
        return response.status_code

    # Delete existing schema if needed
    requests.delete(f"{SCHEMA_API_URL}/Question", headers=headers)

    # Create schema
    schema = {
        "class": "Question",
        "vectorizer": "text2vec-openai",
        "moduleConfig": {
            "generative-openai": {}
        }
    }
    requests.post(SCHEMA_API_URL, headers=headers, json=schema)

    # Process and send data
    batch_data = {"objects": []}
    lines_processed = 0

    for chunk in tqdm(chunks_sample, desc="Processing chunks"):
        nvidia_embedding = get_nvidia_embeddings([chunk.get("english_text", "")])[0]

        obj = {
            "class": "Question",
            "properties": {
                "english_text": chunk.get("english_text", ""),
                "book_name": chunk.get("book_name", ""),
                "page_number": chunk.get("page_number", "")
            },
            "vector": nvidia_embedding
        }
        batch_data["objects"].append(obj)
        lines_processed += 1

        if lines_processed == BATCH_SIZE:
            send_data(batch_data)
            batch_data = {"objects": []}
            lines_processed = 0

    # Send any remaining data
    if batch_data["objects"]:
        send_data(batch_data)

    print("Import finished.")

def query_weaviate(search_concept, limit=2):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {WEAVIATE_API_KEY}"
    }

    search_vector = get_nvidia_embeddings([search_concept])[0]

    query = """
    query($vector: [Float!]!, $limit: Int!) {
      Get {
        Question (
          limit: $limit
          nearVector: {
            vector: $vector
          }
        ) {
          english_text
          book_name
          page_number
        }
      }
    }
    """

    variables = {
        "vector": search_vector,
        "limit": limit
    }

    response = requests.post(
        f"{WEAVIATE_URL}/v1/graphql",
        headers=headers,
        json={"query": query, "variables": variables}
    )

    return response.json()

# Example usage:
# insert_data_into_weaviate(chunks_sample)
# result = query_weaviate("biology")
# print(json.dumps(result, indent=2))