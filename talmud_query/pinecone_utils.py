import openai
import json as JSON
from langchain.embeddings.openai import OpenAIEmbeddings
import asyncio
import httpx
from pydantic import BaseModel, create_model
from typing import Union, Optional, List
from langsmith.wrappers import wrap_openai
from langsmith.run_helpers import get_current_run_tree
from langsmith import traceable, Client
import os
import itertools
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import uuid
from talmud_query.prompts import *
from talmud_query.config import *
from talmud_query.embed_utils import embed_text_openai

def get_index_endpoint(api_key=PINECONE_API_KEY, index_name=INDEX_NAME):
    url = f"https://api.pinecone.io/indexes/{index_name}"
    headers = {"Api-Key": api_key}

    response = httpx.get(url, headers=headers)
    response.raise_for_status()
    response_json = response.json()

    if "host" in response_json:
        return response_json["host"]
    else:
        raise KeyError(f"'host' not found in the response: {response_json}")

def upsert_vectors(vectors, api_key=PINECONE_API_KEY, index_endpoint=None, namespace=NAMESPACE):
    url = f"https://{index_endpoint}/vectors/upsert"
    headers = {
        "Api-Key": api_key,
        "Content-Type": "application/json"
    }
    data = {"vectors": vectors, "namespace": namespace}

    response = httpx.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()

def chunks(iterable, batch_size=200):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

def store_pinecone_embeddings_in_batches(embeddings, index_name, namespace, batch_size=200):
    pc = Pinecone(os.environ.get("PINECONE_API_KEY"), pool_threads=30)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=len(embeddings[0]['embedding']), 
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws', 
                region='us-east-1'
            )
        ) 

    index = pc.Index(index_name)

    # Prepare the data to be upserted in chunks
    for chunk in chunks(embeddings, batch_size):
        data_to_upsert = [
            (
                str(uuid.uuid4()),
                item['embedding'], 
                # rest of the fields as metadata (no hardcoding)
                {key: item[key] for key in item if key != 'embedding'}
            ) 
            for item in chunk
        ]
        
        # Upsert data into Pinecone asynchronously to handle large batches
        async_result = index.upsert(vectors=data_to_upsert, namespace=namespace, async_req=True)
        
        # Wait for the async request to complete
        async_result.result()  # Use `.result()` instead of `.get()`

    print(f"Embeddings stored successfully in batches of {batch_size} in Pinecone.")

@traceable
def query_vectors(vector, api_key=PINECONE_API_KEY, index_endpoint=None, namespace=None, top_k=20, filter=None, run_id=""):
    url = f"https://{index_endpoint}/query"
    headers = {
        "Api-Key": api_key,
        "Content-Type": "application/json"
    }

    if filter:
        filter = {k: v for k, v in filter.items() if v is not None}
        
    data = {
        "namespace": namespace,
        "vector": vector,
        "topK": top_k,
        "includeMetadata": True,
        "filter": filter 
    }

    response = httpx.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()

@traceable
def get_pinecone_vdb_results(embedded_query, index_name, name_space, k=10, filter=None):
    index_endpoint = get_index_endpoint(index_name=index_name)
    
    response = query_vectors(embedded_query, index_endpoint=index_endpoint, namespace=name_space, top_k=k, filter=filter)

    passages = [
        {
            'passage_id': int(result['metadata']['passage_id']),
            'hebrew_text': result['metadata']['hebrew_text'],
            'english_text': result['metadata']['english_text'],
            'translation_id': result['metadata']['translation_id'],
            'book_name': result['metadata']['book_name'],
            'page_number': result['metadata']['page_number'],
            'text_to_embed': result['metadata']['text_to_embed'] if 'text_to_embed' in result['metadata'] else None
        }
        for result in response['matches']
    ]

    # Filter out passages that have English text which includes "sample translation"
    passages = [passage for passage in passages if "sample translation" not in passage['english_text'].lower()]
    return passages

@traceable
def get_context_from_pinecone_vdb(queries, index_name, namespace, k=10, print_output=PRINT_OUTPUT):
    contexts = []

    filter = queries["filter"] if "filter" in queries else None

    for key in queries:
        if key.startswith("query"):
            embedded_query = embed_text_openai(queries[key])
            context = get_pinecone_vdb_results(embedded_query, index_name, namespace, k, filter=filter)
            contexts.extend(context)

    # Remove duplicates
    contexts = [dict(t) for t in {tuple(d.items()) for d in contexts}]
    
    if print_output:
        print("Number of contexts: ", len(contexts))
    return contexts