
import httpx
from langsmith import traceable, Client
import os
import itertools
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
def get_pinecone_vdb_results(embedded_query, index_endpoint, name_space, k=10, filter=None):
    
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

    index_endpoint = get_index_endpoint(api_key=PINECONE_API_KEY, index_name=index_name)

    for key in queries:
        if key.startswith("query"):
            embedded_query = embed_text_openai(queries[key])
            context = get_pinecone_vdb_results(embedded_query, index_endpoint, namespace, k, filter=filter)
            contexts.extend(context)

    # Remove duplicates
    contexts = [dict(t) for t in {tuple(d.items()) for d in contexts}]
    
    if print_output:
        print("Number of contexts: ", len(contexts))
    return contexts

async def get_context_async(query, index_name, namespace, k, print_output):
    return get_context_from_pinecone_vdb(query, index_name, namespace, k, print_output=print_output)

@traceable
def get_context_from_pinecone_vdb_v2(embedded_queries, filter, index_name, namespace, k=10, print_output=PRINT_OUTPUT):
    contexts = []

    index_endpoint = get_index_endpoint(api_key=PINECONE_API_KEY, index_name=index_name)

    for query in embedded_queries:
        context = get_pinecone_vdb_results(query, index_endpoint, namespace, k, filter=filter)
        contexts.extend(context)

    # Remove duplicates
    contexts = [dict(t) for t in {tuple(d.items()) for d in contexts}]
    
    if print_output:
        print("Number of contexts: ", len(contexts))
    return contexts
