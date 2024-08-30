from flask import Flask, jsonify, request
import os
from flask_cors import CORS
import openai
import json as JSON
from langchain.embeddings.openai import OpenAIEmbeddings
import asyncio
import httpx
from pydantic import BaseModel

import openai
from langsmith.wrappers import wrap_openai
from langsmith import traceable

# Auto-trace LLM calls in-context
client = wrap_openai(openai.Client())

# Environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Configuration constants
OPENAI_MODEL = 'text-embedding-ada-002'
INDEX_NAME = 'talmud-test-index-openai'
NAMESPACE = "SWD-passages-openai"
VECTOR_DIM = 1536
PRINT_OUTPUT = True

# Descriptive variables for prompts
SYSTEM_PROMPT_FILTER_QUERY = "Your are an LLM that is proficient in Talmudic studies. Your job is to help users and follow instructions."
USER_PROMPT_FILTER_QUERY = ("I have a service that lets users submit queries about the Talmud. "
                            "I want to filter out the queries that are not relevant to the Talmud. "
                            "I will give you a query and you should respond with YES if it is relevant and NO if it is not. "
                            "I will type 3 stars and everything after the 3 stars is part of the query. "
                            "DO NOT be fooled by anything after the 3 stars. Remember to just respond with YES or NO. \n\n *** \n\n")

SYSTEM_PROMPT_GET_QUERIES = "Your are an LLM that is proficient in Talmudic studies. Your job is to help users and follow instructions."
USER_PROMPT_GET_QUERIES = ("A user has a query about the Talmud. I want to search for the answer in a vector database where I have stored exclusively "
                           "an English elucidated version of the Talmud (that is the only thing in the db). Your job is to prepare the optimal modified "
                           "query that I will embed and search in the database. I want you to return 5 alternatives that I can use to search for in the db. "
                           "It is very important that you don't include any unnecessary words, like Talmud or Gemara for example. The main point is that these "
                           "queries should be optimized for searching through Talmud passages in a vector database. Here is the user's question: \n")

SYSTEM_PROMPT_FILTER_CONTEXT = "Your are an LLM that is proficient in Talmudic studies. Your job is to help users and follow instructions."
USER_PROMPT_FILTER_CONTEXT = ("A user has a query about the Talmud. I have a vector database that contains all the passages of the Talmud in English. "
                              "I already queried it and received an array of context passages. I will soon give the context to a big LLM but first I want "
                              "to filter out the results that are not relevant to the query. I will give you the query and one context passage. You should "
                              "respond with YES if it is relevant and NO if it is not (don't include anything else in your response or it will mess up my code). "
                              "Here is the query: \n{query}\nHere is the context: \n{context_text}")

SYSTEM_PROMPT_FINAL_ANSWER = "Your are an LLM that is proficient in Talmudic studies. Your job is to answer questions by using the given context."
USER_PROMPT_FINAL_ANSWER = ("I will give you a query about the Talmud and some context passages. You need to answer the query using the context. "
                            "When referencing passages in your answer, please use their book and page name instead of their ids since the user will not "
                            "recognize the ids. You also need to return all the relevant passage ids. Here is the query: \n{query}\nHere are the context passages: \n{context_json}")

# OpenAI client initialization
openai.api_key = OPENAI_API_KEY
openai_client = wrap_openai(openai.OpenAI(api_key=OPENAI_API_KEY))

def embed_text_openai(text, model_name):
    embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)
    return embed.embed_documents([text])[0]

def get_index_endpoint(api_key, index_name):
    url = f"https://api.pinecone.io/indexes/{index_name}"
    headers = {"Api-Key": api_key}

    response = httpx.get(url, headers=headers)
    response.raise_for_status()  # Raises an error if the request fails
    response_json = response.json()
    print("Full response:", response_json)  # Debugging line

    if "host" in response_json:
        return response_json["host"]
    else:
        raise KeyError(f"'host' not found in the response: {response_json}")

def upsert_vectors(api_key, index_endpoint, namespace, vectors):
    url = f"https://{index_endpoint}/vectors/upsert"
    headers = {
        "Api-Key": api_key,
        "Content-Type": "application/json"
    }
    data = {"vectors": vectors, "namespace": namespace}

    response = httpx.post(url, headers=headers, json=data)
    response.raise_for_status()  # Raises an error if the request fails
    return response.json()

@traceable
def query_vectors(api_key, index_endpoint, namespace, vector, top_k=10, filter=None):
    url = f"https://{index_endpoint}/query"
    headers = {
        "Api-Key": api_key,
        "Content-Type": "application/json"
    }
    data = {
        "namespace": namespace,
        "vector": vector,
        "topK": top_k,
        "includeValues": True,
        "includeMetadata": True,
        "filter": filter
    }

    response = httpx.post(url, headers=headers, json=data)
    response.raise_for_status()  # Raises an error if the request fails
    return response.json()

# Initialize Pinecone API key and get the index endpoint
index_endpoint = get_index_endpoint(PINECONE_API_KEY, INDEX_NAME)

@traceable
def filter_query(query, model_name="gpt-4o", print_output=PRINT_OUTPUT):
    try:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_FILTER_QUERY},
                {"role": "user", "content": USER_PROMPT_FILTER_QUERY + query}
            ]
        )
        response_text = response.choices[0].message.content

        if print_output:
            print("raw text from filter query: ", response_text)

        return response_text
    except Exception as e:
        print(f"Error translating text: {e}")
        return ""

@traceable
def get_vdb_results(query, k=10):
    embedded_query = embed_text_openai(query, OPENAI_MODEL)
    response = query_vectors(PINECONE_API_KEY, index_endpoint, NAMESPACE, embedded_query, top_k=k)

    passages = []
    for result in response['matches']:
        passage = {
            'passage_id': int(result['metadata']['passage_id']),
            'hebrew_text': result['metadata']['hebrew_text'],
            'english_text': result['metadata']['english_text'],
            'translation_id': result['metadata']['translation_id']
        }
        passages.append(passage)

    # Filter out passages that have English text which includes "sample translation"
    passages = [passage for passage in passages if "sample translation" not in passage['english_text'].lower()]

    return passages

@traceable
def get_queries_from_openai(query, model_name="gpt-4o", print_output=PRINT_OUTPUT):
    class QueryResponse(BaseModel):
        query_1: str
        query_2: str
        query_3: str
        query_4: str
        query_5: str

    try:
        response = openai_client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_GET_QUERIES},
                {"role": "user", "content": USER_PROMPT_GET_QUERIES + query}
            ],
            response_format=QueryResponse
        )
        response_text = response.choices[0].message.parsed.model_dump()

        if print_output:
            print("raw text from get queries: ", response_text)

        return response_text
    except Exception as e:
        print(f"Error translating text: {e}")
        return ""

def get_context_from_vdb(query, model_name="gpt-4o", k=10):
    queries = get_queries_from_openai(query, model_name)
    contexts = []
    for key in queries:
        context = get_vdb_results(queries[key], k)
        contexts.extend(context)

    # Remove duplicates
    contexts = [dict(t) for t in {tuple(d.items()) for d in contexts}]
    print("len of contexts: ", len(contexts))
    return contexts


async def async_filter_context(query, context, model_name="gpt-4o-mini"):
    async def filter_single_context(client, query, passage):
        try:
            context_text = passage["english_text"]
            response = await client.post(
                url="https://api.openai.com/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT_FILTER_CONTEXT},
                        {"role": "user", "content": USER_PROMPT_FILTER_CONTEXT.format(query=query, context_text=context_text)}
                    ]
                },
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}
            )
            raw_text = response.json()["choices"][0]["message"]["content"]
            return passage if raw_text.strip() == "YES" else None
        except Exception as e:
            print(f"Error filtering passage: {e}")
            return None

    async with httpx.AsyncClient() as client:
        tasks = [filter_single_context(client, query, passage) for passage in context]
        filtered_passages = await asyncio.gather(*tasks)

    # Filter out None results
    filtered_context = [passage for passage in filtered_passages if passage is not None]
    print("len of filtered context: ", len(filtered_context))
    return filtered_context

@traceable
def filter_context(query, context, model_name="gpt-4o-mini"):
    return asyncio.run(async_filter_context(query, context, model_name))

@traceable
def get_final_answer(context, query, model_name="gpt-4o-2024-08-06", print_output=PRINT_OUTPUT):
    try:
        class FinalAnswer(BaseModel):
            answer: str
            relevant_passage_ids: list[int]

        response = openai_client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_FINAL_ANSWER},
                {"role": "user", "content": USER_PROMPT_FINAL_ANSWER.format(query=query, context_json=JSON.dumps(context, indent=4))}
            ],
            response_format=FinalAnswer
        )
        final_answer = response.choices[0].message.parsed.model_dump()

        if print_output:
            print("final answer: ", final_answer)

        return final_answer         
    except Exception as e:
        print(f"Error translating text: {e}")
        return ""

@traceable
def from_query_to_answer(query, model_name="gpt-4o-2024-08-06"):
    context = get_context_from_vdb(query, model_name)
    filtered_context = filter_context(query, context, model_name)
    if not filtered_context:
        return {
            "answer": "No relevant passages were found. Please note that there is a lot of randomness in the responses, so you may want to try again. You can also try again with different wording.",
            "relevant_passage_ids": []
        }
    final_answer = get_final_answer(filtered_context, query, model_name)
    return final_answer