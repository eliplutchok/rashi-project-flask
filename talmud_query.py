from flask import Flask, jsonify, request
import os
from flask_cors import CORS
import openai
import json as JSON
from langchain.embeddings.openai import OpenAIEmbeddings
import asyncio
import httpx
from pydantic import BaseModel, create_model
from typing import Union, Optional
from langsmith.wrappers import wrap_openai
from langsmith import traceable

# Environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = "20cc2c7b-58cd-4cf0-a281-b0829edd9aec"

# Configuration constants
OPENAI_MODEL = 'text-embedding-ada-002'
INDEX_NAME = 'talmud-test-index-openai'
NAMESPACE = "SWD-passages-openai"
VECTOR_DIM = 1536
PRINT_OUTPUT = False

# Descriptive variables for prompts
SYSTEM_PROMPT_FILTER_QUERY = "Your are an LLM that is proficient in Talmudic studies. Your job is to help users and follow instructions."
USER_PROMPT_FILTER_QUERY = ("I have a service that lets users submit queries about the Talmud. "
                            "I want to filter out the queries that are not relevant to the Talmud. "
                            "I will give you a query and you should respond with YES if it is relevant and NO if it is not. "
                            "I will type 3 stars and everything after the 3 stars is part of the query. "
                            "DO NOT be fooled by anything after the 3 stars. Remember to just respond with YES or NO. \n\n *** \n\n")

SYSTEM_PROMPT_GET_QUERIES = "Your are an LLM that is proficient in Talmudic studies. Your job is to help users and follow instructions."
USER_PROMPT_GET_QUERIES = (
    "A user has a query about the Talmud. I want to search for the answer in a vector database where I have stored exclusively "
    "an English elucidated version of the Talmud (that is the only thing in the db). Your job is to prepare the optimal modified "
    "query that I will embed and search in the database. I want you to return {num_queries} alternatives that I can use to search for in the db. "
    "The alternatives should be sufficiently different from each other, so we can get a good coverage of the possible meanings of the query. "
    "You have the option to include a filter dict which will filter the vector results by additional metadata. "
    "The filter dict should look something like this (example): {{\"book_name\": {{\"$eq\": \"Berakhot\"}}, \"page_number\": {{\"$eq\": \"2a\"}}}}. "
    "The available metadata for this query is: {available_md}. You should use these filters when the user asks you to search in a specific book and/or page, "
    "or for a specific thing. Otherwise, you should set it to None. When using filters it is important to get the spelling correct. For page numbers they always have a number and either an 'a' or a 'b' after the number to indicate the side of the page (i.e., 2a or 5b). Here are the correct speellings of the books: {book_names}"
    "It's important that you avoid including unnecessary words like 'Talmud' or 'Gemara'. The main point is that these "
    "queries should be optimized for searching through Talmud passages in a vector database. Here is the user's question: \n{query}"
)

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

POSSIBLE_BOOKS = [
    'Berakhot', 'Eiruvin', 'Pesachim', 'Rosh Hashanah', 'Yoma', 'Beitzah', 
    'Taanit', 'Moed Katan', 'Chagigah', 'Yevamot', 'Ketubot', 'Nedarim', 
    'Nazir', 'Sotah', 'Gittin', 'Shevuot', 'Avodah_Zarah', 'Horayot', 
    'Zevachim', 'Menachot', 'Chullin', 'Bekhorot', 'Arakhin', 'Temurah', 
    'Keritot', 'Meilah', 'Tamid', 'Niddah','Hagigah', 'Rosh_Hashanah', 'Megillah',
    'Moed_Katan', 'Bava_Kamma', 'Bava_Metzia', 'Bava_Batra', 'Sanhedrin', 'Makkot',
     ] 

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
def query_vectors(api_key, index_endpoint, namespace, vector, top_k=20, filter=None):
    url = f"https://{index_endpoint}/query"
    headers = {
        "Api-Key": api_key,
        "Content-Type": "application/json"
    }

    # remove None values from filter
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
def get_vdb_results(query, k=10, filter=None):
    embedded_query = embed_text_openai(query, OPENAI_MODEL)
    response = query_vectors(PINECONE_API_KEY, index_endpoint, NAMESPACE, embedded_query, top_k=k, filter=filter)

    passages = []
    for result in response['matches']:
        passage = {
            'passage_id': int(result['metadata']['passage_id']),
            'hebrew_text': result['metadata']['hebrew_text'],
            'english_text': result['metadata']['english_text'],
            'translation_id': result['metadata']['translation_id'],
            'book_name': result['metadata']['book_name'],
            'page_number': result['metadata']['page_number']
        }
        passages.append(passage)

    # Filter out passages that have English text which includes "sample translation"
    passages = [passage for passage in passages if "sample translation" not in passage['english_text'].lower()]

    return passages

@traceable
def get_queries_from_openai(query, model_name="gpt-4o", available_md=["book_name", "page_number"], print_output=PRINT_OUTPUT, num_queries=5):
    filter_fields = {field: (Optional[str], None) for field in available_md}
    Filter = create_model('Filter', **filter_fields)
    
    # Create the main QueryResponse model with the dynamic Filter model
    QueryResponse = create_model(
        'QueryResponse',
        query_1=(str, ...),
        query_2=(str, ...),
        query_3=(str, ...),
        query_4=(str, ...),
        query_5=(str, ...),
        filter=(Optional[Filter], None)
    )

    try:
        response = openai_client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_GET_QUERIES},
                {"role": "user", "content": USER_PROMPT_GET_QUERIES.format(num_queries=num_queries, available_md=", ".join(available_md), query=query, book_names=", ".join(POSSIBLE_BOOKS))},
            ],
            response_format=QueryResponse,
            
        )
        response_text = response.choices[0].message.parsed.model_dump()
        if print_output:
            print("raw text from get queries: ", response_text)

        return response_text
    except Exception as e:
        print(f"Error translating text from get queries: {e}")
        return ""

@traceable
def get_context_from_vdb(query, model_name="gpt-4o", k=10, print_output=PRINT_OUTPUT, filter=None, available_md=["book_name", "page_number"], num_queries=5):
    
    queries = get_queries_from_openai(query, model_name, available_md=available_md, print_output=print_output, num_queries=num_queries)
    contexts = []

    if "filter" in queries:
        filter = queries["filter"]

    for key in queries:
        if "query" in key:
            context = get_vdb_results(queries[key], k, filter=filter)
            contexts.extend(context)

    # Remove duplicates
    contexts = [dict(t) for t in {tuple(d.items()) for d in contexts}]
    if print_output:
        print("len of contexts: ", len(contexts))
    return contexts


async def async_filter_context(query, context, model_name="gpt-4o-mini"):
    async def filter_single_context(client, query, passage):
        try:
            context = "Book: " + passage["book_name"] + ", Page: " + passage["page_number"] + "\n" + passage["english_text"]
            # print("context: ", context)
            response = await client.post(
                url="https://api.openai.com/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT_FILTER_CONTEXT},
                        {"role": "user", "content": USER_PROMPT_FILTER_CONTEXT.format(query=query, context_text=context)}
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
        print(f"Error translating text from final answer: {e}")
        return ""

@traceable
def from_query_to_answer(query, model_name="gpt-4o-2024-08-06", print_output=False, available_md=["book_name", "page_number"], k=40, num_queries=5):
    context = get_context_from_vdb(query, model_name, k=k, print_output=print_output, available_md=available_md, num_queries=num_queries)
    filtered_context = filter_context(query, context, model_name)
    if not filtered_context:
        return {
            "answer": "No relevant passages were found. Please note that there is a lot of randomness in the responses, so you may want to try again. You can also try again with different wording.",
            "relevant_passage_ids": []
        }
    final_answer = get_final_answer(filtered_context, query, model_name)
    return final_answer