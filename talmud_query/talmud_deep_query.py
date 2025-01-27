import openai
import json as JSON
import asyncio
import httpx
from pydantic import BaseModel, create_model
from typing import Union, Optional
from langsmith.wrappers import wrap_openai
from langsmith.run_helpers import get_current_run_tree
from langsmith import traceable, Client
from talmud_query.prompts import (
    SYSTEM_PROMPT_FILTER_QUERY,
    USER_PROMPT_FILTER_QUERY,
    SYSTEM_PROMPT_GET_QUERIES,
    USER_PROMPT_GET_QUERIES,
    SYSTEM_PROMPT_FILTER_CONTEXT,
    USER_PROMPT_FILTER_CONTEXT,
    SYSTEM_PROMPT_FINAL_ANSWER,
    USER_PROMPT_FINAL_ANSWER,
    SYSTEM_PROMPT_DEEP_QUERIES,
    USER_PROMPT_DEEP_QUERIES    
)
from talmud_query.config import OPENAI_API_KEY, PRINT_OUTPUT, POSSIBLE_BOOKS, DB_CONFIGS
from talmud_query.pinecone_utils import get_context_from_pinecone_vdb, get_context_async, get_context_from_pinecone_vdb_v2
from talmud_query.embed_utils import embed_text_openai_batch
from talmud_query.db_utils import get_nearby_passages
# load env variables
from dotenv import load_dotenv
import os

load_dotenv()

@traceable
def filter_query(query, model_name="gpt-4o", print_output=PRINT_OUTPUT, openai_client=None):
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
        print(f"Error filtering query: {e}")
        return ""

@traceable
def get_queries_from_openai(query, model_name="gpt-4o", available_md=[], print_output=PRINT_OUTPUT, num_queries=5, openai_client=None):
    filter_fields = {field: (Optional[str], None) for field in available_md}
    Filter = create_model('Filter', **filter_fields)  
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
                {"role": "user", "content": USER_PROMPT_GET_QUERIES.format(
                    num_queries=num_queries, available_md=", ".join(available_md), query=query, book_names=", ".join(POSSIBLE_BOOKS))}
            ],
            response_format=QueryResponse,
        )
        response_text = response.choices[0].message.parsed.model_dump()

        if print_output:
            print("raw text from get queries: ", response_text)

        return response_text
    except Exception as e:
        print(f"Error retrieving queries from OpenAI: {e}")
        return ""
    
@traceable
def get_deep_queries_from_openai(query, model_name="gpt-4o", available_md=[], num_queries=5, chunk_desc="", openai_client=None):
    QueryResponse = create_model(
        'QueryResponse',
        queries=(list[str], ...),
        filter=(Optional[dict], None)
    )

    try:
        response = openai_client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_DEEP_QUERIES},
                {"role": "user", "content": USER_PROMPT_DEEP_QUERIES.format(
                    chunk_desc=chunk_desc,
                    num_alt_queries=num_queries,
                    available_md=", ".join(available_md),
                    book_names=", ".join(POSSIBLE_BOOKS),   
                    query=query
                )}
            ],
            response_format=QueryResponse,
        )
        response_text = response.choices[0].message.parsed.model_dump()

        if PRINT_OUTPUT:
            print("Raw text from get deep queries: ", response_text)

        return response_text
    except Exception as e:
        print(f"Error retrieving deep queries from OpenAI: {e}")
        return ""

@traceable
async def async_filter_context(query, context, model_name="gpt-4o-mini", text_field='english_text'):
    async def filter_consecutive_passages(client, query, passages):
        try:
            context_text = "\n\n".join([f"Book: {p['book_name']}, Page: {p['page_number']}\n{p[text_field]}" for p in passages])
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
            return passages if raw_text.strip() == "YES" else None
        except Exception as e:
            print(f"Error filtering passages: {e}")
            return passages

    async with httpx.AsyncClient() as client:
        # Assuming context is now an array of arrays, where each inner array contains consecutive passages
        tasks = [filter_consecutive_passages(client, query, passage_group) for passage_group in context]
        filtered_passage_groups = await asyncio.gather(*tasks)

    return [passage for passage_group in filtered_passage_groups if passage_group is not None for passage in passage_group]

@traceable
def filter_context(query, context, model_name="gpt-4o-mini", text_field="english_text"):
    return asyncio.run(async_filter_context(query, context, model_name, text_field))

@traceable
def get_final_answer(query, context, model_name="gpt-4o-2024-08-06", print_output=PRINT_OUTPUT, run_id="", openai_client=None):
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
            response_format=FinalAnswer,
        )
        final_answer = response.choices[0].message.parsed.model_dump()

        if print_output:
            print("Final answer: ", final_answer)

        return final_answer
    except Exception as e:
        print(f"Error retrieving final answer: {e}")
        return ""

@traceable
def talmud_query_v2(
    query, 
    model_name="gpt-4o-2024-08-06", 
    print_output=False, 
    available_md=["book_name", "page_number"], 
    k=40, 
    db_configs=DB_CONFIGS,
    num_queries=5
):
    openai.api_key = OPENAI_API_KEY
    openai_client = wrap_openai(openai.OpenAI(api_key=OPENAI_API_KEY))

    query_alts = []
    for db_config in db_configs:
        query_alts.append(
            {
                "alt_queries": get_deep_queries_from_openai(query, model_name, available_md, num_queries, db_config["chunk_desc"], openai_client),
                "index_name": db_config["index_name"],
                "namespace": db_config["namespace"]
            }
        )
   
    embedded_query_list = []
    for query_alt in query_alts:
        embedded_queries = embed_text_openai_batch([alt_query for alt_query in query_alt["alt_queries"]["queries"]])
        embedded_query_list.append({
            "embedded_queries": embedded_queries,
            "index_name": query_alt["index_name"],
            "namespace": query_alt["namespace"],
            "filter": query_alt["alt_queries"]["filter"]
        })
    
    contexts_list = []
    for embedded_query in embedded_query_list:
        contexts_list.append(get_context_from_pinecone_vdb_v2(
            embedded_query["embedded_queries"],
            embedded_query["filter"],
            embedded_query["index_name"],
            embedded_query["namespace"],
            k,
            print_output
        ))
    
    context = [item for sublist in contexts_list for item in sublist]

    # add nearby passages for each passage in context
    for passage in context:
        context.extend(get_nearby_passages(passage["passage_id"], 10))
    
    # Remove duplicate passages by passage_id
    seen_ids = set()
    context = [passage for passage in context if not (passage['passage_id'] in seen_ids or seen_ids.add(passage['passage_id']))]

    # make sure context is order by book_name and passage_number
    context = sorted(context, key=lambda x: (x['book_name'], x['passage_number']))

    # prepare chunks for filtering - consecutive passages should be in the same chunk
    context_chunks_for_filterring = []
    current_chunk = []
    for passage in context:
        current_chunk.append(passage)
        if passage['book_name'] != context[0]['book_name'] or passage['passage_number'] - context[0]['passage_number'] > 2:
            context_chunks_for_filterring.append(current_chunk)
            current_chunk = []
    if current_chunk:
        context_chunks_for_filterring.append(current_chunk)

    print(f"Number of unique passages: {len(context)}")
    print(f"Number of chunks for filtering: {len(context_chunks_for_filterring)}")
    
    # Filter context asynchronously
    filtered_context = filter_context(query, context_chunks_for_filterring)
    print(f"Number of filtered passages: {len(filtered_context)}")


    run = get_current_run_tree()
    
    if not filtered_context:
        return [{
            "answer": "No relevant passages were found. Please note that there is a lot of randomness in the responses, so you may want to try again. You can also try again with different wording.",
            "relevant_passage_ids": []
        }, run.id]
    
    final_answer = get_final_answer(query, filtered_context, model_name, print_output=print_output, openai_client=openai_client)

    return [final_answer, run.id]