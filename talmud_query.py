import openai
import json as JSON
import asyncio
import httpx
from pydantic import BaseModel, create_model
from typing import Union, Optional
from langsmith.wrappers import wrap_openai
from langsmith.run_helpers import get_current_run_tree
from langsmith import traceable, Client
from prompts import (
    SYSTEM_PROMPT_FILTER_QUERY,
    USER_PROMPT_FILTER_QUERY,
    SYSTEM_PROMPT_GET_QUERIES,
    USER_PROMPT_GET_QUERIES,
    SYSTEM_PROMPT_FILTER_CONTEXT,
    USER_PROMPT_FILTER_CONTEXT,
    SYSTEM_PROMPT_FINAL_ANSWER,
    USER_PROMPT_FINAL_ANSWER,
)
from config import OPENAI_API_KEY, PRINT_OUTPUT, POSSIBLE_BOOKS
from pinecone_utils import get_context_from_pinecone_vdb, get_context_async, get_context_from_pinecone_vdb_v2
from embed_utils import embed_text_openai_batch

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
async def async_filter_context(query, context, model_name="gpt-4o-mini", text_field='english_text'):
    async def filter_single_context(client, query, passage):
        try:
            context_text = f"Book: {passage['book_name']}, Page: {passage['page_number']}\n{passage[text_field]}"
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

    return [passage for passage in filtered_passages if passage is not None]

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
def talmud_query_v1(
    query, 
    model_name="gpt-4o-2024-08-06", 
    print_output=False, 
    available_md=["book_name", "page_number"], 
    k=40, 
    num_alt_queries=5
):
    index_name = "talmud-test-index-openai"
    namespace = "SWD-passages-openai"
    openai.api_key = OPENAI_API_KEY
    openai_client = wrap_openai(openai.OpenAI(api_key=OPENAI_API_KEY))
    
    run = get_current_run_tree()

    query_alts = get_queries_from_openai(query, model_name, available_md=available_md, print_output=print_output, num_queries=num_alt_queries, openai_client=openai_client)
    context = get_context_from_pinecone_vdb(query_alts, index_name, namespace, k, print_output=print_output)
    filtered_context = filter_context(query, context, model_name)
    
    if not filtered_context:
        return [{
            "answer": "No relevant passages were found. Please note that there is a lot of randomness in the responses, so you may want to try again. You can also try again with different wording.",
            "relevant_passage_ids": []
        }, run.id]
    
    final_answer = get_final_answer(query, filtered_context, model_name, print_output=print_output, openai_client=openai_client)

    return [final_answer, run.id]

@traceable
def talmud_query_v2(
    query, 
    model_name="gpt-4o-2024-08-06", 
    print_output=False, 
    available_md=["book_name", "page_number"], 
    k=30, 
    num_alt_queries=4
):
    index_name = "talmud-test-index-openai"
    namespaces = [
        "SWD-passages-openai", 
        "SWD-passages-openai-bold"
    ]
    
    openai.api_key = OPENAI_API_KEY
    openai_client = wrap_openai(openai.OpenAI(api_key=OPENAI_API_KEY))
    
    run = get_current_run_tree()

    query_alts = get_queries_from_openai(query, model_name, available_md=available_md, print_output=print_output, num_queries=num_alt_queries, openai_client=openai_client)   
    filter = query_alts.get("filter")
   
    # embedded_query_list = [embed_text_openai(query_alts[key]) for key in query_alts if key.startswith("query")]
    embedded_query_list = embed_text_openai_batch([query_alts[key] for key in query_alts if key.startswith("query")])
    
    contexts_list = []
    for namespace in namespaces:
        contexts_list.append(get_context_from_pinecone_vdb_v2(embedded_query_list, filter, index_name, namespace, k, print_output))
    
    context = [item for sublist in contexts_list for item in sublist]
    
    # Remove duplicate passages by passage_id
    seen_ids = set()
    context = [passage for passage in context if not (passage['passage_id'] in seen_ids or seen_ids.add(passage['passage_id']))]

    print(f"Number of unique passages: {len(context)}")
    # Filter context asynchronously
    filtered_context = filter_context(query, context, model_name)
    print(f"Number of filtered passages: {len(filtered_context)}")
    
    if not filtered_context:
        return [{
            "answer": "No relevant passages were found. Please note that there is a lot of randomness in the responses, so you may want to try again. You can also try again with different wording.",
            "relevant_passage_ids": []
        }, run.id]
    
    final_answer = get_final_answer(query, filtered_context, model_name, print_output=print_output, openai_client=openai_client)

    return [final_answer, run.id]