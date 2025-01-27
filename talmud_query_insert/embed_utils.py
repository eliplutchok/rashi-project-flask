from langchain.embeddings.openai import OpenAIEmbeddings
from langsmith import traceable
import os
from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_EMBEDDING_MODEL
from openai import OpenAI
import requests

client = OpenAI(
  api_key=os.environ.get('NVIDIA_API_KEY'),
  base_url="https://integrate.api.nvidia.com/v1"
)

def embed_text_openai(text, model_name=OPENAI_EMBEDDING_MODEL):
    embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)
    return embed.embed_documents([text])[0]


@traceable
def embed_text_openai_batch(texts, model_name=OPENAI_EMBEDDING_MODEL):
    embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)
    return embed.embed_documents(texts)

def add_openai_embeddings_to_passages(passages, model_name=OPENAI_EMBEDDING_MODEL):
    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=OPENAI_API_KEY
    )
    english_passages = [(passage['passage_id'], passage["text_to_embed"]) for passage in passages]
    texts = [text for _, text in english_passages]
    embeddings = embed.embed_documents(texts)
    for passage, embedding in zip(passages, embeddings):
        passage["embedding"] = embedding
    return passages

def add_embeddings_to_passages(passages, model):
    english_passages = [(passage['passage_id'], passage["text_to_embed"]) for passage in passages]
    texts = [text for _, text in english_passages]
    embeddings = model.encode(texts, show_progress_bar=True)
    for passage, embedding in zip(passages, embeddings):
        passage["embedding"] = embedding
    return passages

def generate_embeddings(passages, text_field, model):
    english_passages = [(passage['passage_id'], passage[text_field]) for passage in passages]
    texts = [text for _, text in english_passages]
    embeddings = model.encode(texts, show_progress_bar=True)
    return [(passage_id, embedding) for (passage_id, _), embedding in zip(english_passages, embeddings)]

# function to get nvdia embeddings
def get_nvidia_embeddings(texts):
    response = client.embeddings.create(
            input=texts,
            model="nvidia/nv-embed-v1",
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "NONE"}
        )
    
    print(response)
    # embeddings_list = []
    # response_data = response.data
    # for i in range(len(response_data)):
    #     if response_data[i].text:
    #         embeddings_list.append(list(response_data[i])[0][1])
    # return embeddings_list

def add_nvidia_embeddings_to_passages(passages):
    texts = [passage["text_to_embed"] for passage in passages]
    embeddings = get_nvidia_embeddings(texts)
    
    for passage, embedding in zip(passages, embeddings):
        passage["embedding"] = embedding
    
    return passages

def get_hf_embeddings(texts, model_name):
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {os.environ.get('HF_ACCESS_TOKEN')}"}

    response = requests.post(API_URL, headers=headers, json={"inputs": texts})
    return response.json()

def add_hf_embeddings_to_passages(passages, model_name):
    texts = [passage["text_to_embed"] for passage in passages]
    embeddings = get_hf_embeddings(texts, model_name)
    for passage, embedding in zip(passages, embeddings):
        passage["embedding"] = embedding
    return passages