from langchain.embeddings.openai import OpenAIEmbeddings
from langsmith import traceable
from talmud_query.config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_EMBEDDING_MODEL

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