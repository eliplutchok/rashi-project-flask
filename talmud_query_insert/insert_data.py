import os
import os
print(os.getcwd())
os.chdir('/Users/eliplutchok/Documents/Rashi-Project/backend-py')
print(os.getcwd())

from pinecone import ServerlessSpec
from db_utils import fetch_passages, fetch_english_passages, fetch_sentence_passages, fetch_bolded_words_passages
from embed_utils import generate_embeddings, add_openai_embeddings_to_passages, embed_text_openai
from pinecone_insert_utils import store_pinecone_embeddings_in_batches