import os
import weaviate
from weaviate.classes.init import Auth
from talmud_query.embed_utils import embed_text_openai

# Best practice: store your credentials in environment variables
weaviate_url = "https://1exconarfknip3xtkwcg.c0.us-east1.gcp.weaviate.cloud"
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
print(weaviate_api_key)

# Connect to Weaviate Cloud
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key),
)

print(client.is_ready())