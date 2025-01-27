import os
from deep_prompts import (
    CHUNK_DESC_ENG_PASSAGES,
    CHUNK_DESC_HEB_PASSAGES,
    CHUNK_DESC_ELUCIDATED_ENG_PASSAGES,
    CHUNK_DESC_SEN_ENG_PASSAGES,
    CHUNK_DESC_ENG_PAGE,
    CHUNK_DESC_HEB_PAGE,
    CHUNK_DESC_ELUCIDATED_ENG_PAGE,
    CHUNK_DESC_ENG_SEV_PASSAGES,
    CHUNK_DESC_HEB_SEV_PASSAGES,
    CHUNK_DESC_ELUCIDATED_ENG_SEV_PASSAGES
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Configuration constants
OPENAI_MODEL = 'text-embedding-ada-002'
OPENAI_EMBEDDING_MODEL = 'text-embedding-ada-002'
INDEX_NAME = 'talmud-test-index-openai'
NAMESPACE = "SWD-passages-openai"
VECTOR_DIM = 1536
PRINT_OUTPUT = False

POSSIBLE_BOOKS = [
    'Berakhot', 'Eiruvin', 'Pesachim', 'Rosh Hashanah', 'Yoma', 'Beitzah', 
    'Taanit', 'Moed Katan', 'Chagigah', 'Yevamot', 'Ketubot', 'Nedarim', 
    'Nazir', 'Sotah', 'Gittin', 'Shevuot', 'Avodah_Zarah', 'Horayot', 
    'Zevachim', 'Menachot', 'Chullin', 'Bekhorot', 'Arakhin', 'Temurah', 
    'Keritot', 'Meilah', 'Niddah', 'Hagigah', 'Rosh_Hashanah', 'Megillah',
    'Moed_Katan', 'Bava_Kamma', 'Bava_Metzia', 'Bava_Batra', 'Sanhedrin', 'Makkot',
]

DEEP_CHUNK_DESCS = [
    CHUNK_DESC_ENG_PASSAGES,
    # CHUNK_DESC_HEB_PASSAGES,
    CHUNK_DESC_ELUCIDATED_ENG_PASSAGES,
    # CHUNK_DESC_SEN_ENG_PASSAGES,
    # CHUNK_DESC_ENG_PAGE,
    # CHUNK_DESC_HEB_PAGE,
    # CHUNK_DESC_ELUCIDATED_ENG_PAGE,
    # CHUNK_DESC_ENG_SEV_PASSAGES,
    # CHUNK_DESC_HEB_SEV_PASSAGES,
    # CHUNK_DESC_ELUCIDATED_ENG_SEV_PASSAGES
]

DB_CONFIGS = [
    {
        "index_name": "talmud-test-index-openai",
        "namespace": "SWD-passages-openai",
        "chunk_desc": CHUNK_DESC_ENG_PASSAGES,
    },
    {
        "index_name": "talmud-test-index-openai",
        "namespace": "SWD-passages-openai",
        "chunk_desc": CHUNK_DESC_ELUCIDATED_ENG_PASSAGES,
    }   
]
