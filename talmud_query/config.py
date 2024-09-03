import os

OPENAI_API_KEY = "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"
PINECONE_API_KEY = "20cc2c7b-58cd-4cf0-a281-b0829edd9aec"

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