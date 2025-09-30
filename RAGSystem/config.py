import os
# URLs
RETRIEVAL_URL = "https://en.wikipedia.org/wiki/Boletus_edulis" # Where LLM retrieves information from

# Text splitter settings
CHUNK_SIZE = 1000 # characters
CHUNK_OVERLAP = 200 # characters

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LLM
LLM_NAME = "MistralAI"
LLM_MODEL = "open-mistral-7b"
LLM_PROVIDER = "mistralai"
ENV_VAR_NAME = "MISTRAL_API_KEY" # Environment name for system to find key
API_KEY = os.environ.get(ENV_VAR_NAME, "")

