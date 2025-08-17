import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration settings for SmartCampusBot"""

    # Ollama Configuration
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2:7b-chat")

    # Embedding Model
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector Store Configuration
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))

    # Gradio Configuration
    GRADIO_SHARE = os.getenv("GRADIO_SHARE", "False").lower() == "true"
    GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))

    # Application Settings
    MAX_RESPONSE_LENGTH = int(os.getenv("MAX_RESPONSE_LENGTH", "500"))
    RESPONSE_TIMEOUT = int(os.getenv("RESPONSE_TIMEOUT", "30"))

    # Data paths
    DATA_DIR = "./data"
    FAQ_FILE = os.path.join(DATA_DIR, "university_faq.json")
    INTENTS_FILE = os.path.join(DATA_DIR, "university_intents.json")
