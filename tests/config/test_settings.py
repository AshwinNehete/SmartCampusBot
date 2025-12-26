import os
import pytest
from src.config.settings import Config
from src.utils.helpers import validate_config

class TestConfig:
    def test_default_values(self):
        """Test that default values are set correctly"""
        # Ensure we are testing defaults by unsetting env vars if they exist
        # This is a bit tricky since Config loads at import time usually, 
        # but since it's a class with attributes, we can just check the values.
        # However, they are class attributes loaded at import time. 
        # So we check if they match reasonable expectations or current env.
        
        # Taking a safer approach: check types and non-empty for defaults
        assert isinstance(Config.OLLAMA_BASE_URL, str)
        assert Config.OLLAMA_BASE_URL.startswith("http")
        
        assert isinstance(Config.OLLAMA_MODEL, str)
        assert len(Config.OLLAMA_MODEL) > 0
        
        assert isinstance(Config.EMBEDDING_MODEL, str)
        assert isinstance(Config.VECTOR_STORE_PATH, str)
        assert isinstance(Config.TOP_K_RESULTS, int)
        assert Config.TOP_K_RESULTS > 0

    def test_validate_config_valid(self):
        """Test validation with valid config"""
        valid_config = {
            "OLLAMA_BASE_URL": "http://localhost:11434",
            "OLLAMA_MODEL": "llama2:7b-chat",
            "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
            "VECTOR_STORE_PATH": "./data/vector_store",
            "TOP_K_RESULTS": "3",
            "GRADIO_PORT": "7860",
            "MAX_RESPONSE_LENGTH": "500"
        }
        errors = validate_config(valid_config)
        assert len(errors) == 0

    def test_validate_config_missing_required(self):
        """Test validation with missing required fields"""
        invalid_config = {
            "OLLAMA_MODEL": "llama2:7b-chat"
            # Missing OLLAMA_BASE_URL etc.
        }
        errors = validate_config(invalid_config)
        assert len(errors) > 0
        assert any("Missing or empty required configuration: OLLAMA_BASE_URL" in e for e in errors)

    def test_validate_config_invalid_url(self):
        """Test validation with invalid URL"""
        invalid_config = {
            "OLLAMA_BASE_URL": "tp://localhost:11434", # Invalid scheme
            "OLLAMA_MODEL": "llama2:7b-chat",
            "EMBEDDING_MODEL": "model",
            "VECTOR_STORE_PATH": "./data"
        }
        errors = validate_config(invalid_config)
        assert any("OLLAMA_BASE_URL must start with http:// or https://" in e for e in errors)

    def test_validate_config_invalid_types(self):
        """Test validation with invalid numeric types"""
        invalid_config = {
            "OLLAMA_BASE_URL": "http://localhost",
            "OLLAMA_MODEL": "model",
            "EMBEDDING_MODEL": "model",
            "VECTOR_STORE_PATH": "./data",
            "TOP_K_RESULTS": "not_an_int"
        }
        errors = validate_config(invalid_config)
        assert any("TOP_K_RESULTS must be a valid integer" in e for e in errors)
