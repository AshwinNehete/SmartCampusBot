import pytest
from unittest.mock import MagicMock, patch
from src.generator.ollama_client import OllamaClient

class TestOllamaClient:
    @pytest.fixture
    def mock_ollama(self):
        with patch("src.generator.ollama_client.ollama.Client") as mock:
            yield mock

    def test_init(self, mock_ollama):
        # Mocking list() to return empty so init succeeds
        mock_instance = mock_ollama.return_value
        mock_instance.list.return_value = {"models": []}
        
        client = OllamaClient()
        mock_ollama.assert_called_once()
        assert client.client is not None

    def test_init_raises(self, mock_ollama):
        mock_ollama.side_effect = Exception("Connection error")
        
        with pytest.raises(Exception):
            OllamaClient()

    def test_generate_response_success(self, mock_ollama):
        mock_instance = mock_ollama.return_value
        # Mock successful init
        mock_instance.list.return_value = {"models": []}
        
        # Mock chat response
        # Using the object structure for newer clients
        mock_response = MagicMock()
        mock_response.message.content = "Test response"
        mock_instance.chat.return_value = mock_response
        
        client = OllamaClient()
        response = client.generate_response("Test prompt")
        
        assert response == "Test response"
        mock_instance.chat.assert_called_once()

    def test_generate_response_dict_format(self, mock_ollama):
        mock_instance = mock_ollama.return_value
        mock_instance.list.return_value = {"models": []}
        
        # Mock chat response as dict (older clients)
        mock_instance.chat.return_value = {
            "message": {"content": "Dict response"}
        }
        
        client = OllamaClient()
        response = client.generate_response("Test prompt")
        
        assert response == "Dict response"

    def test_generate_response_error(self, mock_ollama):
        mock_instance = mock_ollama.return_value
        mock_instance.list.return_value = {"models": []}
        mock_instance.chat.side_effect = Exception("Chat error")
        
        client = OllamaClient()
        response = client.generate_response("Test prompt")
        
        assert "trouble generating a response" in response

    def test_is_model_available(self, mock_ollama):
        mock_instance = mock_ollama.return_value
        mock_instance.list.return_value = {
            "models": [{"name": "llama2:7b-chat"}, {"name": "other:latest"}]
        }
        
        client = OllamaClient()
        
        assert client.is_model_available("llama2:7b-chat") is True
        assert client.is_model_available("other") is True  # Should match normalized
        assert client.is_model_available("missing") is False

    def test_pull_model_success(self, mock_ollama):
        mock_instance = mock_ollama.return_value
        mock_instance.list.return_value = {"models": []}
        
        client = OllamaClient()
        success = client.pull_model("new-model")
        
        assert success is True
        mock_instance.pull.assert_called_with("new-model")

    def test_pull_model_failure(self, mock_ollama):
        mock_instance = mock_ollama.return_value
        mock_instance.list.return_value = {"models": []}
        mock_instance.pull.side_effect = Exception("Pull error")
        
        client = OllamaClient()
        success = client.pull_model("new-model")
        
        assert success is False
