import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.retriever.embeddings import EmbeddingModel

class TestEmbeddingModel:
    @pytest.fixture
    def mock_sentence_transformer(self):
        with patch("src.retriever.embeddings.SentenceTransformer") as mock:
            yield mock

    def test_init_loads_model(self, mock_sentence_transformer):
        model = EmbeddingModel("test-model")
        mock_sentence_transformer.assert_called_once_with("test-model")
        assert model.model is not None

    def test_encode_single_string(self, mock_sentence_transformer):
        mock_model_instance = mock_sentence_transformer.return_value
        mock_model_instance.encode.return_value = np.array([[0.1, 0.2]])
        
        model = EmbeddingModel()
        embeddings = model.encode("test text")
        
        mock_model_instance.encode.assert_called_with(["test text"], convert_to_numpy=True)
        assert isinstance(embeddings, np.ndarray)

    def test_encode_list_strings(self, mock_sentence_transformer):
        mock_model_instance = mock_sentence_transformer.return_value
        mock_model_instance.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        
        model = EmbeddingModel()
        embeddings = model.encode(["text1", "text2"])
        
        mock_model_instance.encode.assert_called_with(["text1", "text2"], convert_to_numpy=True)
        assert embeddings.shape == (2, 2)

    def test_encode_error(self, mock_sentence_transformer):
        mock_model_instance = mock_sentence_transformer.return_value
        mock_model_instance.encode.side_effect = Exception("Model error")
        
        model = EmbeddingModel()
        with pytest.raises(Exception):
            model.encode("test")

    def test_get_embedding_dimension(self, mock_sentence_transformer):
        mock_model_instance = mock_sentence_transformer.return_value
        mock_model_instance.get_sentence_embedding_dimension.return_value = 384
        
        model = EmbeddingModel()
        dim = model.get_embedding_dimension()
        
        assert dim == 384
