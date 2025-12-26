import pytest
import numpy as np
import os
from unittest.mock import MagicMock, patch
from src.retriever.vector_store import VectorStore

class TestVectorStore:
    @pytest.fixture
    def vector_store(self):
        # We can use a real FAISS index for tests since it's fast and easy,
        # but mocking is also fine. Let's use real simple index to test logic 
        # unless it requires complex deps setup which we have.
        # But to be pure unit tests, we should probably verify calls. 
        # However, testing FAISS wrapper logic often is best done with real FAISS 
        # to ensure dimension matching etc works.
        
        # Let's try mocking FAISS to avoid binary dependencies issues in test env 
        # if the env was different, but here we have the env.
        # Let's use real FAISS but mocked save/load io.
        return VectorStore(dimension=4)

    def test_init(self, vector_store):
        assert vector_store.dimension == 4
        assert vector_store.index is not None
        assert vector_store.index.ntotal == 0

    def test_add_documents(self, vector_store):
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        documents = [{"id": 1}, {"id": 2}]
        
        vector_store.add_documents(embeddings, documents)
        
        assert vector_store.index.ntotal == 2
        assert len(vector_store.documents) == 2
        assert len(vector_store.metadata) == 2

    def test_search(self, vector_store):
        # Add orthogonal vectors (normalized)
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]  # This is different
        ])
        documents = [{"content": "doc1"}, {"content": "doc2"}]
        vector_store.add_documents(embeddings, documents)
        
        # Query close to doc1
        query = np.array([[0.9, 0.1, 0.0, 0.0]])
        results = vector_store.search(query, k=1)
        
        assert len(results) == 1
        assert results[0][0]["content"] == "doc1"
        # Score should be close to 0.9 (dot product)

    def test_search_empty(self, vector_store):
        query = np.array([[1.0, 0.0, 0.0, 0.0]])
        results = vector_store.search(query)
        assert results == []

    @patch("src.retriever.vector_store.faiss.write_index")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("pickle.dump")
    def test_save(self, mock_pickle, mock_open, mock_write_index, vector_store, tmp_path):
        vector_store.index_path = str(tmp_path / "test_index")
        vector_store.save()
        
        mock_write_index.assert_called_once()
        mock_pickle.assert_called_once()

    @patch("src.retriever.vector_store.faiss.read_index")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("pickle.load")
    def test_load(self, mock_pickle, mock_open, mock_read_index, vector_store, tmp_path):
        vector_store.index_path = str(tmp_path / "test_index")
        
        # Mock pickle return
        mock_pickle.return_value = {
            "documents": [{"test": "doc"}],
            "metadata": [{}],
            "dimension": 4
        }
        
        success = vector_store.load()
        
        assert success is True
        mock_read_index.assert_called_once()
        assert len(vector_store.documents) == 1

    def test_load_no_file(self, vector_store):
        vector_store.index_path = "/non_existent/path"
        success = vector_store.load()
        assert success is False
