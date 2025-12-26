import pytest
from unittest.mock import MagicMock, patch
from src.rag.pipeline import RAGPipeline

class TestRAGPipeline:
    @pytest.fixture
    def mock_components(self):
        with patch("src.rag.pipeline.EmbeddingModel") as mock_emb, \
             patch("src.rag.pipeline.VectorStore") as mock_vs, \
             patch("src.rag.pipeline.OllamaClient") as mock_client:
            
            yield {
                "emb": mock_emb,
                "vs": mock_vs,
                "client": mock_client
            }

    def test_init(self, mock_components):
        pipeline = RAGPipeline()
        assert pipeline.embedding_model is not None
        assert pipeline.vector_store is not None
        assert pipeline.llm_client is not None

    def test_retrieve_relevant_docs(self, mock_components):
        mock_emb_instance = mock_components["emb"].return_value
        mock_vs_instance = mock_components["vs"].return_value
        
        # Setup mocks
        mock_emb_instance.encode.return_value = "query_embedding"
        mock_vs_instance.search.return_value = [({"doc": 1}, 0.9)]
        
        pipeline = RAGPipeline()
        results = pipeline.retrieve_relevant_docs("query")
        
        assert len(results) == 1
        mock_emb_instance.encode.assert_called_with(["query"])
        mock_vs_instance.search.assert_called()

    def test_generate_response_with_context(self, mock_components):
        mock_client_instance = mock_components["client"].return_value
        mock_client_instance.generate_response.return_value = "Response"
        
        pipeline = RAGPipeline()
        context = [{"question": "q", "answer": "a"}]
        response = pipeline.generate_response("query", context)
        
        assert response == "Response"
        mock_client_instance.generate_response.assert_called()

    def test_generate_response_no_context(self, mock_components):
        mock_client_instance = mock_components["client"].return_value
        mock_client_instance.generate_response.return_value = "Fallback Response"
        
        pipeline = RAGPipeline()
        response = pipeline.generate_response("query", [])
        
        assert response == "Fallback Response"
        # Verify fallback prompt used (indirectly via client call args if strict, 
        # but just ensuring it runs is good for now)
        mock_client_instance.generate_response.assert_called()
        call_args = mock_client_instance.generate_response.call_args
        assert "no relevant information was found" in call_args[1]['prompt']

    def test_query_success(self, mock_components):
        mock_vs_instance = mock_components["vs"].return_value
        mock_client_instance = mock_components["client"].return_value
        
        mock_vs_instance.search.return_value = [({"doc": 1}, 0.9)]
        mock_client_instance.generate_response.return_value = "Final Answer"
        
        pipeline = RAGPipeline()
        result = pipeline.query("My Question")
        
        assert result["success"] is True
        assert result["response"] == "Final Answer"
        assert result["query"] == "My Question"
        assert len(result["retrieved_docs"]) == 1
        assert pipeline.query_count == 1

    def test_query_error(self, mock_components):
        mock_vs_instance = mock_components["vs"].return_value
        mock_vs_instance.search.side_effect = Exception("Search failed")
        
        pipeline = RAGPipeline()
        result = pipeline.query("My Question")
        
        assert result["success"] is False
        assert "error" in result
        assert "technical difficulties" in result["response"]

    def test_health_check_healthy(self, mock_components):
        mock_emb_instance = mock_components["emb"].return_value
        mock_vs_instance = mock_components["vs"].return_value
        mock_client_instance = mock_components["client"].return_value
        
        mock_emb_instance.encode.return_value = ["emb"]
        mock_vs_instance.index.ntotal = 10
        mock_client_instance.is_model_available.return_value = True
        
        pipeline = RAGPipeline()
        health = pipeline.health_check()
        
        assert health["overall"] is True
        assert health["embedding_model"] is True
        assert health["vector_store"] is True
        assert health["llm_client"] is True

    def test_health_check_unhealthy(self, mock_components):
        mock_client_instance = mock_components["client"].return_value
        mock_client_instance.is_model_available.return_value = False
        
        pipeline = RAGPipeline()
        health = pipeline.health_check()
        
        assert health["overall"] is False
        assert health["llm_client"] is False
