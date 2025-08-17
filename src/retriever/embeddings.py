from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import logging


class EmbeddingModel:
    """Handles text embeddings using SentenceTransformers"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logging.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logging.info("Embedding model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load embedding model: {e}")
            raise

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text(s) into embeddings

        Args:
            texts: Single text or list of texts to encode

        Returns:
            numpy array of embeddings
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if isinstance(texts, str):
            texts = [texts]

        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logging.error(f"Failed to encode texts: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self.model.get_sentence_embedding_dimension()

    def similarity(
        self, embeddings1: np.ndarray, embeddings2: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarity between embeddings

        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings

        Returns:
            Similarity scores
        """
        from sklearn.metrics.pairwise import cosine_similarity

        return cosine_similarity(embeddings1, embeddings2)
