import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple
import logging


class VectorStore:
    """FAISS-based vector store for semantic search"""

    def __init__(self, dimension: int, index_path: str = None):
        self.dimension = dimension
        self.index_path = index_path
        self.index = None
        self.documents = []
        self.metadata = []

        # Initialize FAISS index
        self._initialize_index()

    def _initialize_index(self):
        """Initialize FAISS index"""
        self.index = faiss.IndexFlatIP(
            self.dimension
        )  # Inner product for cosine similarity
        logging.info(f"Initialized FAISS index with dimension: {self.dimension}")

    def add_documents(
        self, embeddings: np.ndarray, documents: List[Dict], metadata: List[Dict] = None
    ):
        """
        Add documents and their embeddings to the vector store

        Args:
            embeddings: Document embeddings
            documents: List of document dictionaries
            metadata: Optional metadata for each document
        """
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Add to FAISS index
        self.index.add(embeddings.astype("float32"))

        # Store documents and metadata
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))

        logging.info(f"Added {len(documents)} documents to vector store")

    def search(
        self, query_embedding: np.ndarray, k: int = 3
    ) -> List[Tuple[Dict, float]]:
        """
        Search for similar documents

        Args:
            query_embedding: Query embedding
            k: Number of results to return

        Returns:
            List of (document, score) tuples
        """
        if self.index.ntotal == 0:
            logging.warning("Vector store is empty")
            return []

        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(
            query_embedding, keepdims=True
        )

        # Search
        scores, indices = self.index.search(query_embedding.astype("float32"), k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))

        return results

    def save(self, path: str = None):
        """Save the vector store to disk"""
        save_path = path or self.index_path
        if not save_path:
            raise ValueError("No path specified for saving")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, f"{save_path}.index")

        # Save documents and metadata
        with open(f"{save_path}.pkl", "wb") as f:
            pickle.dump(
                {
                    "documents": self.documents,
                    "metadata": self.metadata,
                    "dimension": self.dimension,
                },
                f,
            )

        logging.info(f"Vector store saved to {save_path}")

    def load(self, path: str = None):
        """Load the vector store from disk"""
        load_path = path or self.index_path
        if not load_path:
            raise ValueError("No path specified for loading")

        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{load_path}.index")

            # Load documents and metadata
            with open(f"{load_path}.pkl", "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.metadata = data["metadata"]
                self.dimension = data["dimension"]

            logging.info(f"Vector store loaded from {load_path}")
            return True
        except FileNotFoundError:
            logging.info("No existing vector store found, will create new one")
            return False
        except Exception as e:
            logging.error(f"Failed to load vector store: {e}")
            return False
