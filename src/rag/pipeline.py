import json
import logging
from typing import List, Dict, Optional, Tuple
import time

from ..retriever.embeddings import EmbeddingModel
from ..retriever.vector_store import VectorStore
from ..generator.ollama_client import OllamaClient
from ..generator.prompt_templates import PromptTemplates
from ..config.settings import Config


class RAGPipeline:
    """Complete RAG pipeline for SmartCampusBot"""

    def __init__(self, config: Config = None):
        self.config = config or Config()

        # Initialize components
        self.embedding_model = None
        self.vector_store = None
        self.llm_client = None
        self.prompt_templates = PromptTemplates()

        # Performance tracking
        self.query_count = 0
        self.total_response_time = 0.0

        self._initialize_components()

    def _initialize_components(self):
        """Initialize all RAG components"""
        logging.info("Initializing RAG pipeline components...")

        # Initialize embedding model
        self.embedding_model = EmbeddingModel(self.config.EMBEDDING_MODEL)

        # Initialize vector store
        embedding_dim = self.embedding_model.get_embedding_dimension()
        self.vector_store = VectorStore(
            dimension=embedding_dim, index_path=self.config.VECTOR_STORE_PATH
        )

        # Try to load existing vector store
        if not self.vector_store.load():
            logging.info(
                "No existing vector store found, will need to build from FAQ data"
            )

        # Initialize LLM client
        self.llm_client = OllamaClient(
            base_url=self.config.OLLAMA_BASE_URL, model=self.config.OLLAMA_MODEL
        )

        logging.info("RAG pipeline initialized successfully")

    def build_knowledge_base(self, faq_data_path: str):
        """
        Build the knowledge base from FAQ data

        Args:
            faq_data_path: Path to the FAQ JSON file
        """
        logging.info(f"Building knowledge base from {faq_data_path}")

        # Load FAQ data
        with open(faq_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        faqs = data.get("faqs", [])
        if not faqs:
            raise ValueError("No FAQ data found in the file")

        # Prepare documents for embedding
        documents = []
        texts_to_embed = []

        for faq in faqs:
            # Combine question and answer for better retrieval
            combined_text = f"{faq['question']} {faq['answer']}"
            texts_to_embed.append(combined_text)
            documents.append(faq)

        # Generate embeddings
        logging.info(f"Generating embeddings for {len(documents)} FAQ entries")
        embeddings = self.embedding_model.encode(texts_to_embed)

        # Add to vector store
        self.vector_store.add_documents(embeddings, documents)

        # Save the vector store
        self.vector_store.save()

        logging.info(f"Knowledge base built successfully with {len(documents)} entries")

    def build_knowledge_base_from_files(self, faq_path: str, intents_path: str | None = None):
        logging.info("Building knowledge base from FAQ%s",
                    " + intents" if intents_path else "")

        documents: List[Dict] = []
        texts_to_embed: List[str] = []

        # ---- Load FAQ ----
        with open(faq_path, "r", encoding="utf-8") as f:
            faq_data = json.load(f)
        faqs = faq_data.get("faqs", [])
        if not faqs:
            raise ValueError("No FAQ data found in the file")
        for faq in faqs:
            text = f"{faq['question']} {faq['answer']}"
            texts_to_embed.append(text)
            documents.append({
                "type": "faq",
                "question": faq.get("question"),
                "answer": faq.get("answer"),
                "category": faq.get("category"),
                "keywords": faq.get("keywords", []),
                "_raw": faq,
            })

        # ---- Load Intents (optional) ----
        if intents_path:
            with open(intents_path, "r", encoding="utf-8") as f:
                intents_data = json.load(f)
            intents = intents_data.get("intents", [])
            for it in intents:
                tag = it.get("tag")
                responses = it.get("responses", [])
                # top_resp = responses[0] if responses else ""
                for pattern in it.get("patterns", []):
                    # You can choose different compositions; including tag helps retrieval cluster similar patterns
                    text = f"{pattern} {tag}"
                    # Optionally: text = f"{pattern} {tag} {top_resp}"
                    texts_to_embed.append(text)
                    documents.append({
                        "type": "intent",
                        "tag": tag,
                        "pattern": pattern,
                        "responses": responses,
                        "_raw": it,
                    })

        logging.info("Generating embeddings for %d documents", len(documents))
        embeddings = self.embedding_model.encode(texts_to_embed)

        self.vector_store.add_documents(embeddings, documents)
        self.vector_store.save()
        logging.info("Knowledge base built successfully with %d entries", len(documents))

    def retrieve_relevant_docs(
        self, query: str, k: int = None
    ) -> List[Tuple[Dict, float]]:
        """
        Retrieve relevant documents for a query

        Args:
            query: User query
            k: Number of documents to retrieve

        Returns:
            List of (document, score) tuples
        """
        k = k or self.config.TOP_K_RESULTS

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])

        # Search vector store
        results = self.vector_store.search(query_embedding, k=k)

        return results

    def generate_response(self, query: str, context_docs: List[Dict]) -> str:
        """
        Generate response using LLM with context

        Args:
            query: User query
            context_docs: Retrieved context documents

        Returns:
            Generated response
        """
        if not context_docs:
            # Use fallback prompt if no context
            prompt = self.prompt_templates.create_fallback_prompt(query)
        else:
            # Create QA prompt with context
            prompt = self.prompt_templates.create_qa_prompt(query, context_docs)

        # Generate response
        response = self.llm_client.generate_response(
            prompt=prompt, max_tokens=self.config.MAX_RESPONSE_LENGTH
        )

        return response

    def query(self, user_query: str) -> Dict:
        """
        Process a user query through the complete RAG pipeline

        Args:
            user_query: User's question

        Returns:
            Dictionary containing response and metadata
        """
        start_time = time.time()

        try:
            # Retrieve relevant documents
            retrieved_docs = self.retrieve_relevant_docs(user_query)

            # Extract documents (without scores for generation)
            context_docs = [doc for doc, score in retrieved_docs]

            # Generate response
            response = self.generate_response(user_query, context_docs)

            # Calculate response time
            response_time = time.time() - start_time

            # Update performance metrics
            self.query_count += 1
            self.total_response_time += response_time

            result = {
                "query": user_query,
                "response": response,
                "retrieved_docs": retrieved_docs,
                "response_time": response_time,
                "success": True,
            }

            logging.info(f"Query processed in {response_time:.2f}s")
            return result

        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return {
                "query": user_query,
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again later or contact university support.",
                "retrieved_docs": [],
                "response_time": time.time() - start_time,
                "success": False,
                "error": str(e),
            }

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        avg_response_time = (
            self.total_response_time / self.query_count if self.query_count > 0 else 0
        )

        return {
            "total_queries": self.query_count,
            "average_response_time": avg_response_time,
            "total_response_time": self.total_response_time,
        }

    def health_check(self) -> Dict:
        """Perform health check on all components"""
        health_status = {
            "embedding_model": False,
            "vector_store": False,
            "llm_client": False,
            "overall": False,
        }

        try:
            # Check embedding model
            test_embedding = self.embedding_model.encode(["test"])
            health_status["embedding_model"] = len(test_embedding) > 0
        except Exception as e:
            logging.error(f"Embedding model health check failed: {e}")

        try:
            # Check vector store
            health_status["vector_store"] = self.vector_store.index.ntotal > 0
        except Exception as e:
            logging.error(f"Vector store health check failed: {e}")

        try:
            # Check LLM client
            health_status["llm_client"] = self.llm_client.is_model_available()
        except Exception as e:
            logging.error(f"LLM client health check failed: {e}")

        # Overall health
        health_status["overall"] = all(
            [
                health_status["embedding_model"],
                health_status["vector_store"],
                health_status["llm_client"],
            ]
        )

        return health_status
