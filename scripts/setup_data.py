#!/usr/bin/env python3
"""
Script to set up the knowledge base for SmartCampusBot
"""

import os
import sys
import logging
import argparse
import requests

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.rag.pipeline import RAGPipeline
from src.config.settings import Config


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("setup_data.log")],
    )


def check_ollama_connection(config):
    """Check if Ollama is running and has the required model"""
    from src.generator.ollama_client import OllamaClient

    try:
        client = OllamaClient(config.OLLAMA_BASE_URL, config.OLLAMA_MODEL)

        if not client.is_model_available():
            logging.warning(f"Model {config.OLLAMA_MODEL} not found")
            response = input(
                f"Would you like to pull the model {config.OLLAMA_MODEL}? (y/n): "
            )
            if response.lower() == "y":
                if client.pull_model():
                    logging.info("Model pulled successfully")
                    return True
                else:
                    logging.error("Failed to pull model")
                    return False
            else:
                return False

        logging.info("Ollama connection and model check passed")
        return True

    # except Exception as e:
    #     logging.error(f"Failed to connect to Ollama: {e}")
    #     logging.error("Make sure Ollama is running with: ollama serve")
    #     return False
    except requests.RequestException as e:
        logging.error(f"Failed to connect to Ollama at {config.OLLAMA_BASE_URL}: {e}")
        raise
    except (KeyError, TypeError, ValueError) as e:
        logging.error(f"Unexpected /api/tags payload from {config.OLLAMA_BASE_URL}: {e}")
        raise


def build_knowledge_base(config, faq_file=None, intents_file=None):
    """Build the knowledge base from FAQ data"""
    faq_path = faq_file or config.FAQ_FILE
    intents_path = intents_file or config.INTENTS_FILE

    if not os.path.exists(faq_path):
        logging.error(f"FAQ file not found: {faq_path}")
        return False

    try:
        # Initialize RAG pipeline
        logging.info("Initializing RAG pipeline...")
        pipeline = RAGPipeline(config)

        # Build knowledge base
        logging.info(f"Building knowledge base from {faq_path}")
        pipeline.build_knowledge_base_from_files(faq_path, intents_path)

        # Test the pipeline
        logging.info("Testing the pipeline...")
        test_query = "How do I register for classes?"
        result = pipeline.query(test_query)

        if result["success"]:
            logging.info(
                f"Test query successful! Response time: {result['response_time']:.2f}s"
            )
            logging.info(f"Test response: {result['response'][:100]}...")
            return True
        else:
            logging.error("Test query failed")
            return False

    except Exception as e:
        logging.error(f"Failed to build knowledge base: {e}")
        return False


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup SmartCampusBot knowledge base")
    parser.add_argument("--faq-file", help="Path to FAQ JSON file")
    parser.add_argument(
        "--skip-ollama-check", action="store_true", help="Skip Ollama connection check"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Load configuration
    config = Config()

    logging.info("Starting SmartCampusBot data setup...")

    # Check Ollama connection
    if not args.skip_ollama_check:
        logging.info("Checking Ollama connection...")
        if not check_ollama_connection(config):
            logging.error("Ollama check failed. Setup aborted.")
            sys.exit(1)

    # Create data directory if it doesn't exist
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # Build knowledge base
    if build_knowledge_base(config, args.faq_file):
        logging.info("✅ SmartCampusBot setup completed successfully!")
        logging.info(f"Vector store saved to: {config.VECTOR_STORE_PATH}")
        logging.info("You can now run the chatbot with: python scripts/run_app.py")
    else:
        logging.error("❌ Setup failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
