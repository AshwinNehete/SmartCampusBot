#!/usr/bin/env python3
"""
Script to run the SmartCampusBot Gradio application
"""

import os
import sys
import logging
import argparse

# Add the src and app directories to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.gradio_app import SmartCampusBotApp
from src.config.settings import Config


def setup_logging(debug=False):
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def check_prerequisites():
    """Check if all prerequisites are met"""
    config = Config()

    # Check if vector store exists
    if not os.path.exists(f"{config.VECTOR_STORE_PATH}.index"):
        logging.error("Vector store not found!")
        logging.error("Please run the setup script first: python scripts/setup_data.py")
        return False

    # Check if FAQ file exists
    if not os.path.exists(config.FAQ_FILE):
        logging.error(f"FAQ file not found: {config.FAQ_FILE}")
        return False

    logging.info("Prerequisites check passed ✅")
    return True


def main():
    """Main function to run the application"""
    parser = argparse.ArgumentParser(description="Run SmartCampusBot")
    parser.add_argument("--port", type=int, help="Port to run the server on")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--skip-checks", action="store_true", help="Skip prerequisite checks"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.debug)

    logging.info("Starting SmartCampusBot...")

    # Check prerequisites
    if not args.skip_checks and not check_prerequisites():
        sys.exit(1)

    try:
        # Create and launch the application
        app = SmartCampusBotApp()

        logging.info("🚀 Launching SmartCampusBot interface...")

        app.launch(share=args.share, port=args.port)

    except KeyboardInterrupt:
        logging.info("👋 Application stopped by user")
    except Exception as e:
        logging.error(f"❌ Failed to start application: {e}")
        raise


if __name__ == "__main__":
    main()
