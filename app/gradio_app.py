import gradio as gr
import logging
import os
import sys
from typing import List, Tuple

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.rag.pipeline import RAGPipeline
from src.config.settings import Config
from src.generator.prompt_templates import PromptTemplates

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SmartCampusBotApp:
    """Gradio application for SmartCampusBot"""

    def __init__(self):
        self.config = Config()
        self.rag_pipeline = None
        self.chat_history = []
        self.prompt_templates = PromptTemplates()

        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize the RAG pipeline"""
        try:
            logging.info("Initializing SmartCampusBot...")
            self.rag_pipeline = RAGPipeline(self.config)

            # Check if knowledge base exists, if not build it
            if not os.path.exists(f"{self.config.VECTOR_STORE_PATH}.index"):
                logging.info("Building knowledge base from FAQ data...")
                self.rag_pipeline.build_knowledge_base(self.config.FAQ_FILE)

            # Perform health check
            health_status = self.rag_pipeline.health_check()
            if not health_status["overall"]:
                logging.warning("Some components failed health check:")
                for component, status in health_status.items():
                    if not status and component != "overall":
                        logging.warning(f"  - {component}: FAILED")

            logging.info("SmartCampusBot initialized successfully!")

        except Exception as e:
            logging.error(f"Failed to initialize SmartCampusBot: {e}")
            raise

    def process_message(
        self, message: str, history: List[List[str]]
    ) -> Tuple[str, List[List[str]]]:
        """
        Process user message and return response

        Args:
            message: User's message
            history: Chat history

        Returns:
            Tuple of (empty_string, updated_history)
        """
        if not message.strip():
            return "", history

        try:
            # Process query through RAG pipeline
            result = self.rag_pipeline.query(message.strip())

            response = result["response"]
            response_time = result["response_time"]

            # Add performance info for debugging (optional)
            if logging.getLogger().isEnabledFor(logging.INFO):
                logging.info(f"Response generated in {response_time:.2f}s")

            # Update history
            history.append([message, response])

            return "", history

        except Exception as e:
            logging.error(f"Error processing message: {e}")
            error_response = "I apologize, but I'm experiencing technical difficulties. Please try again later."
            history.append([message, error_response])
            return "", history

    def get_welcome_message(self):
        """Get welcome message for the chatbot"""
        return self.prompt_templates.create_greeting_prompt()

    def clear_chat(self):
        """Clear chat history"""
        return []

    def get_stats(self):
        """Get performance statistics"""
        if self.rag_pipeline:
            stats = self.rag_pipeline.get_performance_stats()
            return f"""
            **Performance Statistics:**
            - Total queries processed: {stats["total_queries"]}
            - Average response time: {stats["average_response_time"]:.2f} seconds
            """
        return "Statistics not available"

    def create_interface(self):
        """Create and configure Gradio interface"""

        # Custom CSS for better styling
        css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        .chat-container {
            height: 500px !important;
        }
        .message-box {
            min-height: 100px !important;
        }
        """

        with gr.Blocks(css=css, title="SmartCampusBot", theme=gr.themes.Soft()) as demo:
            gr.Markdown(
                """
                # 🎓 SmartCampusBot
                ### Your AI-powered University Assistant
                
                Ask me questions about class registration, financial aid, campus facilities, 
                academic policies, and more!
                """
            )

            with gr.Row():
                with gr.Column(scale=4):
                    # Chat interface
                    chatbot = gr.Chatbot(
                        value=[[None, self.get_welcome_message()]],
                        elem_classes=["chat-container"],
                        show_label=False,
                        container=False,
                    )

                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Ask me anything about university services...",
                            show_label=False,
                            elem_classes=["message-box"],
                            scale=4,
                        )
                        submit_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Column(scale=1):
                    # Sidebar with controls and info
                    gr.Markdown("### Quick Actions")

                    clear_btn = gr.Button("Clear Chat", variant="secondary")
                    stats_btn = gr.Button("Show Stats", variant="secondary")

                    # Statistics display
                    stats_display = gr.Markdown("", visible=False)

                    # Sample questions
                    gr.Markdown("""
                    ### Sample Questions:
                    - How do I register for classes?
                    - When is tuition due?
                    - Where is the library?
                    - How do I apply for financial aid?
                    - What are the graduation requirements?
                    """)

            # Event handlers
            def submit_and_clear(message, history):
                """Submit message and clear input"""
                _, new_history = self.process_message(message, history)
                return "", new_history

            # Handle message submission
            submit_btn.click(
                fn=submit_and_clear, inputs=[msg, chatbot], outputs=[msg, chatbot]
            )

            msg.submit(
                fn=submit_and_clear, inputs=[msg, chatbot], outputs=[msg, chatbot]
            )

            # Handle clear chat
            clear_btn.click(fn=self.clear_chat, outputs=chatbot).then(
                fn=lambda: [[None, self.get_welcome_message()]], outputs=chatbot
            )

            # Handle stats display
            def toggle_stats():
                stats_text = self.get_stats()
                return gr.Markdown(stats_text, visible=True)

            stats_btn.click(fn=toggle_stats, outputs=stats_display)

            # Add footer
            gr.Markdown(
                """
                ---
                *SmartCampusBot is powered by RAG (Retrieval-Augmented Generation) technology 
                using Sentence Transformers for embeddings and Ollama for response generation.*
                """,
                elem_classes=["footer"],
            )

        return demo

    def launch(self, share: bool = None, port: int = None):
        """Launch the Gradio application"""
        demo = self.create_interface()

        share_app = share if share is not None else self.config.GRADIO_SHARE
        app_port = port if port is not None else self.config.GRADIO_PORT

        logging.info(f"Launching SmartCampusBot on port {app_port}")

        demo.launch(share=share_app, server_port=app_port, server_name="0.0.0.0")


def main():
    """Main function to run the application"""
    try:
        app = SmartCampusBotApp()
        app.launch()
    except KeyboardInterrupt:
        logging.info("Application stopped by user")
    except Exception as e:
        logging.error(f"Failed to start application: {e}")
        raise


if __name__ == "__main__":
    main()
