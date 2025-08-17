# 🎓 SmartCampusBot

A conversational AI chatbot designed to provide instant answers to university student questions using Retrieval-Augmented Generation (RAG) technology.

## 🌟 Features

- **Intelligent Q&A**: Answers common university questions using semantic search
- **RAG Architecture**: Combines retrieval and generation for accurate responses
- **Fast Response Time**: Target response time under 2 seconds
- **User-Friendly Interface**: Clean Gradio web interface
- **Comprehensive Coverage**: Handles registration, financial aid, facilities, and more
- **Local Deployment**: Runs entirely on your local machine

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   RAG Pipeline   │───▶│   Response      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │    Components       │
                    │                     │
                    │ • Sentence Trans.   │
                    │ • FAISS Vector DB   │
                    │ • Ollama LLM        │
                    │ • FAQ Knowledge     │
                    └─────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Ollama installed and running**
   ```bash
   # Install Ollama (macOS/Linux)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama service
   ollama serve
   
   # Pull a model (in another terminal)
   ollama pull llama2:7b-chat
   ```

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd smart_campus_bot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your preferred settings
   ```

5. **Initialize the knowledge base**
   ```bash
   python scripts/setup_data.py
   ```

6. **Run the application**
   ```bash
   python scripts/run_app.py
   ```

7. **Open your browser** and navigate to `http://localhost:7860`

Here is how the app will look like:

![App Screenshot](/assets/image.png)

## 📁 Project Structure

```
smart_campus_bot/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── .gitignore               # Git ignore rules
├── data/
│   ├── university_faq.json  # FAQ dataset
│   └── dataset_loader.py    # Data loading utilities
├── src/
│   ├── config/
│   │   └── settings.py      # Configuration management
│   ├── retriever/
│   │   ├── embeddings.py    # Sentence transformer embeddings
│   │   └── vector_store.py  # FAISS vector database
│   ├── generator/
│   │   ├── ollama_client.py # Ollama LLM client
│   │   └── prompt_templates.py # Prompt engineering
│   ├── rag/
│   │   └── pipeline.py      # Main RAG pipeline
│   └── utils/
│       └── helpers.py       # Utility functions
├── app/
│   └── gradio_app.py        # Gradio web interface
└── scripts/
    ├── setup_data.py        # Knowledge base setup
    └── run_app.py           # Application launcher
```

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2:7b-chat

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Vector Store Configuration
VECTOR_STORE_PATH=./data/vector_store
TOP_K_RESULTS=3

# Gradio Configuration
GRADIO_SHARE=False
GRADIO_PORT=7860

# Application Settings
MAX_RESPONSE_LENGTH=500
RESPONSE_TIMEOUT=30
```

## 📊 Dataset

The bot comes with a comprehensive FAQ dataset covering:

- **Class Registration** (enrollment, add/drop procedures)
- **Financial Services** (tuition, financial aid, scholarships)
- **Academic Policies** (graduation requirements, transcripts)
- **Campus Facilities** (library, dining, recreation)
- **Student Services** (health center, career services, ID cards)
- **Technology** (WiFi, IT support, online systems)
- **Transportation** (parking, shuttle services)

### Adding Custom FAQ Data

To add your own university's FAQ data:

1. Edit `data/university_faq.json`
2. Follow the existing JSON structure:
   ```json
   {
     "question": "Your question here",
     "answer": "Detailed answer here",
     "category": "registration|financial|academic|facilities|services|technology|transportation",
     "keywords": ["keyword1", "keyword2", "keyword3"]
   }
   ```
3. Rebuild the knowledge base:
   ```bash
   python scripts/setup_data.py
   ```

## 🔧 Usage

### Basic Usage

1. Start the application
2. Open your browser to `http://localhost:7860`
3. Type your question in the chat interface
4. Get instant AI-powered responses

### Command Line Options

```bash
# Run with custom port
python scripts/run_app.py --port 8080

# Enable public sharing
python scripts/run_app.py --share

# Debug mode
python scripts/run_app.py --debug

# Skip prerequisite checks
python scripts/run_app.py --skip-checks
```

### API Integration

The RAG pipeline can be used programmatically:

```python
from src.rag.pipeline import RAGPipeline
from src.config.settings import Config

# Initialize pipeline
config = Config()
pipeline = RAGPipeline(config)

# Query the bot
result = pipeline.query("How do I register for classes?")
print(result['response'])
print(f"Response time: {result['response_time']:.2f}s")
```

## 🔍 How It Works

### 1. **Question Processing**
- User submits a question through the Gradio interface
- Question is preprocessed and cleaned

### 2. **Document Retrieval**
- Question is encoded using Sentence Transformers
- FAISS performs semantic similarity search
- Top-K most relevant FAQ entries are retrieved

### 3. **Response Generation**
- Retrieved documents provide context
- Ollama generates natural language response
- Response is formatted and returned to user

### 4. **Performance Tracking**
- Response times are monitored
- Usage statistics are collected
- Health checks ensure system reliability

## 🚨 Troubleshooting

### Common Issues

1. **"Failed to connect to Ollama"**
   ```bash
   # Make sure Ollama is running
   ollama serve
   
   # Check if model is available
   ollama list
   
   # Pull model if needed
   ollama pull llama2:7b-chat
   ```

2. **"Vector store not found"**
   ```bash
   # Run the setup script
   python scripts/setup_data.py
   ```

3. **"Module not found" errors**
   ```bash
   # Make sure you're in the virtual environment
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

4. **Slow responses**
   - Use a smaller model (e.g., `llama2:7b-chat` instead of `llama2:13b-chat`)
   - Reduce `MAX_RESPONSE_LENGTH` in .env
   - Ensure sufficient RAM (8GB+ recommended)

### Performance Optimization

- **RAM**: At least 8GB recommended
- **CPU**: Multi-core processor for better performance
- **Storage**: SSD recommended for faster vector store access
- **Model Size**: Balance between quality and speed

## 📈 Monitoring

### Performance Metrics

The application tracks:
- Total queries processed
- Average response time
- Component health status
- Error rates

Access stats through the Gradio interface or programmatically:

```python
stats = pipeline.get_performance_stats()
health = pipeline.health_check()
```

## 🧪 Testing

### Test the Setup

```bash
# Test with setup script
python scripts/setup_data.py

# Manual testing
python -c "
from src.rag.pipeline import RAGPipeline
from src.config.settings import Config

pipeline = RAGPipeline(Config())
result = pipeline.query('How do I register for classes?')
print('Success!' if result['success'] else 'Failed!')
"
```

### Sample Questions

Try these questions to test the bot:

- "How do I register for classes?"
- "When is tuition due?"
- "Where is the library located?"
- "How do I apply for financial aid?"
- "What are the graduation requirements?"

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests (when available)
python -m pytest tests/

# Format code
black src/ app/ scripts/
```

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgments

- **Sentence Transformers** for embedding models
- **FAISS** for efficient similarity search
- **Ollama** for local LLM inference
- **Gradio** for the web interface
- **Hugging Face** for transformer models

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Review existing issues on GitHub
3. Create a new issue with detailed information
4. Include logs and error messages

---

**Built with ❤️ for university students and administrators**