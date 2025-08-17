# 🕌 Islamic Contract RAG System

A comprehensive RAG (Retrieval-Augmented Generation) system for analyzing contracts against Islamic regulations and Sharia compliance.

## 🎯 Features

- **Arabic Document Processing**: Advanced OCR and text extraction for Arabic documents
- **Islamic Knowledge Base**: Vector database with Islamic regulations and Fiqh documents
- **Smart Contract Analysis**: AI-powered analysis of contract clauses
- **Compliance Checking**: Automated detection of Sharia-compliant and non-compliant clauses
- **Intelligent Recommendations**: Suggestions for contract modifications
- **User-Friendly Interface**: Web-based GUI for easy interaction

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python
- **Frontend**: Streamlit
- **Document Processing**: PyMuPDF, python-docx, Tesseract OCR
- **Arabic NLP**: CAMeL Tools, PyArabic
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence Transformers
- **LLM**: Ollama (with Arabic models) / Hugging Face Transformers

## 🚀 Quick Start

### 1. Setup Environment Variables
```bash
# Interactive setup (recommended)
./setup_keys.sh

# Or manually copy and edit
cp .env.example .env
# Edit .env file and add your API keys:
# OPENAI_API_KEY=your_openai_api_key_here
# MISTRAL_API_KEY=your_mistral_api_key_here
```

### 2. Run Setup
```bash
./setup.sh
```

### 3. Start the System
```bash
./start_system.sh
```

### 4. Access the Application
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## 🚀 Standalone GPT Application

For a simplified experience using only GPT models without external knowledge base:

```bash
# Set up environment variables first (see above)
./start_standalone.sh
```

This launches a streamlined Streamlit app that:
- Uses GPT-4 for direct contract analysis
- Includes Mistral AI OCR for document processing
- Requires no local knowledge base setup
- Provides immediate Sharia compliance analysis

Access at: http://localhost:8502

## ☁️ Streamlit Cloud Deployment

To deploy the standalone app on Streamlit Cloud:

1. **Fork this repository** on GitHub
2. **Connect to Streamlit Cloud**: https://streamlit.io/cloud
3. **Add secrets**: In your Streamlit Cloud app settings, add:
   ```toml
   # Copy content from .streamlit/secrets.toml.example
   OPENAI_API_KEY = "your_actual_openai_api_key"
   MISTRAL_API_KEY = "your_actual_mistral_api_key"  # optional
   ```
4. **Set app path**: `backend/standalone_app.py`
5. **Deploy**: Your app will be live at `https://yourapp.streamlit.app`

### Streamlit Cloud Features:
- 🔐 Secure API key management via secrets
- 🌐 Public or private app hosting
- 🔄 Automatic updates from GitHub
- 📊 Usage analytics and logs

## 🔑 API Keys Required

This system requires API keys for optimal functionality:

- **OpenAI API Key**: For GPT-based analysis (required for standalone app)
- **Mistral API Key**: For advanced OCR capabilities (optional, enhances document processing)

You can obtain these keys from:
- OpenAI: https://platform.openai.com/api-keys
- Mistral AI: https://console.mistral.ai/

## 📋 Usage

### 1. Add Islamic Documents
- Upload Fiqh documents, Islamic rulings, and regulations
- Supports PDF and DOCX formats
- Automatic text extraction and processing

### 2. Analyze Contracts
- Upload contract files or paste text directly
- Get detailed analysis of each clause
- Receive compliance ratings and recommendations

### 3. Query Knowledge Base
- Ask questions about Islamic rulings
- Get AI-powered answers with references
- Explore the knowledge base interactively

## 🔧 Manual Setup

If the automatic setup fails, follow these steps:

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Tesseract OCR**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr tesseract-ocr-ara
   
   # macOS
   brew install tesseract tesseract-lang
   ```

3. **Install Ollama (Optional)**:
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama pull aya
   ```

4. **Start Services**:
   ```bash
   # Terminal 1: Backend
   ./start_backend.sh
   
   # Terminal 2: Frontend
   ./start_frontend.sh
   ```

## 📊 API Endpoints

- `POST /upload_islamic_document` - Upload Islamic documents
- `POST /analyze_contract` - Analyze contract files
- `POST /analyze_contract_text` - Analyze contract text
- `POST /query` - Query the knowledge base
- `GET /stats` - Get system statistics
- `GET /health` - Health check

## 🔍 Configuration

Edit `config.yaml` to customize:
- LLM backend and models
- Database settings
- Document processing parameters
- Arabic text processing options

## 📁 Directory Structure

```
islamic-contract-rag/
├── backend/           # Backend API code
├── frontend/          # Streamlit frontend
├── data/             # Data storage
│   ├── islamic_documents/
│   ├── contracts/
│   └── chroma_db/
├── test/             # Test files and documentation
├── config.yaml       # Configuration file
├── requirements.txt  # Python dependencies
├── .env.example      # Environment variables template
└── setup.sh         # Setup script
```

## 🆘 Troubleshooting

1. **API not responding**: Check if backend is running on port 8000
2. **OCR not working**: Ensure Tesseract and Arabic language pack are installed
3. **LLM errors**: Verify Ollama is running or switch to HuggingFace backend
4. **Memory issues**: Reduce chunk size in configuration

## 📞 Support

For issues and questions, please check the logs in the `logs/` directory or review the system status at `/health` endpoint.

## ⚖️ Disclaimer

This is a prototype system for educational and development purposes. For production use in Islamic finance or legal contexts, please consult with qualified Islamic scholars and legal experts.

## 📜 License

This project is open source and available under the MIT License.
