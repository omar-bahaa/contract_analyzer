#!/bin/bash

# Islamic Contract RAG System Setup Script
echo "🕌 Setting up Islamic Contract RAG System..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python 3 found"

# Check if pip is installed
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo "❌ pip is not installed. Please install pip."
    exit 1
fi

echo "✅ pip found"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Install Tesseract OCR and Arabic language pack
echo "🔍 Setting up Tesseract OCR..."
if command -v apt-get &> /dev/null; then
    # Ubuntu/Debian
    echo "Installing Tesseract for Ubuntu/Debian..."
    sudo apt-get update
    sudo apt-get install -y tesseract-ocr tesseract-ocr-ara
elif command -v yum &> /dev/null; then
    # CentOS/RHEL
    echo "Installing Tesseract for CentOS/RHEL..."
    sudo yum install -y tesseract tesseract-langpack-ara
elif command -v brew &> /dev/null; then
    # macOS
    echo "Installing Tesseract for macOS..."
    brew install tesseract tesseract-lang
else
    echo "⚠️ Could not detect package manager. Please install Tesseract OCR manually:"
    echo "   - Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-ara"
    echo "   - CentOS/RHEL: sudo yum install tesseract tesseract-langpack-ara"
    echo "   - macOS: brew install tesseract tesseract-lang"
fi

# Setup Ollama (optional)
echo "🤖 Setting up Ollama (optional)..."
read -p "Do you want to install Ollama for local LLM support? (y/n): " install_ollama

if [ "$install_ollama" = "y" ] || [ "$install_ollama" = "Y" ]; then
    if command -v curl &> /dev/null; then
        echo "📥 Installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
        
        echo "🔽 Pulling Arabic model (aya)..."
        ollama pull aya
        
        echo "🚀 Starting Ollama service..."
        ollama serve &
        sleep 5
        
        echo "✅ Ollama setup complete"
    else
        echo "❌ curl not found. Please install Ollama manually from https://ollama.ai"
    fi
else
    echo "⏭️ Skipping Ollama installation"
fi

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/islamic_documents
mkdir -p data/contracts
mkdir -p data/chroma_db

# Create sample Islamic documents
echo "📄 Creating sample Islamic documents..."
cat > data/islamic_documents/sample_riba_ruling.txt << 'EOF'
أحكام الربا في الشريعة الإسلامية

الربا محرم في الإسلام بجميع أشكاله وصوره، وقد جاء تحريمه في القرآن الكريم والسنة النبوية الشريفة.

قال الله تعالى: "وَأَحَلَّ اللَّهُ الْبَيْعَ وَحَرَّمَ الرِّبَا" (البقرة: 275)

وقال تعالى: "يَا أَيُّهَا الَّذِينَ آمَنُوا اتَّقُوا اللَّهَ وَذَرُوا مَا بَقِيَ مِنَ الرِّبَا إِنْ كُنْتُمْ مُؤْمِنِينَ" (البقرة: 278)

أنواع الربا:
1. ربا الفضل: وهو الزيادة في أحد البدلين المتجانسين
2. ربا النسيئة: وهو الزيادة مقابل التأجيل في الدين

الحكمة من تحريم الربا:
- منع استغلال المحتاجين
- تحقيق العدالة الاجتماعية
- منع تركز الثروة في أيدي قلة
- تشجيع الاستثمار الحقيقي

البدائل الشرعية للربا:
- المرابحة
- المشاركة
- المضاربة
- الإجارة
- السلم والاستصناع
EOF

cat > data/islamic_documents/sample_gharar_ruling.txt << 'EOF'
أحكام الغرر في الفقه الإسلامي

الغرر في اللغة: الخطر والجهالة وعدم التأكد من الشيء

الغرر في الاصطلاح الشرعي: هو ما كان مجهول العاقبة، لا يدرى أيحصل أم لا

حكم الغرر:
الغرر محرم في الشريعة الإسلامية، لقول النبي صلى الله عليه وسلم: "نهى رسول الله صلى الله عليه وسلم عن بيع الغرر" (رواه مسلم)

أنواع الغرر:
1. الغرر في المعقود عليه: كبيع الطير في الهواء
2. الغرر في الثمن: كأن يقول بعتك بما يرضى به فلان
3. الغرر في أجل التسليم: كأن يقول بعتك عند قدوم الحجاج
4. الغرر في صفة المعقود عليه: كبيع الثوب دون تحديد لونه أو نوعه

شروط انتفاء الغرر:
- العلم بالمعقود عليه
- العلم بالثمن
- العلم بأجل التسليم
- القدرة على التسليم
- تحديد الكمية والنوعية

الغرر المغتفر:
- الغرر اليسير الذي لا يمكن الاحتراز منه
- ما دعت إليه الحاجة والضرورة
EOF

# Create configuration file
echo "⚙️ Creating configuration file..."
cat > config.yaml << 'EOF'
# Islamic Contract RAG System Configuration

# API Configuration
api:
  host: "0.0.0.0"
  port: 8000
  debug: false

# Frontend Configuration
frontend:
  host: "0.0.0.0"
  port: 8501

# Database Configuration
database:
  chroma_db_path: "./data/chroma_db"
  collection_name: "islamic_documents"

# LLM Configuration
llm:
  backend: "ollama"  # ollama, huggingface
  model_name: "aya"  # for ollama
  temperature: 0.7
  max_tokens: 500

# Document Processing Configuration
document_processing:
  supported_formats: ["pdf", "docx", "doc"]
  chunk_size: 1000
  chunk_overlap: 200
  ocr_languages: ["ara", "eng"]

# Arabic Text Processing
arabic_processing:
  normalize_alef: true
  normalize_teh_marbuta: true
  remove_diacritics: true
  normalize_hamza: true

# Data Paths
paths:
  islamic_documents: "./data/islamic_documents"
  contracts: "./data/contracts"
  uploads: "./data/uploads"

# Logging
logging:
  level: "INFO"
  file: "./logs/system.log"
EOF

# Create startup scripts
echo "🚀 Creating startup scripts..."

# Backend startup script
cat > start_backend.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Islamic Contract RAG Backend..."

# Activate virtual environment
source venv/bin/activate

# Create logs directory
mkdir -p logs

# Start the backend API
echo "Starting FastAPI backend on http://localhost:8000"
cd backend && python main.py
EOF

# Frontend startup script
cat > start_frontend.sh << 'EOF'
#!/bin/bash
echo "🌐 Starting Islamic Contract RAG Frontend..."

# Activate virtual environment
source venv/bin/activate

# Start Streamlit frontend
echo "Starting Streamlit frontend on http://localhost:8501"
cd frontend && streamlit run app.py --server.port 8501 --server.address 0.0.0.0
EOF

# Combined startup script
cat > start_system.sh << 'EOF'
#!/bin/bash
echo "🕌 Starting Islamic Contract RAG System..."

# Function to cleanup background processes
cleanup() {
    echo "🛑 Stopping services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Activate virtual environment
source venv/bin/activate

# Start backend in background
echo "🚀 Starting backend API..."
cd backend && python main.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 5

# Start frontend
echo "🌐 Starting frontend..."
cd frontend && streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
FRONTEND_PID=$!
cd ..

echo "✅ System started successfully!"
echo "🌐 Frontend: http://localhost:8501"
echo "🔌 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID
EOF

# Make scripts executable
chmod +x start_backend.sh
chmod +x start_frontend.sh
chmod +x start_system.sh
chmod +x setup.sh

# Create README
echo "📝 Creating README..."
cat > README.md << 'EOF'
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

1. **Run Setup**:
   ```bash
   ./setup.sh
   ```

2. **Start the System**:
   ```bash
   ./start_system.sh
   ```

3. **Access the Application**:
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

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
├── config.yaml       # Configuration file
├── requirements.txt  # Python dependencies
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
EOF

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Run: ./start_system.sh"
echo "2. Open: http://localhost:8501"
echo "3. Start uploading Islamic documents and analyzing contracts!"
echo ""
echo "📚 For detailed instructions, see README.md"
