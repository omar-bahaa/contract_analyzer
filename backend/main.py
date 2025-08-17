from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import tempfile
import shutil
import logging
import asyncio
from datetime import datetime
import yaml

# Import our custom modules
from document_processor import ArabicDocumentProcessor
from rag_engine import IslamicKnowledgeBase, ContractAnalysisRAG
from llm_interface import ArabicLLMInterface, IslamicContractAnalyzer

# Load configuration
def load_config(config_path: str = "../config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {}
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

# Load global configuration
config = load_config()

# Setup logging based on config
log_level = config.get('logging', {}).get('level', 'INFO')
logging.basicConfig(level=getattr(logging, log_level.upper()))
logger = logging.getLogger(__name__)

# Log loaded configuration
logger.info(f"Loaded configuration with LLM backend: {config.get('llm', {}).get('backend', 'ollama')}")
logger.info(f"Using model: {config.get('llm', {}).get('model_name', 'aya')}")

# Pydantic models for API
class ContractAnalysisRequest(BaseModel):
    contract_text: str
    analysis_type: str = "full"  # full, quick, specific
    specific_questions: Optional[List[str]] = None

class DocumentUploadResponse(BaseModel):
    success: bool
    message: str
    document_id: str
    metadata: Dict[str, Any]

class AnalysisResponse(BaseModel):
    success: bool
    analysis_id: str
    compliant_clauses: List[Dict]
    non_compliant_clauses: List[Dict]
    questionable_clauses: List[Dict]
    recommendations: List[str]
    detailed_analysis: str
    detailed_recommendations: str
    confidence: float
    processing_time: float

class QueryRequest(BaseModel):
    question: str
    context: Optional[str] = ""

class QueryResponse(BaseModel):
    success: bool
    answer: str
    confidence: float
    references: List[Dict]
    processing_time: float

# Initialize FastAPI app
app = FastAPI(
    title="Islamic Contract RAG API",
    description="RAG system for analyzing contracts against Islamic regulations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
document_processor = None
knowledge_base = None
rag_engine = None
llm_interface = None
contract_analyzer = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global document_processor, knowledge_base, rag_engine, llm_interface, contract_analyzer
    
    try:
        logger.info("Initializing components...")
        
        # Initialize document processor
        document_processor = ArabicDocumentProcessor()
        logger.info("Document processor initialized")
        
        # Initialize knowledge base using config
        db_config = config.get('database', {})
        knowledge_base = IslamicKnowledgeBase(
            db_path=db_config.get('chroma_db_path', './data/chroma_db'),
            collection_name=db_config.get('collection_name', 'islamic_documents'),
            config=config
        )
        logger.info("Knowledge base initialized")
        
        # Initialize RAG engine
        rag_engine = ContractAnalysisRAG(knowledge_base)
        logger.info("RAG engine initialized")
        
        # Initialize LLM interface using config
        llm_config = config.get('llm', {})
        backend = llm_config.get('backend', 'ollama')
        model_name = llm_config.get('model_name', 'aya')
        
        # Prepare LLM interface parameters
        llm_kwargs = {
            'backend': backend,
            'model_name': model_name
        }
        
        # Add OpenAI-specific parameters if using OpenAI backend
        if backend == 'openai':
            llm_kwargs['openai_api_key'] = llm_config.get('openai_api_key')
            llm_kwargs['openai_base_url'] = llm_config.get('openai_base_url')
            # Use the OpenAI model name
            llm_kwargs['model_name'] = llm_config.get('openai_model', 'gpt-4o')
        
        try:
            llm_interface = ArabicLLMInterface(**llm_kwargs)
            logger.info(f"{backend.title()} LLM interface initialized with model: {llm_kwargs['model_name']}")
        except Exception as e:
            logger.warning(f"{backend.title()} not available: {e}")
            # Fallback logic
            if backend == "ollama":
                try:
                    llm_interface = ArabicLLMInterface(backend="huggingface")
                    logger.info("Fallback: HuggingFace LLM interface initialized")
                except Exception as e2:
                    logger.error(f"No LLM backend available: {e2}")
                    llm_interface = None
            elif backend == "openai":
                try:
                    llm_interface = ArabicLLMInterface(backend="ollama", model_name=model_name)
                    logger.info("Fallback: Ollama LLM interface initialized")
                except Exception as e2:
                    logger.error(f"No LLM backend available: {e2}")
                    llm_interface = None
            else:
                try:
                    llm_interface = ArabicLLMInterface(backend="ollama", model_name=model_name)
                    logger.info("Fallback: Ollama LLM interface initialized")
                except Exception as e2:
                    logger.error(f"No LLM backend available: {e2}")
                    llm_interface = None
        
        # Initialize contract analyzer
        if llm_interface:
            contract_analyzer = IslamicContractAnalyzer(llm_interface)
            logger.info("Contract analyzer initialized")
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Islamic Contract RAG API",
        "status": "running",
        "version": "1.0.0",
        "components": {
            "document_processor": document_processor is not None,
            "knowledge_base": knowledge_base is not None,
            "rag_engine": rag_engine is not None,
            "llm_interface": llm_interface is not None,
            "contract_analyzer": contract_analyzer is not None
        }
    }

@app.get("/stats")
async def get_knowledge_base_stats():
    """Get knowledge base statistics"""
    try:
        if not knowledge_base:
            raise HTTPException(status_code=500, detail="Knowledge base not initialized")
        
        stats = knowledge_base.get_collection_stats()
        return JSONResponse(content=stats)
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_islamic_document")
async def upload_islamic_document(
    file: UploadFile = File(...),
    document_type: str = "فقه",
    source: str = "",
    author: str = "",
    topic: str = ""
):
    """Upload an Islamic document to the knowledge base"""
    try:
        if not document_processor or not knowledge_base:
            raise HTTPException(status_code=500, detail="Components not initialized")
        
        # Validate file type
        if not file.filename.lower().endswith(('.pdf', '.docx', '.doc')):
            raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF and DOCX files are supported.")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Process document
            processed_doc = document_processor.process_document(temp_file_path)
            
            if not processed_doc['full_text']:
                raise HTTPException(status_code=400, detail="No text could be extracted from the document")
            
            # Prepare metadata
            metadata = {
                "document_type": document_type,
                "source": source or file.filename,
                "author": author,
                "topic": topic,
                "filename": file.filename,
                "upload_date": datetime.now().isoformat(),
                **processed_doc['metadata']
            }
            
            # Add to knowledge base
            document_id = knowledge_base.add_islamic_document(
                processed_doc['full_text'],
                metadata
            )
            
            return DocumentUploadResponse(
                success=True,
                message=f"Document uploaded successfully. Extracted {len(processed_doc['full_text'])} characters.",
                document_id=document_id,
                metadata=metadata
            )
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_contract")
async def analyze_contract(
    file: UploadFile = File(...),
    analysis_type: str = "full"
):
    """Analyze a contract document for Islamic compliance"""
    try:
        if not all([document_processor, rag_engine, contract_analyzer]):
            raise HTTPException(status_code=500, detail="Components not initialized")
        
        # Validate file type
        if not file.filename.lower().endswith(('.pdf', '.docx', '.doc')):
            raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF and DOCX files are supported.")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Process contract document
            processed_contract = document_processor.process_document(temp_file_path)
            
            if not processed_contract['full_text']:
                raise HTTPException(status_code=400, detail="No text could be extracted from the contract")
            
            # Extract clauses
            clauses = processed_contract.get('clauses', [])
            if not clauses:
                clauses = document_processor.extract_clauses_from_contract(processed_contract['full_text'])
            
            clause_texts = [clause['text'] for clause in clauses]
            
            # Perform RAG analysis
            analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            rag_results = rag_engine.analyze_contract_compliance(clause_texts)
            
            # Perform LLM analysis if available
            if contract_analyzer:
                final_results = contract_analyzer.analyze_contract_with_llm(
                    rag_results, 
                    processed_contract['full_text']
                )
            else:
                final_results = rag_results
                final_results['detailed_analysis'] = "التحليل المفصل غير متاح - نموذج اللغة غير مُحمل"
                final_results['detailed_recommendations'] = "التوصيات المفصلة غير متاحة - نموذج اللغة غير مُحمل"
            
            return AnalysisResponse(
                success=True,
                analysis_id=analysis_id,
                compliant_clauses=final_results.get('compliant_clauses', []),
                non_compliant_clauses=final_results.get('non_compliant_clauses', []),
                questionable_clauses=final_results.get('questionable_clauses', []),
                recommendations=final_results.get('recommendations', []),
                detailed_analysis=final_results.get('detailed_analysis', ''),
                detailed_recommendations=final_results.get('detailed_recommendations', ''),
                confidence=final_results.get('analysis_confidence', 0.5),
                processing_time=final_results.get('processing_time', 0.0)
            )
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing contract: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_contract_text")
async def analyze_contract_text(request: ContractAnalysisRequest):
    """Analyze contract text directly (without file upload)"""
    try:
        if not all([rag_engine, contract_analyzer]):
            raise HTTPException(status_code=500, detail="Components not initialized")
        
        # Extract clauses from text
        clauses = document_processor.extract_clauses_from_contract(request.contract_text)
        clause_texts = [clause['text'] for clause in clauses]
        
        if not clause_texts:
            # If no clauses detected, treat entire text as one clause
            clause_texts = [request.contract_text]
        
        # Perform RAG analysis
        analysis_id = f"text_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        rag_results = rag_engine.analyze_contract_compliance(clause_texts)
        
        # Perform LLM analysis if available
        if contract_analyzer:
            final_results = contract_analyzer.analyze_contract_with_llm(
                rag_results, 
                request.contract_text
            )
        else:
            final_results = rag_results
            final_results['detailed_analysis'] = "التحليل المفصل غير متاح - نموذج اللغة غير مُحمل"
            final_results['detailed_recommendations'] = "التوصيات المفصلة غير متاحة - نموذج اللغة غير مُحمل"
        
        return AnalysisResponse(
            success=True,
            analysis_id=analysis_id,
            compliant_clauses=final_results.get('compliant_clauses', []),
            non_compliant_clauses=final_results.get('non_compliant_clauses', []),
            questionable_clauses=final_results.get('questionable_clauses', []),
            recommendations=final_results.get('recommendations', []),
            detailed_analysis=final_results.get('detailed_analysis', ''),
            detailed_recommendations=final_results.get('detailed_recommendations', ''),
            confidence=final_results.get('analysis_confidence', 0.5),
            processing_time=final_results.get('processing_time', 0.0)
        )
        
    except Exception as e:
        logger.error(f"Error analyzing contract text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_islamic_knowledge(request: QueryRequest):
    """Query the Islamic knowledge base"""
    try:
        if not knowledge_base:
            raise HTTPException(status_code=500, detail="Knowledge base not initialized")
        
        # Query knowledge base
        results = knowledge_base.query_islamic_knowledge(
            request.question,
            n_results=5
        )
        
        # Get LLM response if available
        llm_response = ""
        confidence = 0.5
        processing_time = 0.0
        
        if contract_analyzer:
            llm_result = contract_analyzer.quick_query(request.question, request.context)
            llm_response = llm_result.text
            confidence = llm_result.confidence
            processing_time = llm_result.processing_time
        else:
            llm_response = "الإجابة المفصلة غير متاحة - نموذج اللغة غير مُحمل"
        
        return QueryResponse(
            success=True,
            answer=llm_response,
            confidence=confidence,
            references=results,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error querying knowledge: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Detailed health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    # Check document processor
    try:
        if document_processor:
            test_text = "نص تجريبي للمعالجة"
            document_processor.normalize_arabic_text(test_text)
            health_status["components"]["document_processor"] = "healthy"
        else:
            health_status["components"]["document_processor"] = "not_initialized"
    except Exception as e:
        health_status["components"]["document_processor"] = f"error: {str(e)}"
    
    # Check knowledge base
    try:
        if knowledge_base:
            stats = knowledge_base.get_collection_stats()
            health_status["components"]["knowledge_base"] = f"healthy (docs: {stats.get('total_documents', 0)})"
        else:
            health_status["components"]["knowledge_base"] = "not_initialized"
    except Exception as e:
        health_status["components"]["knowledge_base"] = f"error: {str(e)}"
    
    # Check LLM interface
    try:
        if llm_interface:
            health_status["components"]["llm_interface"] = f"healthy (backend: {llm_interface.backend})"
        else:
            health_status["components"]["llm_interface"] = "not_initialized"
    except Exception as e:
        health_status["components"]["llm_interface"] = f"error: {str(e)}"
    
    return health_status

if __name__ == "__main__":
    import uvicorn
    api_config = config.get('api', {})
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8000)
    debug = api_config.get('debug', False)
    
    logger.info(f"Starting API server on {host}:{port} (debug={debug})")
    # Note: debug parameter should be passed as reload for uvicorn
    uvicorn.run(app, host=host, port=port, reload=debug)
