# Setup GPU environment BEFORE any imports
import os
import sys

# Set CUDA environment variables before ANY imports
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import logging
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import json
import re
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IslamicKnowledgeBase:
    """
    Vector database for Islamic regulations and documents using ChromaDB
    with Arabic-optimized embeddings and retrieval
    """
    
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "islamic_documents", config: Optional[Dict[str, Any]] = None):
        self.db_path = db_path
        self.collection_name = collection_name
        self.config = config or {}
        self.setup_database()
        self.setup_embeddings()
        self.setup_text_splitter()
    
    def setup_database(self):
        """Initialize ChromaDB client and collection"""
        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
            
            # Create or get collection
            try:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
            except Exception:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Using existing collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            raise
    
    def setup_embeddings(self):
        """Setup embeddings backend (sentence transformers or OpenAI)"""
        try:
            # Get embeddings config
            embeddings_config = self.config.get('embeddings', {})
            backend = embeddings_config.get('backend', 'sentence_transformers')
            
            if backend == 'openai':
                self.setup_openai_embeddings(embeddings_config)
            else:
                self.setup_sentence_transformer_embeddings(embeddings_config)
                
        except Exception as e:
            logger.error(f"Error loading primary embedding model: {e}")
            # Final fallback to simpler model
            try:
                logger.info("Falling back to basic sentence transformer model...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                self.embedding_backend = 'sentence_transformers'
                logger.warning(f"⚠️ Using fallback embedding model: all-MiniLM-L6-v2 on CPU")
            except Exception as e2:
                logger.error(f"❌ Failed to load any embedding model: {e2}")
                raise
    
    def setup_openai_embeddings(self, embeddings_config: Dict[str, Any]):
        """Setup OpenAI embeddings"""
        try:
            import openai
            
            # Get OpenAI API key from config or environment
            llm_config = self.config.get('llm', {})
            api_key = llm_config.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
            base_url = llm_config.get('openai_base_url') or os.getenv('OPENAI_BASE_URL')
            
            if not api_key:
                raise Exception("OpenAI API key not provided")
            
            # Initialize OpenAI client
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
                
            self.openai_client = openai.OpenAI(**client_kwargs)
            
            # Set embedding model and parameters
            self.embedding_model_name = embeddings_config.get('openai_embedding_model', 'text-embedding-3-large')
            self.embedding_dimensions = embeddings_config.get('openai_embedding_dimensions', 3072)
            self.embedding_backend = 'openai'
            
            # Test the embeddings
            test_text = "اختبار النموذج للنصوص العربية test"
            test_embedding = self.encode_text(test_text)
            
            logger.info(f"✅ OpenAI embeddings initialized: {self.embedding_model_name}")
            logger.info(f"   Embedding dimensions: {len(test_embedding)}")
            
        except ImportError:
            logger.error("openai library not installed. Run: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Error setting up OpenAI embeddings: {e}")
            raise
    
    def setup_sentence_transformer_embeddings(self, embeddings_config: Dict[str, Any]):
        """Setup sentence transformer embeddings"""
        try:
            # Use a multilingual model that works well with Arabic
            model_name = embeddings_config.get('model_name', 'paraphrase-multilingual-MiniLM-L12-v2')
            
            # Initialize PyTorch and CUDA properly
            import torch
            
            # Clear any existing CUDA context
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            
            device = "cpu"  # Default to CPU
            
            # Proper CUDA initialization
            try:
                # Force CUDA initialization
                torch.cuda.init()
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    # Test CUDA functionality
                    device_name = torch.cuda.get_device_name(0)
                    memory_allocated = torch.cuda.get_device_properties(0).total_memory
                    
                    # Create test tensor to verify GPU access
                    test_tensor = torch.tensor([1.0], device='cuda:0')
                    result = test_tensor + 1
                    
                    device = "cuda:0"
                    logger.info(f"🚀 GPU successfully initialized: {device_name}")
                    logger.info(f"   Total GPU memory: {memory_allocated // (1024**3)} GB")
                    
                    # Clean up test tensor
                    del test_tensor, result
                    torch.cuda.empty_cache()
                    
                else:
                    logger.info("CUDA available but no devices found, using CPU")
                    device = "cpu"
                    
            except Exception as cuda_error:
                logger.info(f"🔄 GPU is likely in use by Ollama (optimal setup)")
                logger.info(f"   CUDA details: {cuda_error}")
                logger.info(f"✅ Using CPU for embeddings (efficient for this workload)")
                device = "cpu"
            
            # Load embedding model with proper device handling
            try:
                logger.info(f"Loading {model_name} on {device}...")
                self.embedding_model = SentenceTransformer(model_name, device=device)
                self.embedding_backend = 'sentence_transformers'
                logger.info(f"✅ Loaded embedding model: {model_name} on {device}")
                
                # Test the model with Arabic text
                test_text = "اختبار النموذج للنصوص العربية والإنجليزية test"
                test_embedding = self.encode_text(test_text)
                logger.info(f"✅ Model test successful, embedding shape: {np.array(test_embedding).shape}")
                
                # Show device info
                if hasattr(self.embedding_model, 'device'):
                    logger.info(f"   Model device: {self.embedding_model.device}")
                    
            except Exception as model_error:
                if device.startswith("cuda"):
                    # Fallback to CPU if GPU fails
                    logger.warning(f"GPU embedding failed: {model_error}")
                    logger.info("Retrying with CPU...")
                    device = "cpu"
                    self.embedding_model = SentenceTransformer(model_name, device=device)
                    self.embedding_backend = 'sentence_transformers'
                    logger.info(f"✅ Loaded embedding model: {model_name} on {device} (fallback)")
                else:
                    raise model_error
                    
        except Exception as e:
            logger.error(f"Error setting up sentence transformer embeddings: {e}")
            raise
    
    def setup_text_splitter(self):
        """Setup text splitter optimized for Arabic text"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "،", "؛", " ", ""]
        )
    
    def encode_text(self, text: str) -> List[float]:
        """Encode text using the configured embedding backend"""
        if hasattr(self, 'embedding_backend') and self.embedding_backend == 'openai':
            # Use OpenAI embeddings
            response = self.openai_client.embeddings.create(
                model=self.embedding_model_name,
                input=text,
                dimensions=self.embedding_dimensions if 'text-embedding-3' in self.embedding_model_name else None
            )
            return response.data[0].embedding
        else:
            # Use sentence transformer
            return self.embedding_model.encode(text).tolist()
    
    
    def add_islamic_document(self, 
                           document_text: str, 
                           metadata: Dict[str, Any],
                           document_id: Optional[str] = None) -> str:
        """Add an Islamic regulation document to the knowledge base"""
        try:
            if not document_id:
                document_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Split document into chunks
            chunks = self.text_splitter.split_text(document_text)
            
            # Prepare data for ChromaDB
            chunk_ids = []
            chunk_texts = []
            chunk_metadatas = []
            chunk_embeddings = []
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:  # Skip very short chunks
                    continue
                
                chunk_id = f"{document_id}_chunk_{i}"
                chunk_ids.append(chunk_id)
                chunk_texts.append(chunk)
                
                # Create embedding
                embedding = self.encode_text(chunk)
                chunk_embeddings.append(embedding)
                
                # Prepare metadata
                chunk_metadata = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "chunk_size": len(chunk),
                    "timestamp": datetime.now().isoformat(),
                    **metadata
                }
                chunk_metadatas.append(chunk_metadata)
            
            # Add to ChromaDB
            self.collection.add(
                ids=chunk_ids,
                documents=chunk_texts,
                metadatas=chunk_metadatas,
                embeddings=chunk_embeddings
            )
            
            logger.info(f"Added document {document_id} with {len(chunk_ids)} chunks")
            return document_id
            
        except Exception as e:
            logger.error(f"Error adding document to knowledge base: {e}")
            raise
    
    def query_islamic_knowledge(self, 
                              query: str, 
                              n_results: int = 10,
                              filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Query the Islamic knowledge base"""
        try:
            # Create query embedding
            query_embedding = self.encode_text(query)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            
            # Get sample of documents to analyze
            sample = self.collection.get(limit=min(100, count))
            
            stats = {
                "total_documents": count,
                "sample_size": len(sample['ids']) if sample['ids'] else 0
            }
            
            if sample['metadatas']:
                # Analyze document types
                doc_types = {}
                sources = {}
                
                for metadata in sample['metadatas']:
                    doc_type = metadata.get('document_type', 'unknown')
                    source = metadata.get('source', 'unknown')
                    
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                    sources[source] = sources.get(source, 0) + 1
                
                stats['document_types'] = doc_types
                stats['sources'] = sources
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"total_documents": 0, "error": str(e)}

class ContractAnalysisRAG:
    """
    RAG pipeline for analyzing contracts against Islamic regulations
    """
    
    def __init__(self, knowledge_base: IslamicKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.setup_reranking()
    
    def setup_reranking(self):
        """Setup reranking criteria for Islamic compliance"""
        self.reranking_criteria = {
            'riba': ['الربا', 'الفوائد', 'الفائدة', 'نسبة إضافية'],
            'gharar': ['الغرر', 'الجهالة', 'عدم التأكد', 'المجهول'],
            'haram_activities': ['الخمر', 'الميسر', 'القمار', 'المحرم'],
            'contracts': ['العقد', 'الاتفاق', 'التعاقد', 'الالتزام'],
            'islamic_principles': ['الحلال', 'الشريعة', 'الإسلامية', 'الفقه'],
            'commercial_terms': ['البيع', 'الشراء', 'التجارة', 'المال']
        }
    
    def enhance_query(self, original_query: str, contract_clauses: List[str]) -> str:
        """Enhance the user query with contract context"""
        try:
            # Extract key terms from contract clauses
            contract_terms = []
            for clause in contract_clauses[:5]:  # Use first 5 clauses for context
                # Simple keyword extraction (can be enhanced with NER)
                words = clause.split()
                important_words = [word for word in words if len(word) > 3]
                contract_terms.extend(important_words[:3])
            
            # Create enhanced query
            enhanced_query = f"{original_query}\n\nسياق العقد: {' '.join(contract_terms[:10])}"
            
            return enhanced_query
            
        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            return original_query
    
    def rerank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Rerank results based on Islamic compliance criteria
        TODO: smarter reranking system (reimplement)
        """
        try:
            for result in results:
                score = 0
                text = result['text'].lower()
                
                # Score based on relevance to Islamic criteria
                for category, keywords in self.reranking_criteria.items():
                    for keyword in keywords:
                        if keyword in text:
                            if category in ['riba', 'gharar', 'haram_activities']:
                                score += 3  # Higher weight for prohibition-related content
                            elif category == 'islamic_principles':
                                score += 2
                            else:
                                score += 1
                
                # Boost recent documents
                # TODO msh mohem awy yasta
                if 'timestamp' in result['metadata']:
                    try:
                        timestamp = datetime.fromisoformat(result['metadata']['timestamp'])
                        days_old = (datetime.now() - timestamp).days
                        if days_old < 30:
                            score += 1
                    except:
                        pass
                
                # Boost authoritative sources
                source = result['metadata'].get('source', '').lower()
                if any(auth in source for auth in ['قرآن', 'حديث', 'فقه', 'شريعة']):
                    score += 2
                
                result['rerank_score'] = score
            
            # Sort by combined score (distance + rerank_score)
            return sorted(results, key=lambda x: x.get('rerank_score', 0), reverse=True)
            
        except Exception as e:
            logger.error(f"Error reranking results: {e}")
            return results
    
    def analyze_contract_compliance(self, 
                                  contract_clauses: List[str], 
                                  query: str = "تحليل مطابقة العقد للشريعة الإسلامية") -> Dict[str, Any]:
        """Analyze contract clauses for Islamic compliance"""
        try:
            analysis_results = {
                'compliant_clauses': [],
                'non_compliant_clauses': [],
                'questionable_clauses': [],
                'recommendations': [],
                'supporting_references': []
            }
            
            for i, clause in enumerate(contract_clauses):
                # Create specific query for this clause
                clause_query = f"{query} - {clause[:200]}..."
                
                # Enhance query with context
                enhanced_query = self.enhance_query(clause_query, [clause])
                
                # Query knowledge base
                results = self.knowledge_base.query_islamic_knowledge(
                    enhanced_query, 
                    n_results=5
                )
                
                # Rerank results
                reranked_results = self.rerank_results(results, clause_query)
                
                # Analyze compliance
                compliance_status = self.determine_compliance_status(clause, reranked_results)
                
                clause_analysis = {
                    'clause_id': i + 1,
                    'clause_text': clause,
                    'compliance_status': compliance_status['status'],
                    'confidence': compliance_status['confidence'],
                    'reasoning': compliance_status['reasoning'],
                    'references': reranked_results[:3]  # Top 3 references
                }
                
                # Categorize based on compliance status
                if compliance_status['status'] == 'compliant':
                    analysis_results['compliant_clauses'].append(clause_analysis)
                elif compliance_status['status'] == 'non_compliant':
                    analysis_results['non_compliant_clauses'].append(clause_analysis)
                else:
                    analysis_results['questionable_clauses'].append(clause_analysis)
                
                # Collect all references
                analysis_results['supporting_references'].extend(reranked_results[:2])
            
            # Generate recommendations
            analysis_results['recommendations'] = self.generate_recommendations(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing contract compliance: {e}")
            return {'error': str(e)}
    
    def determine_compliance_status(self, clause: str, references: List[Dict]) -> Dict[str, Any]:
        """
        Determine compliance status of a clause based on retrieved references
        TODO: should be by LLM
        """
        try:
            # Simple rule-based compliance detection
            clause_lower = clause.lower()
            
            # Check for obvious non-compliance indicators
            non_compliant_indicators = ['الربا', 'الفوائد', 'فائدة مركبة', 'الغرر', 'الميسر', 'القمار']
            compliant_indicators = ['الحلال', 'مطابق للشريعة', 'جائز شرعا', 'وفق الأحكام الإسلامية']
            
            non_compliant_score = sum(1 for indicator in non_compliant_indicators if indicator in clause_lower)
            compliant_score = sum(1 for indicator in compliant_indicators if indicator in clause_lower)
            
            # Analyze references
            reference_score = 0
            relevant_references = []
            
            for ref in references[:3]:
                ref_text = ref['text'].lower()
                if any(indicator in ref_text for indicator in non_compliant_indicators):
                    reference_score -= 1
                    relevant_references.append(ref)
                elif any(indicator in ref_text for indicator in compliant_indicators):
                    reference_score += 1
                    relevant_references.append(ref)
            
            # Determine status
            total_score = compliant_score - non_compliant_score + reference_score
            
            if total_score > 0:
                status = 'compliant'
                confidence = min(0.8, 0.5 + (total_score * 0.1))
            elif total_score < -1:
                status = 'non_compliant'
                confidence = min(0.8, 0.5 + (abs(total_score) * 0.1))
            else:
                status = 'questionable'
                confidence = 0.4
            
            reasoning = f"تم تحديد الحالة بناء على التحليل النصي والمراجع المسترجعة. النتيجة: {total_score}"
            
            return {
                'status': status,
                'confidence': confidence,
                'reasoning': reasoning,
                'relevant_references': relevant_references
            }
            
        except Exception as e:
            logger.error(f"Error determining compliance status: {e}")
            return {
                'status': 'unknown',
                'confidence': 0.0,
                'reasoning': f"خطأ في التحليل: {str(e)}",
                'relevant_references': []
            }
    
    def generate_recommendations(self, analysis_results: Dict) -> List[str]:
        """
        Generate recommendations based on analysis results
        TODO: should be by LLM
        """
        recommendations = []
        
        try:
            non_compliant_count = len(analysis_results['non_compliant_clauses'])
            questionable_count = len(analysis_results['questionable_clauses'])
            
            if non_compliant_count > 0:
                recommendations.append(f"يحتوي العقد على {non_compliant_count} بنود غير متوافقة مع الشريعة الإسلامية ويجب تعديلها.")
            
            if questionable_count > 0:
                recommendations.append(f"يحتوي العقد على {questionable_count} بنود تحتاج لمراجعة إضافية من متخصص في الفقه الإسلامي.")
            
            # Specific recommendations based on common issues
            for clause in analysis_results['non_compliant_clauses']:
                if 'الربا' in clause['clause_text'] or 'الفوائد' in clause['clause_text']:
                    recommendations.append("يجب إزالة أي بنود تتعلق بالربا أو الفوائد واستبدالها ببدائل إسلامية مثل المرابحة أو المشاركة.")
                
                if 'الغرر' in clause['clause_text']:
                    recommendations.append("يجب توضيح البنود الغامضة وإزالة عناصر الغرر لضمان الوضوح التام في العقد.")
            
            if not recommendations:
                recommendations.append("العقد يبدو متوافقاً مع الأحكام الإسلامية بشكل عام.")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("حدث خطأ في توليد التوصيات.")
        
        return recommendations

# Example usage
if __name__ == "__main__":
    # Initialize knowledge base
    kb = IslamicKnowledgeBase()
    
    # Initialize RAG system
    rag = ContractAnalysisRAG(kb)
    
    # Example: Add a sample Islamic document
    sample_document = """
    الربا محرم في الإسلام بجميع أشكاله وصوره. 
    قال الله تعالى: "وأحل الله البيع وحرم الربا".
    والربا هو الزيادة المشروطة في الدين مقابل الأجل أو الزيادة في أحد البدلين المتجانسين.
    """
    
    metadata = {
        "document_type": "فقه",
        "source": "أحكام الربا في الفقه الإسلامي",
        "author": "مؤلف تجريبي",
        "topic": "الربا"
    }
    
    # Add to knowledge base
    doc_id = kb.add_islamic_document(sample_document, metadata)
    print(f"Added document: {doc_id}")
    
    # Example query
    results = kb.query_islamic_knowledge("ما حكم الربا في الإسلام؟")
    print(f"Found {len(results)} results")
    for result in results:
        print(f"- {result['text'][:100]}...")
