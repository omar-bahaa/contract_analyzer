#!/usr/bin/env python3
"""
Utility script to populate the Islamic knowledge base with sample documents
"""

import os
import sys
import logging
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from rag_engine import IslamicKnowledgeBase

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def populate_knowledge_base():
    """Populate the knowledge base with sample Islamic documents"""
    
    # Initialize knowledge base
    kb = IslamicKnowledgeBase(
        db_path="./data/chroma_db",
        collection_name="islamic_documents"
    )
    
    # Path to Islamic documents
    docs_path = Path("./data/islamic_documents")
    
    if not docs_path.exists():
        logger.error(f"Documents directory not found: {docs_path}")
        return
    
    # Document metadata mapping
    doc_metadata = {
        "riba_rulings.txt": {
            "document_type": "فقه",
            "source": "أحكام الربا في الشريعة الإسلامية",
            "author": "مجموعة مؤلفين",
            "topic": "الربا",
            "language": "ar"
        },
        "gharar_rulings.txt": {
            "document_type": "فقه",
            "source": "أحكام الغرر في الفقه الإسلامي",
            "author": "مجموعة مؤلفين",
            "topic": "الغرر",
            "language": "ar"
        },
        "contract_rulings.txt": {
            "document_type": "فقه",
            "source": "أحكام العقود في الشريعة الإسلامية",
            "author": "مجموعة مؤلفين",
            "topic": "العقود",
            "language": "ar"
        }
    }
    
    # Process each document
    for doc_file in docs_path.glob("*.txt"):
        try:
            logger.info(f"Processing document: {doc_file.name}")
            
            # Read document content
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                logger.warning(f"Empty document: {doc_file.name}")
                continue
            
            # Get metadata
            metadata = doc_metadata.get(doc_file.name, {
                "document_type": "فقه",
                "source": doc_file.stem,
                "author": "غير محدد",
                "topic": "أحكام إسلامية",
                "language": "ar"
            })
            
            # Add to knowledge base
            doc_id = kb.add_islamic_document(content, metadata, doc_file.stem)
            
            logger.info(f"Added document {doc_id} with {len(content)} characters")
            
        except Exception as e:
            logger.error(f"Error processing {doc_file.name}: {e}")
    
    # Get final statistics
    stats = kb.get_collection_stats()
    logger.info(f"Knowledge base populated successfully!")
    logger.info(f"Total documents: {stats.get('total_documents', 0)}")
    
    if 'document_types' in stats:
        logger.info("Document types:")
        for doc_type, count in stats['document_types'].items():
            logger.info(f"  - {doc_type}: {count}")

def test_knowledge_base():
    """Test the knowledge base with sample queries"""
    
    logger.info("Testing knowledge base...")
    
    # Initialize knowledge base
    kb = IslamicKnowledgeBase(
        db_path="./data/chroma_db",
        collection_name="islamic_documents"
    )
    
    # Test queries
    test_queries = [
        "ما حكم الربا في الإسلام؟",
        "ما هو الغرر وما حكمه؟",
        "ما هي شروط العقد في الشريعة؟",
        "ما هي البدائل الشرعية للربا؟",
        "أنواع الغرر في العقود"
    ]
    
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        results = kb.query_islamic_knowledge(query, n_results=3)
        
        logger.info(f"Found {len(results)} results:")
        for i, result in enumerate(results[:2]):
            logger.info(f"  Result {i+1}: {result['text'][:100]}...")
            logger.info(f"  Source: {result['metadata'].get('source', 'Unknown')}")
            logger.info(f"  Distance: {result.get('distance', 'N/A')}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Populate Islamic knowledge base")
    parser.add_argument("--populate", action="store_true", help="Populate knowledge base")
    parser.add_argument("--test", action="store_true", help="Test knowledge base")
    parser.add_argument("--reset", action="store_true", help="Reset knowledge base")
    
    args = parser.parse_args()
    
    try:
        if args.reset:
            # Remove existing database
            import shutil
            db_path = Path("./data/chroma_db")
            if db_path.exists():
                shutil.rmtree(db_path)
                logger.info("Knowledge base reset successfully")
        
        if args.populate or not any([args.test, args.reset]):
            populate_knowledge_base()
        
        if args.test:
            test_knowledge_base()
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
