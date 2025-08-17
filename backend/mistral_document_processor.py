"""
Standalone Document Processor using Mistral AI for OCR
Alternative to tesseract OCR using Mistral's vision capabilities
"""

import os
import logging
import base64
import io
from typing import Dict, Any, Optional, List
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import requests
from docx import Document
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MistralDocumentProcessor:
    """
    Document processor using Mistral AI for OCR instead of tesseract
    """
    
    def __init__(self, config_path: str = "../config.yaml"):
        """Initialize the processor with configuration"""
        self.config = self._load_config(config_path)
        self.mistral_api_key = self._get_mistral_api_key()
        self.mistral_api_url = "https://api.mistral.ai/v1/chat/completions"
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return {}
    
    def _get_mistral_api_key(self) -> str:
        """Get Mistral API key from config or environment"""
        # Try config first, then environment variable
        api_key = (
            self.config.get('mistral', {}).get('api_key') or
            os.getenv('MISTRAL_API_KEY')
        )
        
        if not api_key:
            logger.warning("Mistral API key not found. OCR functionality will be limited.")
        
        return api_key
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process document and extract text using appropriate method
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.pdf':
                return self._process_pdf(file_path)
            elif file_extension in ['.docx', '.doc']:
                return self._process_word(file_path)
            elif file_extension == '.txt':
                return self._process_text(file_path)
            elif file_extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                return self._process_image(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return {
                "full_text": f"خطأ في معالجة الملف: {str(e)}",
                "pages": [],
                "metadata": {
                    "file_name": file_path.name,
                    "file_size": 0,
                    "page_count": 0,
                    "processing_method": "error"
                }
            }
    
    def _process_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Process PDF file"""
        try:
            doc = fitz.open(file_path)
            pages = []
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Try to extract text directly first
                page_text = page.get_text()
                
                # If text is minimal or empty, use Mistral OCR
                if len(page_text.strip()) < 50:  # Threshold for minimal text
                    logger.info(f"Using Mistral OCR for page {page_num + 1}")
                    
                    # Convert page to image
                    mat = fitz.Matrix(2.0, 2.0)  # High resolution
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    
                    # Use Mistral OCR
                    ocr_text = self._mistral_ocr(img_data)
                    
                    if ocr_text and len(ocr_text.strip()) > len(page_text.strip()):
                        page_text = ocr_text
                        logger.info(f"Mistral OCR improved text extraction for page {page_num + 1}")
                
                pages.append({
                    "page_number": page_num + 1,
                    "text": page_text,
                    "char_count": len(page_text)
                })
                full_text += page_text + "\n"
            
            doc.close()
            
            return {
                "full_text": full_text,
                "pages": pages,
                "metadata": {
                    "file_name": file_path.name,
                    "file_size": file_path.stat().st_size,
                    "page_count": len(pages),
                    "processing_method": "pdf_with_mistral_ocr"
                }
            }
            
        except Exception as e:
            raise Exception(f"PDF processing failed: {e}")
    
    def _process_word(self, file_path: Path) -> Dict[str, Any]:
        """Process Word document"""
        try:
            doc = Document(file_path)
            full_text = ""
            
            for paragraph in doc.paragraphs:
                full_text += paragraph.text + "\n"
            
            return {
                "full_text": full_text,
                "pages": [{
                    "page_number": 1,
                    "text": full_text,
                    "char_count": len(full_text)
                }],
                "metadata": {
                    "file_name": file_path.name,
                    "file_size": file_path.stat().st_size,
                    "page_count": 1,
                    "processing_method": "word_direct"
                }
            }
            
        except Exception as e:
            raise Exception(f"Word document processing failed: {e}")
    
    def _process_text(self, file_path: Path) -> Dict[str, Any]:
        """Process text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                full_text = file.read()
            
            return {
                "full_text": full_text,
                "pages": [{
                    "page_number": 1,
                    "text": full_text,
                    "char_count": len(full_text)
                }],
                "metadata": {
                    "file_name": file_path.name,
                    "file_size": file_path.stat().st_size,
                    "page_count": 1,
                    "processing_method": "text_direct"
                }
            }
            
        except Exception as e:
            raise Exception(f"Text file processing failed: {e}")
    
    def _process_image(self, file_path: Path) -> Dict[str, Any]:
        """Process image file using Mistral OCR"""
        try:
            # Read image file
            with open(file_path, 'rb') as img_file:
                img_data = img_file.read()
            
            # Use Mistral OCR
            extracted_text = self._mistral_ocr(img_data)
            
            if not extracted_text:
                extracted_text = "لم يتم استخراج نص من الصورة"
            
            return {
                "full_text": extracted_text,
                "pages": [{
                    "page_number": 1,
                    "text": extracted_text,
                    "char_count": len(extracted_text)
                }],
                "metadata": {
                    "file_name": file_path.name,
                    "file_size": file_path.stat().st_size,
                    "page_count": 1,
                    "processing_method": "mistral_ocr"
                }
            }
            
        except Exception as e:
            raise Exception(f"Image processing failed: {e}")
    
    def _mistral_ocr(self, image_data: bytes) -> Optional[str]:
        """
        Extract text from image using Mistral AI vision model
        
        Args:
            image_data: Raw image data as bytes
            
        Returns:
            Extracted text or None if failed
        """
        if not self.mistral_api_key:
            logger.warning("Mistral API key not available. Skipping OCR.")
            return None
        
        try:
            # Convert image to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare the request
            headers = {
                "Authorization": f"Bearer {self.mistral_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "mistral-ocr-latest",  # Mistral's OCR model
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """يرجى استخراج كل النص الموجود في هذه الصورة. النص قد يكون باللغة العربية أو الإنجليزية أو كليهما.

يرجى:
1. استخراج النص كما هو دون تحليل أو تفسير
2. الحفاظ على تنسيق النص قدر الإمكان
3. إذا كان النص باللغة العربية، تأكد من الدقة في الاستخراج
4. إذا لم تجد نص في الصورة، قل "لا يوجد نص في الصورة"

النص المستخرج:"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.1  # Low temperature for accurate OCR
            }
            
            # Make the request
            response = requests.post(
                self.mistral_api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                extracted_text = result['choices'][0]['message']['content'].strip()
                
                # Clean up the response
                if "لا يوجد نص" in extracted_text or "no text" in extracted_text.lower():
                    return None
                
                logger.info("Mistral OCR completed successfully")
                return extracted_text
            else:
                logger.error(f"Mistral API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Mistral OCR failed: {e}")
            return None
    
    def extract_text_from_image_url(self, image_url: str) -> Optional[str]:
        """
        Extract text from image URL using Mistral AI
        
        Args:
            image_url: URL of the image
            
        Returns:
            Extracted text or None if failed
        """
        if not self.mistral_api_key:
            logger.warning("Mistral API key not available. Skipping OCR.")
            return None
        
        try:
            headers = {
                "Authorization": f"Bearer {self.mistral_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "mistral-ocr-latest",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """يرجى استخراج كل النص الموجود في هذه الصورة بدقة. النص قد يكون باللغة العربية أو الإنجليزية.

استخرج النص كما هو بدون تعديل أو تفسير:"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            }
                        ]
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.1
            }
            
            response = requests.post(
                self.mistral_api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                logger.error(f"Mistral API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Mistral OCR from URL failed: {e}")
            return None

# Example usage
if __name__ == "__main__":
    processor = MistralDocumentProcessor()
    
    # Test with a sample file if available
    test_file = "./test_document.pdf"
    if os.path.exists(test_file):
        result = processor.process_document(test_file)
        print(f"Extracted text length: {len(result['full_text'])}")
        print(f"Pages processed: {result['metadata']['page_count']}")
        print(f"Processing method: {result['metadata']['processing_method']}")
    else:
        print("No test file found")
