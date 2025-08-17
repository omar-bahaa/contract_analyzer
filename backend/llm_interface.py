import os
import logging
from typing import List, Dict, Any, Optional
import requests
import json
from dataclasses import dataclass
import time

from prompts import IslamicContractPrompts

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Structure for LLM response"""
    text: str
    confidence: float
    processing_time: float
    model_used: str
    error: Optional[str] = None

class ArabicLLMInterface:
    """
    Interface for Arabic LLM integration supporting multiple backends:
    - Ollama (local)
    - Hugging Face Transformers (local)
    - OpenAI (remote API)
    """
    
    def __init__(self, backend: str = "ollama", model_name: str = "falcon", **kwargs):
        self.backend = backend
        self.model_name = model_name
        
        # OpenAI specific parameters
        self.openai_api_key = kwargs.get('openai_api_key', os.getenv('OPENAI_API_KEY'))
        self.openai_base_url = kwargs.get('openai_base_url', os.getenv('OPENAI_BASE_URL'))
        
        self.setup_backend()
    
    def setup_backend(self):
        """Setup the selected LLM backend"""
        if self.backend == "ollama":
            self.setup_ollama()
        elif self.backend == "huggingface":
            self.setup_huggingface()
        elif self.backend == "openai":
            self.setup_openai()
        else:
            logger.error(f"Unsupported backend: {self.backend}")
    
    def setup_ollama(self):
        """Setup Ollama backend"""
        self.ollama_url = "http://localhost:11434"
        self.ollama_api_url = f"{self.ollama_url}/api/generate"
        
        # Check if Ollama is running
        try:
            response = requests.get(f"{self.ollama_url}/api/version", timeout=10)
            if response.status_code == 200:
                logger.info("Ollama server is running")
                
                # Check if the model exists
                models_response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
                if models_response.status_code == 200:
                    models = models_response.json().get('models', [])
                    model_names = [model['name'] for model in models]
                    if self.model_name not in model_names and f"{self.model_name}:latest" not in model_names:
                        logger.warning(f"Model {self.model_name} not found. Available models: {model_names}")
                        # Try to use the first available model
                        if model_names:
                            self.model_name = model_names[0]
                            logger.info(f"Using available model: {self.model_name}")
                        else:
                            logger.error("No models available in Ollama")
                    else:
                        logger.info(f"Model {self.model_name} is available")
            else:
                logger.warning("Ollama server is not responding properly")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Ollama server is not accessible: {e}. Make sure it's running.")
    
    def setup_huggingface(self):
        """Setup Hugging Face Transformers backend"""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # Determine device
            device = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU
            device_name = "GPU" if device == 0 else "CPU"
            
            # Use a free Arabic model
            model_options = [
                "tiiuae/falcon-7b-instruct",    # Optimized for Arabic
                "microsoft/DialoGPT-medium",  # Fallback general model
                "aubmindlab/bert-base-arabert",  # Arabic BERT (for understanding)
                "CAMeL-Lab/bert-base-arabic-camelbert-mix"  # Another Arabic option
            ]
            
            for model in model_options:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model)
                    self.model = AutoModelForCausalLM.from_pretrained(model)
                    
                    # Move model to GPU if available
                    if device == 0:
                        self.model = self.model.cuda()
                    
                    self.generator = pipeline(
                        "text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        max_length=512,
                        device=device
                    )
                    self.model_name = model
                    logger.info(f"Loaded HuggingFace model: {model} on {device_name}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load model {model}: {e}")
                    continue
            else:
                raise Exception("Failed to load any HuggingFace model")
                
        except ImportError:
            logger.error("transformers library not installed")
            raise
    
    def setup_openai(self):
        """Setup OpenAI backend"""
        try:
            import openai
            
            # Check API key
            if not self.openai_api_key:
                raise Exception("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass openai_api_key parameter.")
            
            # Initialize OpenAI client
            client_kwargs = {
                "api_key": self.openai_api_key
            }
            
            # Add base URL if provided (for custom endpoints)
            if self.openai_base_url:
                client_kwargs["base_url"] = self.openai_base_url
            
            self.openai_client = openai.OpenAI(**client_kwargs)
            
            # Test the connection
            try:
                # Simple test call to verify API key works
                test_response = self.openai_client.models.list()
                logger.info(f"OpenAI API connection successful. Available models: {len(test_response.data)}")
                logger.info(f"Using OpenAI model: {self.model_name}")
            except Exception as e:
                logger.warning(f"OpenAI API test failed: {e}")
                
        except ImportError:
            logger.error("openai library not installed. Run: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Error setting up OpenAI backend: {e}")
            raise
    
    def generate_response(self, 
                         prompt: str, 
                         max_tokens: int = 500,
                         temperature: float = 0.7) -> LLMResponse:
        """Generate response using the configured backend"""
        start_time = time.time()
        
        try:
            logger.info(f"Generating response using {self.backend} backend with model {self.model_name}")
            logger.debug(f"Prompt length: {len(prompt)} characters")
            
            if self.backend == "ollama":
                response = self._generate_ollama(prompt, max_tokens, temperature)
            elif self.backend == "huggingface":
                response = self._generate_huggingface(prompt, max_tokens, temperature)
            elif self.backend == "openai":
                response = self._generate_openai(prompt, max_tokens, temperature)
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
            
            processing_time = time.time() - start_time
            logger.info(f"Response generated successfully in {processing_time:.2f} seconds")
            
            return LLMResponse(
                text=response,
                confidence=0.8,  # Default confidence
                processing_time=processing_time,
                model_used=self.model_name
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error generating response after {processing_time:.2f}s: {e}")
            
            return LLMResponse(
                text=f"عذراً، حدث خطأ في معالجة الطلب: {str(e)}",
                confidence=0.0,
                processing_time=processing_time,
                model_used=self.model_name,
                error=str(e)
            )
    
    def _generate_ollama(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate response using Ollama"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "num_gpu": -1,  # Use all available GPUs
                    "num_thread": 8  # Optimize CPU threads if needed
                }
            }
            
            response = requests.post(
                self.ollama_api_url,
                json=payload,
                timeout=300  # Increased timeout to 5 minutes
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama connection error: {e}")
    
    def _generate_huggingface(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate response using Hugging Face Transformers"""
        try:
            # Generate response
            result = self.generator(
                prompt,
                max_length=len(prompt.split()) + max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = result[0]['generated_text']
            
            # Extract only the new generated part
            if generated_text.startswith(prompt):
                response = generated_text[len(prompt):].strip()
            else:
                response = generated_text.strip()
            
            return response
            
        except Exception as e:
            raise Exception(f"HuggingFace generation error: {e}")
    
    def _generate_openai(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate response using OpenAI"""
        try:
            # Create chat completion
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "أنت خبير في الفقه الإسلامي والقانون الشرعي. أجب باللغة العربية بشكل واضح ومفصل."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise Exception(f"OpenAI generation error: {e}")

class IslamicContractAnalyzer:
    """
    Main class for analyzing contracts using Arabic LLM with Islamic knowledge
    """
    
    def __init__(self, llm_interface: ArabicLLMInterface):
        self.llm = llm_interface
        self.prompts = IslamicContractPrompts()
        self.setup_prompts()
    
    def setup_prompts(self):
        """Setup Arabic prompts for Islamic contract analysis"""
        self.system_prompt = self.prompts.get_system_prompt()
        self.analysis_prompt_template = self.prompts.get_analysis_prompt_template()
        self.recommendation_prompt_template = self.prompts.get_recommendation_prompt_template()
    
    def analyze_contract_with_llm(self, 
                                 analysis_results: Dict[str, Any], 
                                 contract_text: str) -> Dict[str, Any]:
        """Analyze contract using LLM with retrieved Islamic knowledge"""
        try:
            # Prepare references text
            references_text = self._format_references(analysis_results['supporting_references'])
            
            # Prepare contract clauses for analysis
            all_clauses = []
            all_clauses.extend([c['clause_text'] for c in analysis_results['non_compliant_clauses']])
            all_clauses.extend([c['clause_text'] for c in analysis_results['questionable_clauses']])
            all_clauses.extend([c['clause_text'] for c in analysis_results['compliant_clauses'][:3]])  # Sample of compliant ones
            
            clauses_text = "\n\n".join([f"البند {i+1}: {clause}" for i, clause in enumerate(all_clauses[:10])])
            
            # Create analysis prompt
            analysis_prompt = f"{self.system_prompt}\n\n{self.analysis_prompt_template.format(references=references_text, contract_clauses=clauses_text)}"
            
            # Generate analysis
            analysis_response = self.llm.generate_response(
                analysis_prompt,
                max_tokens=800,
                temperature=0.7
            )
            
            # Generate specific recommendations
            recommendations_response = self._generate_recommendations(analysis_results)
            
            # Combine results
            llm_analysis = {
                'detailed_analysis': analysis_response.text,
                'analysis_confidence': analysis_response.confidence,
                'detailed_recommendations': recommendations_response.text,
                'recommendations_confidence': recommendations_response.confidence,
                'processing_time': analysis_response.processing_time + recommendations_response.processing_time,
                'model_used': analysis_response.model_used
            }
            
            # Merge with original analysis results
            final_results = {**analysis_results, **llm_analysis}
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return {
                **analysis_results,
                'detailed_analysis': f"حدث خطأ في التحليل المفصل: {str(e)}",
                'detailed_recommendations': "لم يتم توليد توصيات مفصلة بسبب خطأ تقني.",
                'analysis_confidence': 0.0,
                'recommendations_confidence': 0.0,
                'error': str(e)
            }
    
    def _format_references(self, references: List[Dict]) -> str:
        """Format references for LLM prompt"""
        if not references:
            return "لا توجد مراجع مسترجعة."
        
        formatted_refs = []
        seen_texts = set()
        
        for i, ref in enumerate(references[:5]):  # Limit to top 5 references
            ref_text = ref['text']
            if ref_text not in seen_texts:  # Avoid duplicates
                metadata = ref.get('metadata', {})
                source = metadata.get('source', 'مصدر غير محدد')
                doc_type = metadata.get('document_type', 'نوع غير محدد')
                
                formatted_ref = f"المرجع {i+1}:\nالمصدر: {source}\nالنوع: {doc_type}\nالنص: {ref_text}\n"
                formatted_refs.append(formatted_ref)
                seen_texts.add(ref_text)
        
        return "\n---\n".join(formatted_refs)
    
    def _generate_recommendations(self, analysis_results: Dict) -> LLMResponse:
        """Generate detailed recommendations using LLM"""
        try:
            # Prepare non-compliant clauses
            non_compliant_text = "\n".join([
                f"- {clause['clause_text'][:200]}... (السبب: {clause['reasoning']})"
                for clause in analysis_results['non_compliant_clauses'][:5]
            ])
            
            # Prepare questionable clauses
            questionable_text = "\n".join([
                f"- {clause['clause_text'][:200]}... (السبب: {clause['reasoning']})"
                for clause in analysis_results['questionable_clauses'][:5]
            ])
            
            # Create recommendations prompt
            recommendations_prompt = f"{self.system_prompt}\n\n{self.recommendation_prompt_template.format(non_compliant_clauses=non_compliant_text, questionable_clauses=questionable_text)}"
            
            # Generate recommendations
            return self.llm.generate_response(
                recommendations_prompt,
                max_tokens=600,
                temperature=0.6
            )
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return LLMResponse(
                text=f"حدث خطأ في توليد التوصيات: {str(e)}",
                confidence=0.0,
                processing_time=0.0,
                model_used=self.llm.model_name,
                error=str(e)
            )
    
    def quick_query(self, question: str, context: str = "") -> LLMResponse:
        """Quick query for specific Islamic rulings"""
        try:
            prompt = f"{self.system_prompt}\n\n{self.prompts.get_quick_query_prompt().format(question=question, context=context)}"
            
            return self.llm.generate_response(
                prompt,
                max_tokens=400,
                temperature=0.6
            )
            
        except Exception as e:
            logger.error(f"Error in quick query: {e}")
            return LLMResponse(
                text=f"حدث خطأ في معالجة السؤال: {str(e)}",
                confidence=0.0,
                processing_time=0.0,
                model_used=self.llm.model_name,
                error=str(e)
            )

# Example usage and testing
if __name__ == "__main__":
    # Test with Ollama (if available)
    try:
        llm_interface = ArabicLLMInterface(backend="ollama", model_name="aya")
        analyzer = IslamicContractAnalyzer(llm_interface)
        
        # Test quick query
        response = analyzer.quick_query("ما حكم الربا في الإسلام؟")
        print(f"Response: {response.text}")
        print(f"Confidence: {response.confidence}")
        print(f"Processing time: {response.processing_time:.2f}s")
        
    except Exception as e:
        print(f"Ollama test failed: {e}")
        
        # Fallback to HuggingFace
        try:
            llm_interface = ArabicLLMInterface(backend="huggingface")
            analyzer = IslamicContractAnalyzer(llm_interface)
            print("Using HuggingFace backend as fallback")
            
        except Exception as e2:
            print(f"HuggingFace test also failed: {e2}")
            print("No LLM backend available for testing")
