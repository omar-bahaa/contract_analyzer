"""
Standalone Islamic Contract Analyzer using GPT
Analyzes contracts using only GPT knowledge without external knowledge base
"""

import os
import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
import yaml

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

from llm_interface import ArabicLLMInterface, LLMResponse
from standalone_prompts import StandalonePromptBuilder, StandaloneContractPrompts

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Structure for analysis results"""
    summary: str
    detailed_analysis: str
    compliance_status: str
    recommendations: str
    confidence: float
    processing_time: float
    model_used: str
    error: Optional[str] = None

class StandaloneContractAnalyzer:
    """
    Standalone contract analyzer using GPT without external knowledge base
    """
    
    def __init__(self, config_path: str = "./config.yaml"):
        """Initialize the analyzer with configuration"""
        self.config = self._load_config(config_path)
        self.prompt_builder = StandalonePromptBuilder()
        self.prompts = StandaloneContractPrompts()
        self.llm = self._setup_llm()
    
    def _get_api_key(self, key_name: str) -> Optional[str]:
        """
        Get API key from Streamlit secrets (for hosted apps) or environment variables (for local)
        
        Args:
            key_name: Name of the API key (e.g., 'OPENAI_API_KEY', 'MISTRAL_API_KEY')
        
        Returns:
            API key string or None if not found
        """
        # First try Streamlit secrets (for hosted deployment)
        if HAS_STREAMLIT:
            try:
                import streamlit as st
                if hasattr(st, 'secrets') and key_name in st.secrets:
                    return st.secrets[key_name]
            except Exception as e:
                logger.debug(f"Could not read {key_name} from Streamlit secrets: {e}")
        
        # Fallback to environment variables (for local development)
        return os.getenv(key_name)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            # Return default config for OpenAI
            return {
                'llm': {
                    'backend': 'openai',
                    'openai_model': 'gpt-4o',
                    'temperature': 0.7,
                    'openai_api_key': os.getenv('OPENAI_API_KEY')
                }
            }
    
    def _setup_llm(self) -> ArabicLLMInterface:
        """Setup LLM interface"""
        llm_config = self.config.get('llm', {})
        
        # Use OpenAI as default for standalone analysis
        backend = llm_config.get('backend', 'openai')
        model_name = llm_config.get('openai_model', 'gpt-4o')
        
        # Get API key from Streamlit secrets or environment variables
        openai_api_key = self._get_api_key('OPENAI_API_KEY')
        
        # OpenAI specific parameters
        openai_kwargs = {
            'openai_api_key': openai_api_key,
            'openai_base_url': llm_config.get('openai_base_url')
        }
        
        llm = ArabicLLMInterface(
            backend=backend,
            model_name=model_name,
            **openai_kwargs
        )
        
        logger.info(f"LLM interface initialized with {backend} backend")
        return llm
    
    def analyze_contract(self, contract_text: str, analysis_type: str = "comprehensive") -> AnalysisResult:
        """
        Analyze contract using GPT without external knowledge base
        
        Args:
            contract_text: The contract text to analyze
            analysis_type: Type of analysis ('comprehensive', 'riba', 'gharar', 'summary')
        
        Returns:
            AnalysisResult with the analysis findings
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting {analysis_type} analysis of contract")
            
            # Generate contract summary first
            summary = self._generate_summary(contract_text)
            
            # Generate detailed analysis based on type
            if analysis_type == "comprehensive":
                detailed_analysis = self._comprehensive_analysis(contract_text)
            elif analysis_type == "riba":
                detailed_analysis = self._riba_analysis(contract_text)
            elif analysis_type == "gharar":
                detailed_analysis = self._gharar_analysis(contract_text)
            elif analysis_type == "summary":
                detailed_analysis = summary.text
            else:
                detailed_analysis = self._comprehensive_analysis(contract_text)
            
            # Determine compliance status
            compliance_status = self._determine_compliance(detailed_analysis)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(contract_text, detailed_analysis)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Analysis completed in {processing_time:.2f} seconds")
            
            return AnalysisResult(
                summary=summary.text,
                detailed_analysis=detailed_analysis.text if hasattr(detailed_analysis, 'text') else detailed_analysis,
                compliance_status=compliance_status,
                recommendations=recommendations.text,
                confidence=0.85,  # High confidence for GPT-4
                processing_time=processing_time,
                model_used=self.llm.model_name
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error in contract analysis: {e}")
            
            return AnalysisResult(
                summary=f"حدث خطأ في تلخيص العقد: {str(e)}",
                detailed_analysis=f"حدث خطأ في التحليل المفصل: {str(e)}",
                compliance_status="غير محدد بسبب خطأ تقني",
                recommendations=f"لم يتم توليد توصيات بسبب خطأ تقني: {str(e)}",
                confidence=0.0,
                processing_time=processing_time,
                model_used=self.llm.model_name,
                error=str(e)
            )
    
    def _generate_summary(self, contract_text: str) -> LLMResponse:
        """Generate contract summary"""
        prompt = self.prompt_builder.build_summary_prompt(contract_text)
        
        return self.llm.generate_response(
            prompt,
            max_tokens=600,
            temperature=0.5  # Lower temperature for more factual summary
        )
    
    def _comprehensive_analysis(self, contract_text: str) -> LLMResponse:
        """Perform comprehensive contract analysis"""
        prompt = self.prompt_builder.build_analysis_prompt(contract_text)
        
        return self.llm.generate_response(
            prompt,
            max_tokens=1200,
            temperature=0.6
        )
    
    def _riba_analysis(self, contract_text: str) -> LLMResponse:
        """Analyze contract for riba (interest) issues"""
        prompt = self.prompt_builder.build_specialized_prompt(contract_text, "riba")
        
        return self.llm.generate_response(
            prompt,
            max_tokens=800,
            temperature=0.5
        )
    
    def _gharar_analysis(self, contract_text: str) -> LLMResponse:
        """Analyze contract for gharar (excessive uncertainty) issues"""
        prompt = self.prompt_builder.build_specialized_prompt(contract_text, "gharar")
        
        return self.llm.generate_response(
            prompt,
            max_tokens=800,
            temperature=0.5
        )
    
    def _determine_compliance(self, analysis_text: str) -> str:
        """Determine overall compliance status from analysis"""
        # Simple keyword-based determination
        # In a more sophisticated version, this could use another LLM call
        
        analysis_lower = analysis_text.lower() if isinstance(analysis_text, str) else ""
        
        # Check for non-compliance indicators
        non_compliant_keywords = ["محرم", "مخالف", "غير جائز", "ربا", "حرام", "غير مطابق"]
        questionable_keywords = ["مشكوك", "يحتاج مراجعة", "غير واضح", "مبهم"]
        compliant_keywords = ["مطابق", "جائز", "صحيح", "متوافق", "مقبول"]
        
        non_compliant_count = sum(1 for keyword in non_compliant_keywords if keyword in analysis_lower)
        questionable_count = sum(1 for keyword in questionable_keywords if keyword in analysis_lower)
        compliant_count = sum(1 for keyword in compliant_keywords if keyword in analysis_lower)
        
        if non_compliant_count > 0:
            return "غير مطابق للشريعة الإسلامية"
        elif questionable_count > compliant_count:
            return "يحتاج مراجعة شرعية"
        elif compliant_count > 0:
            return "مطابق للشريعة الإسلامية"
        else:
            return "يحتاج تقييم إضافي"
    
    def _generate_recommendations(self, contract_text: str, analysis: str) -> LLMResponse:
        """Generate specific recommendations"""
        system_prompt = self.prompts.get_system_prompt()
        
        prompt = f"""{system_prompt}

بناءً على التحليل التالي للعقد:

{analysis}

والعقد الأصلي:
{contract_text[:1000]}...

يرجى تقديم توصيات عملية ومحددة لتحسين هذا العقد وجعله أكثر توافقاً مع الشريعة الإسلامية. 

يجب أن تشمل التوصيات:
1. التعديلات المطلوبة للبنود المخالفة
2. إضافات مقترحة لتعزيز المطابقة الشرعية
3. بدائل شرعية للممارسات المشكوك فيها
4. خطوات عملية للتنفيذ

كن محدداً وعملياً في توصياتك."""

        return self.llm.generate_response(
            prompt,
            max_tokens=800,
            temperature=0.6
        )
    
    def quick_query(self, question: str, context: str = "") -> LLMResponse:
        """Quick query for specific Islamic rulings"""
        prompt = self.prompt_builder.build_quick_query_prompt(question, context)
        
        return self.llm.generate_response(
            prompt,
            max_tokens=500,
            temperature=0.6
        )
    
    def analyze_specific_clause(self, clause_text: str) -> LLMResponse:
        """Analyze a specific contract clause"""
        prompt = self.prompt_builder.build_clause_analysis_prompt(clause_text)
        
        return self.llm.generate_response(
            prompt,
            max_tokens=400,
            temperature=0.6
        )

# Example usage and testing
if __name__ == "__main__":
    # Test the standalone analyzer
    try:
        analyzer = StandaloneContractAnalyzer()
        
        # Test with a sample contract clause
        sample_clause = """
        يلتزم المقترض بدفع فائدة سنوية قدرها 12% على المبلغ المقترض، 
        وفي حالة التأخير في السداد يتم إضافة غرامة تأخير بنسبة 2% شهرياً
        """
        
        print("Testing standalone contract analyzer...")
        result = analyzer.analyze_contract(sample_clause, "riba")
        
        print(f"Summary: {result.summary[:200]}...")
        print(f"Compliance: {result.compliance_status}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Model used: {result.model_used}")
        
        if result.error:
            print(f"Error: {result.error}")
        
    except Exception as e:
        print(f"Test failed: {e}")
