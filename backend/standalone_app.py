"""
Streamlit Application for Standalone Islamic Contract Analysis
Uses GPT models without external knowledge base
"""

import streamlit as st
import time
import os
import sys
from pathlib import Path

# Add backend directory to path
# backend_path = Path(__file__).parent.parent / "backend"
# sys.path.append(str(backend_path))

from standalone_analyzer import StandaloneContractAnalyzer, AnalysisResult
from mistral_document_processor import MistralDocumentProcessor

# Page configuration
st.set_page_config(
    page_title="تحليل العقود الإسلامية - GPT",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for RTL support and Arabic text
st.markdown("""
<style>
    .rtl {
        direction: rtl;
        text-align: right;
    }
    .arabic-text {
        font-family: 'Arial', 'Tahoma', sans-serif;
        direction: rtl;
        text-align: right;
    }
    .analysis-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        direction: rtl;
        text-align: right;
    }
    .compliance-good {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        direction: rtl;
    }
    .compliance-warning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        direction: rtl;
    }
    .compliance-bad {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        direction: rtl;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_analyzer():
    """Load the standalone analyzer (cached)"""
    try:
        analyzer = StandaloneContractAnalyzer()
        return analyzer
    except Exception as e:
        st.error(f"خطأ في تحميل المحلل: {str(e)}")
        return None

def display_analysis_result(result: AnalysisResult):
    """Display analysis results in a structured format"""
    
    # Compliance status with color coding
    if "مطابق" in result.compliance_status:
        st.markdown(f"""
        <div class="compliance-good">
            <h4>✅ حالة المطابقة الشرعية</h4>
            <p>{result.compliance_status}</p>
        </div>
        """, unsafe_allow_html=True)
    elif "يحتاج مراجعة" in result.compliance_status:
        st.markdown(f"""
        <div class="compliance-warning">
            <h4>⚠️ حالة المطابقة الشرعية</h4>
            <p>{result.compliance_status}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="compliance-bad">
            <h4>❌ حالة المطابقة الشرعية</h4>
            <p>{result.compliance_status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📋 الملخص", "🔍 التحليل المفصل", "💡 التوصيات", "📊 معلومات إضافية"])
    
    with tab1:
        st.markdown(f"""
        <div class="analysis-box">
            <h4>ملخص العقد</h4>
            <p class="arabic-text">{result.summary}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown(f"""
        <div class="analysis-box">
            <h4>التحليل الشرعي المفصل</h4>
            <p class="arabic-text">{result.detailed_analysis}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown(f"""
        <div class="analysis-box">
            <h4>التوصيات والحلول</h4>
            <p class="arabic-text">{result.recommendations}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("مستوى الثقة", f"{result.confidence:.1%}")
            st.metric("وقت المعالجة", f"{result.processing_time:.2f} ثانية")
        with col2:
            st.metric("النموذج المستخدم", result.model_used)
            if result.error:
                st.error(f"خطأ: {result.error}")

def main():
    """Main Streamlit application"""
    
    # Title and header
    st.title("🕌 تحليل العقود الإسلامية باستخدام GPT")
    st.markdown("### تحليل شرعي شامل للعقود والاتفاقيات بدون قاعدة معرفة خارجية")
    
    # Check for API keys (try Streamlit secrets first, then environment)
    openai_key = None
    mistral_key = None
    
    try:
        # Try Streamlit secrets first (for hosted deployment)
        if 'OPENAI_API_KEY' in st.secrets:
            openai_key = st.secrets['OPENAI_API_KEY']
        if 'MISTRAL_API_KEY' in st.secrets:
            mistral_key = st.secrets['MISTRAL_API_KEY']
    except:
        pass
    
    # Fallback to environment variables (for local development)
    if not openai_key:
        openai_key = os.getenv('OPENAI_API_KEY')
    if not mistral_key:
        mistral_key = os.getenv('MISTRAL_API_KEY')
    
    if not openai_key:
        st.error("""
        🔑 **مطلوب: مفتاح OpenAI API**
        
        لا يمكن تشغيل التطبيق بدون مفتاح OpenAI API. يرجى:
        
        **للنشر على Streamlit Cloud:**
        1. إضافة `OPENAI_API_KEY` في ملف `.streamlit/secrets.toml`
        
        **للتطوير المحلي:**
        1. الحصول على مفتاح من: https://platform.openai.com/api-keys
        2. تشغيل: `./setup_keys.sh` لإعداد المفاتيح
        3. أو تعيين متغير البيئة: `export OPENAI_API_KEY=your_key_here`
        """)
        st.stop()
    
    if not mistral_key:
        st.warning("""
        ⚠️ **مفتاح Mistral API غير موجود**
        
        سيعمل التطبيق بوظائف محدودة. للحصول على أفضل تجربة:
        
        **للنشر على Streamlit Cloud:**
        - إضافة `MISTRAL_API_KEY` في ملف `.streamlit/secrets.toml`
        
        **للتطوير المحلي:**
        - احصل على مفتاح من: https://console.mistral.ai/
        - شغل: `./setup_keys.sh` لإعداد المفاتيح
        """)
    
    # Load analyzer
    analyzer = load_analyzer()
    if not analyzer:
        st.stop()
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ إعدادات التحليل")
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "نوع التحليل",
            ["comprehensive", "riba", "gharar", "summary"],
            format_func=lambda x: {
                "comprehensive": "تحليل شامل",
                "riba": "تحليل الربا",
                "gharar": "تحليل الغرر",
                "summary": "ملخص فقط"
            }[x]
        )
        
        st.markdown("---")
        st.markdown("### 📝 معلومات التطبيق")
        st.info("""
        هذا التطبيق يستخدم نماذج GPT لتحليل العقود شرعياً بدون الحاجة لقاعدة معرفة خارجية.
        
        **المزايا:**
        - تحليل سريع ومباشر
        - لا يحتاج لقاعدة بيانات
        - يعتمد على المعرفة المدمجة في GPT
        - استخراج النصوص باستخدام Mistral AI OCR
        
        **ملاحظة:** للحصول على أفضل النتائج، استخدم نماذج GPT-4 أو أحدث.
        
        **متطلبات التشغيل:**
        - مفتاح OpenAI API (للتحليل)
        - مفتاح Mistral API (لاستخراج النصوص من الصور)
        """)
    
    # Main content area
    st.markdown("---")
    
    # Input methods
    input_method = st.radio(
        "طريقة إدخال العقد:",
        ["نص مباشر", "رفع ملف"],
        horizontal=True
    )
    
    contract_text = ""
    
    if input_method == "نص مباشر":
        st.markdown("### 📝 أدخل نص العقد")
        contract_text = st.text_area(
            "نص العقد:",
            height=200,
            placeholder="أدخل نص العقد أو الاتفاقية هنا...",
            help="يمكنك نسخ ولصق نص العقد مباشرة في هذا المربع"
        )
    
    else:  # File upload
        st.markdown("### 📁 رفع ملف العقد")
        uploaded_file = st.file_uploader(
            "اختر ملف العقد",
            type=['pdf', 'docx', 'doc', 'txt'],
            help="يدعم التطبيق ملفات PDF و Word و النصوص"
        )
        
        if uploaded_file is not None:
            try:
                # Process the uploaded file
                processor = MistralDocumentProcessor()
                with st.spinner("جاري معالجة الملف باستخدام Mistral AI..."):
                    # Save uploaded file temporarily
                    temp_path = f"/tmp/{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Extract text
                    extracted_text = processor.process_document(temp_path)
                    contract_text = extracted_text["full_text"]
                    
                    # Clean up
                    os.remove(temp_path)
                
                st.success(f"تم استخراج النص بنجاح باستخدام {extracted_text['metadata']['processing_method']}!")
                st.info(f"طول النص: {len(contract_text)} حرف | عدد الصفحات: {extracted_text['metadata']['page_count']}")
                
                # Show preview
                if st.checkbox("عرض معاينة النص المستخرج"):
                    st.text_area("معاينة النص:", value=contract_text[:1000] + "..." if len(contract_text) > 1000 else contract_text, height=150, disabled=True)
                    
            except Exception as e:
                st.error(f"خطأ في معالجة الملف: {str(e)}")
    
    # Analysis section
    if contract_text.strip():
        st.markdown("---")
        
        # Quick stats about the contract
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("عدد الأحرف", len(contract_text))
        with col2:
            st.metric("عدد الكلمات", len(contract_text.split()))
        with col3:
            st.metric("عدد الأسطر", len(contract_text.split('\n')))
        
        # Analyze button
        if st.button("🔍 تحليل العقد شرعياً", type="primary", use_container_width=True):
            
            with st.spinner("جاري تحليل العقد... قد يستغرق هذا بضع دقائق"):
                try:
                    # Perform analysis
                    result = analyzer.analyze_contract(contract_text, analysis_type)
                    
                    # Display results
                    st.markdown("## 📊 نتائج التحليل الشرعي")
                    display_analysis_result(result)
                    
                    # Download results option
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Prepare report text
                        report_text = f"""
تقرير التحليل الشرعي للعقد
==============================

حالة المطابقة: {result.compliance_status}
مستوى الثقة: {result.confidence:.1%}
النموذج المستخدم: {result.model_used}
وقت المعالجة: {result.processing_time:.2f} ثانية

الملخص:
{result.summary}

التحليل المفصل:
{result.detailed_analysis}

التوصيات:
{result.recommendations}

تم إنشاء هذا التقرير بواسطة نظام تحليل العقود الإسلامية
                        """
                        
                        st.download_button(
                            label="📥 تحميل التقرير كملف نصي",
                            data=report_text,
                            file_name=f"تحليل_شرعي_{int(time.time())}.txt",
                            mime="text/plain"
                        )
                    
                    with col2:
                        if st.button("📋 نسخ النتائج"):
                            st.code(report_text, language=None)
                
                except Exception as e:
                    st.error(f"حدث خطأ أثناء التحليل: {str(e)}")
                    st.info("يرجى المحاولة مرة أخرى أو التحقق من إعدادات الاتصال بـ OpenAI")
    
    else:
        st.info("يرجى إدخال نص العقد أو رفع ملف للبدء في التحليل")
    
    # Quick Query Section
    st.markdown("---")
    st.markdown("## ❓ استفسار سريع")
    
    with st.expander("طرح سؤال فقهي سريع"):
        question = st.text_input("سؤالك:", placeholder="مثال: ما حكم شرط الضمان في عقد البيع؟")
        context = st.text_input("السياق (اختياري):", placeholder="معلومات إضافية تساعد في فهم السؤال")
        
        if st.button("📝 احصل على الإجابة"):
            if question.strip():
                with st.spinner("جاري البحث عن الإجابة..."):
                    try:
                        response = analyzer.quick_query(question, context)
                        st.markdown(f"""
                        <div class="analysis-box">
                            <h4>الإجابة الشرعية</h4>
                            <p class="arabic-text">{response.text}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.caption(f"وقت المعالجة: {response.processing_time:.2f} ثانية | النموذج: {response.model_used}")
                        
                    except Exception as e:
                        st.error(f"خطأ في الاستفسار: {str(e)}")
            else:
                st.warning("يرجى إدخال سؤال أولاً")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🕌 نظام تحليل العقود الإسلامية | تطوير فريق التقنية الشرعية</p>
        <p>⚠️ ملاحظة: هذا النظام مساعد فقط ولا يغني عن استشارة أهل الاختصاص</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
