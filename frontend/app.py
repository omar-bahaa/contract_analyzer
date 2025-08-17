import streamlit as st
import requests
import json
import time
from typing import Dict, Any, List
import os
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="نظام تحليل العقود الإسلامية",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Arabic support
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    
    .compliant {
        border-left-color: #10b981;
        background: #f0fdf4;
    }
    
    .non-compliant {
        border-left-color: #ef4444;
        background: #fef2f2;
    }
    
    .questionable {
        border-left-color: #f59e0b;
        background: #fffbeb;
    }
    
    .arabic-text {
        font-family: 'Arial', 'Tahoma', sans-serif;
        direction: rtl;
        text-align: right;
    }
    
    .clause-item {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .reference-item {
        background: #f8fafc;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        border-left: 3px solid #6366f1;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000"

# Helper functions
def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_islamic_document(file, document_type, source, author, topic):
    """Upload an Islamic document to the knowledge base"""
    try:
        files = {"file": file}
        data = {
            "document_type": document_type,
            "source": source,
            "author": author,
            "topic": topic
        }
        response = requests.post(f"{API_BASE_URL}/upload_islamic_document", files=files, data=data)
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def analyze_contract_file(file):
    """Analyze a contract file"""
    try:
        files = {"file": file}
        data = {"analysis_type": "full"}
        response = requests.post(f"{API_BASE_URL}/analyze_contract", files=files, data=data)
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def analyze_contract_text(text):
    """Analyze contract text"""
    try:
        data = {"contract_text": text, "analysis_type": "full"}
        response = requests.post(f"{API_BASE_URL}/analyze_contract_text", json=data)
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def query_knowledge_base(question, context=""):
    """Query the Islamic knowledge base"""
    try:
        data = {"question": question, "context": context}
        response = requests.post(f"{API_BASE_URL}/query", json=data)
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_knowledge_base_stats():
    """Get knowledge base statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        return response.json()
    except Exception as e:
        return {"total_documents": 0, "error": str(e)}

def display_clause_analysis(clauses, clause_type, icon, color_class):
    """Display clause analysis results"""
    if not clauses:
        return
    
    st.markdown(f"### {icon} {clause_type} ({len(clauses)} بند)")
    
    for i, clause in enumerate(clauses):
        with st.container():
            st.markdown(f"""
            <div class="clause-item {color_class}">
                <h4>البند {clause.get('clause_id', i+1)}</h4>
                <p class="arabic-text">{clause.get('clause_text', '')[:300]}...</p>
                <p><strong>السبب:</strong> {clause.get('reasoning', 'غير محدد')}</p>
                <p><strong>مستوى الثقة:</strong> {clause.get('confidence', 0):.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show references if available
            if clause.get('references'):
                with st.expander("المراجع المدعمة"):
                    for ref in clause['references'][:3]:
                        st.markdown(f"""
                        <div class="reference-item">
                            <p>{ref.get('text', '')[:200]}...</p>
                            <small>المصدر: {ref.get('metadata', {}).get('source', 'غير محدد')}</small>
                        </div>
                        """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🕌 نظام تحليل العقود الإسلامية</h1>
        <p>نظام ذكي لتحليل العقود والتأكد من مطابقتها للشريعة الإسلامية</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API status
    api_status = check_api_health()
    
    if not api_status:
        st.error("❌ الخادم غير متاح. تأكد من تشغيل الخادم الخلفي على المنفذ 8000")
        st.info("لتشغيل الخادم: `python backend/main.py`")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("📊 إحصائيات قاعدة المعرفة")
        
        # Get and display stats
        stats = get_knowledge_base_stats()
        st.metric("إجمالي الوثائق", stats.get('total_documents', 0))
        
        if 'document_types' in stats:
            st.subheader("أنواع الوثائق")
            for doc_type, count in stats['document_types'].items():
                st.text(f"{doc_type}: {count}")
        
        st.markdown("---")
        st.subheader("ℹ️ معلومات النظام")
        st.info("""
        **الميزات:**
        - تحليل العقود بالعربية
        - استخراج البنود تلقائياً
        - مقارنة مع الشريعة الإسلامية
        - توصيات للتعديل
        - دعم PDF و DOCX
        """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 تحليل العقود", "📚 إضافة وثائق إسلامية", "🔍 استعلام المعرفة", "ℹ️ حول النظام"])
    
    with tab1:
        st.header("📋 تحليل العقود للمطابقة الشرعية")
        
        # Analysis method selection
        analysis_method = st.radio(
            "اختر طريقة التحليل:",
            ["رفع ملف عقد", "إدخال نص العقد"],
            horizontal=True
        )
        
        if analysis_method == "رفع ملف عقد":
            st.subheader("رفع ملف العقد")
            uploaded_file = st.file_uploader(
                "اختر ملف العقد (PDF أو DOCX)",
                type=['pdf', 'docx', 'doc'],
                help="يدعم النظام ملفات PDF و Microsoft Word"
            )
            
            if uploaded_file is not None:
                if st.button("🔍 تحليل العقد", type="primary", use_container_width=True):
                    with st.spinner("جاري تحليل العقد..."):
                        # Reset file pointer
                        uploaded_file.seek(0)
                        
                        # Analyze contract
                        result = analyze_contract_file(uploaded_file)
                        
                        if result.get('success'):
                            st.success("✅ تم تحليل العقد بنجاح!")
                            
                            # Display metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown("""
                                <div class="metric-card compliant">
                                    <h3>✅ بنود متوافقة</h3>
                                    <h2>{}</h2>
                                </div>
                                """.format(len(result.get('compliant_clauses', []))), unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown("""
                                <div class="metric-card non-compliant">
                                    <h3>❌ بنود غير متوافقة</h3>
                                    <h2>{}</h2>
                                </div>
                                """.format(len(result.get('non_compliant_clauses', []))), unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown("""
                                <div class="metric-card questionable">
                                    <h3>⚠️ بنود تحتاج مراجعة</h3>
                                    <h2>{}</h2>
                                </div>
                                """.format(len(result.get('questionable_clauses', []))), unsafe_allow_html=True)
                            
                            # Detailed Analysis
                            if result.get('detailed_analysis'):
                                st.subheader("📝 التحليل المفصل")
                                st.markdown(f"<div class='arabic-text'>{result['detailed_analysis']}</div>", unsafe_allow_html=True)
                            
                            # Display clauses by category
                            st.subheader("📊 تفاصيل تحليل البنود")
                            
                            # Non-compliant clauses
                            display_clause_analysis(
                                result.get('non_compliant_clauses', []),
                                "بنود غير متوافقة مع الشريعة",
                                "❌",
                                "non-compliant"
                            )
                            
                            # Questionable clauses
                            display_clause_analysis(
                                result.get('questionable_clauses', []),
                                "بنود تحتاج مراجعة",
                                "⚠️",
                                "questionable"
                            )
                            
                            # Compliant clauses
                            display_clause_analysis(
                                result.get('compliant_clauses', []),
                                "بنود متوافقة مع الشريعة",
                                "✅",
                                "compliant"
                            )
                            
                            # Recommendations
                            if result.get('recommendations'):
                                st.subheader("💡 التوصيات")
                                for i, rec in enumerate(result['recommendations']):
                                    st.markdown(f"**{i+1}.** {rec}")
                            
                            # Detailed recommendations
                            if result.get('detailed_recommendations'):
                                st.subheader("📋 التوصيات المفصلة")
                                st.markdown(f"<div class='arabic-text'>{result['detailed_recommendations']}</div>", unsafe_allow_html=True)
                            
                            # Analysis metadata
                            with st.expander("📈 معلومات التحليل"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("مستوى الثقة", f"{result.get('confidence', 0):.2f}")
                                with col2:
                                    st.metric("وقت المعالجة", f"{result.get('processing_time', 0):.2f}s")
                                
                                st.text(f"معرف التحليل: {result.get('analysis_id', 'غير محدد')}")
                        
                        else:
                            st.error(f"❌ خطأ في تحليل العقد: {result.get('error', 'خطأ غير محدد')}")
        
        elif analysis_method == "إدخال نص العقد":
            st.subheader("إدخال نص العقد")
            contract_text = st.text_area(
                "أدخل نص العقد هنا:",
                height=300,
                help="أدخل نص العقد باللغة العربية"
            )
            
            if contract_text and st.button("🔍 تحليل النص", type="primary", use_container_width=True):
                with st.spinner("جاري تحليل النص..."):
                    result = analyze_contract_text(contract_text)
                    
                    if result.get('success'):
                        st.success("✅ تم تحليل النص بنجاح!")
                        
                        # Display same analysis interface as file upload
                        # (Same code as above for displaying results)
                        # ... (implementation similar to file analysis)
                    
                    else:
                        st.error(f"❌ خطأ في تحليل النص: {result.get('error', 'خطأ غير محدد')}")
    
    with tab2:
        st.header("📚 إضافة وثائق إسلامية لقاعدة المعرفة")
        st.info("أضف وثائق الفقه الإسلامي والأحكام الشرعية لتحسين دقة التحليل")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_doc = st.file_uploader(
                "اختر وثيقة إسلامية (PDF أو DOCX)",
                type=['pdf', 'docx', 'doc'],
                help="ارفع وثائق الفقه والأحكام الإسلامية"
            )
        
        with col2:
            document_type = st.selectbox(
                "نوع الوثيقة",
                ["فقه", "حديث", "قرآن", "فتوى", "أحكام", "أخرى"]
            )
            
            source = st.text_input("المصدر", placeholder="مثال: صحيح البخاري")
            author = st.text_input("المؤلف", placeholder="مثال: الإمام البخاري")
            topic = st.text_input("الموضوع", placeholder="مثال: أحكام البيع")
        
        if uploaded_doc is not None:
            if st.button("📤 إضافة إلى قاعدة المعرفة", type="primary", use_container_width=True):
                with st.spinner("جاري معالجة الوثيقة..."):
                    # Reset file pointer
                    uploaded_doc.seek(0)
                    
                    result = upload_islamic_document(
                        uploaded_doc, document_type, source, author, topic
                    )
                    
                    if result.get('success'):
                        st.success("✅ تم إضافة الوثيقة بنجاح!")
                        st.info(f"معرف الوثيقة: {result.get('document_id')}")
                        st.json(result.get('metadata', {}))
                    else:
                        st.error(f"❌ خطأ في إضافة الوثيقة: {result.get('error', 'خطأ غير محدد')}")
    
    with tab3:
        st.header("🔍 استعلام قاعدة المعرفة الإسلامية")
        st.info("اسأل أي سؤال حول الأحكام الشرعية والفقه الإسلامي")
        
        question = st.text_area(
            "أدخل سؤالك:",
            height=100,
            placeholder="مثال: ما حكم الربا في الإسلام؟"
        )
        
        context = st.text_area(
            "السياق (اختياري):",
            height=80,
            placeholder="أي معلومات إضافية تساعد في الإجابة"
        )
        
        if question and st.button("🔍 بحث", type="primary", use_container_width=True):
            with st.spinner("جاري البحث في قاعدة المعرفة..."):
                result = query_knowledge_base(question, context)
                
                if result.get('success'):
                    st.subheader("📝 الإجابة")
                    st.markdown(f"<div class='arabic-text'>{result.get('answer', 'لم يتم العثور على إجابة')}</div>", unsafe_allow_html=True)
                    
                    # Display references
                    if result.get('references'):
                        st.subheader("📚 المراجع المدعمة")
                        for i, ref in enumerate(result['references'][:5]):
                            with st.expander(f"مرجع {i+1} - {ref.get('metadata', {}).get('source', 'مصدر غير محدد')}"):
                                st.markdown(f"<div class='arabic-text'>{ref.get('text', '')}</div>", unsafe_allow_html=True)
                                
                                metadata = ref.get('metadata', {})
                                if metadata:
                                    st.json(metadata)
                    
                    # Query metadata
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("مستوى الثقة", f"{result.get('confidence', 0):.2f}")
                    with col2:
                        st.metric("وقت المعالجة", f"{result.get('processing_time', 0):.2f}s")
                
                else:
                    st.error(f"❌ خطأ في البحث: {result.get('error', 'خطأ غير محدد')}")
    
    with tab4:
        st.header("ℹ️ حول النظام")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 الهدف")
            st.write("""
            نظام ذكي لتحليل العقود والاتفاقيات للتأكد من مطابقتها للشريعة الإسلامية،
            مع تقديم توصيات للتعديل وضمان الامتثال للأحكام الشرعية.
            """)
            
            st.subheader("⚙️ التقنيات المستخدمة")
            st.write("""
            - **معالجة النصوص العربية**: CAMeL Tools, PyArabic
            - **استخراج النصوص**: PyMuPDF, python-docx, Tesseract OCR
            - **قاعدة البيانات المتجهة**: ChromaDB
            - **النماذج اللغوية**: Sentence Transformers, Ollama
            - **واجهة المستخدم**: Streamlit
            - **الخادم الخلفي**: FastAPI
            """)
        
        with col2:
            st.subheader("🔧 الميزات")
            st.write("""
            - **تحليل شامل**: فحص جميع بنود العقد
            - **دعم متعدد الصيغ**: PDF, DOCX, نص مباشر
            - **معالجة عربية متقدمة**: OCR ومعالجة النصوص
            - **قاعدة معرفة إسلامية**: وثائق الفقه والأحكام
            - **توصيات ذكية**: اقتراحات للتعديل
            - **واجهة سهلة**: تصميم بسيط ومفهوم
            """)
            
            st.subheader("📞 الدعم")
            st.write("""
            للدعم الفني أو الاستفسارات، يرجى التواصل مع فريق التطوير.
            
            **ملاحظة**: هذا نموذج أولي للاختبار والتطوير.
            """)
        
        st.markdown("---")
        st.markdown("**© 2024 نظام تحليل العقود الإسلامية - جميع الحقوق محفوظة**")

if __name__ == "__main__":
    main()
