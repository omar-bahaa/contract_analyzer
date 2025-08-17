#!/bin/bash

# Standalone Islamic Contract Analyzer - Streamlit Launcher
# This script launches the GPT-based contract analyzer without external knowledge base

echo "🕌 بدء تشغيل محلل العقود الإسلامية المستقل"
echo "================================================"

# Check if required environment variables are set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  تحذير: متغير البيئة OPENAI_API_KEY غير محدد"
    echo "   يرجى تعيين مفتاح OpenAI API:"
    echo "   export OPENAI_API_KEY='your-api-key-here'"
    echo ""
fi

# Check if required packages are installed
echo "🔍 فحص المتطلبات..."

python3 -c "import streamlit" 2>/dev/null || {
    echo "❌ Streamlit غير مثبت. يرجى تشغيل:"
    echo "   pip install streamlit"
    exit 1
}

python3 -c "import openai" 2>/dev/null || {
    echo "❌ OpenAI library غير مثبت. يرجى تشغيل:"
    echo "   pip install openai"
    exit 1
}

echo "✅ جميع المتطلبات متوفرة"
echo ""

# Set default port if not specified
STREAMLIT_PORT=${STREAMLIT_PORT:-8502}

echo "🚀 بدء تشغيل التطبيق على المنفذ $STREAMLIT_PORT"
echo "🌐 ستتمكن من الوصول للتطبيق على: http://localhost:$STREAMLIT_PORT"
echo ""
echo "📝 للإيقاف: اضغط Ctrl+C"
echo "================================================"

# Launch Streamlit app
cd backend
streamlit run standalone_app.py \
    --server.port $STREAMLIT_PORT \
    --server.headless true \
    --server.fileWatcherType none \
    --browser.gatherUsageStats false
