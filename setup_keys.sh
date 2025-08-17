#!/bin/bash

# API Key Setup Script for Islamic Contract RAG System
# This script helps users set up their API keys securely

echo "🔑 Islamic Contract RAG System - API Key Setup"
echo "=============================================="
echo ""

# Check if .env file exists
if [ -f ".env" ]; then
    echo "📄 Found existing .env file."
    read -p "Do you want to update it? (y/n): " update_env
    if [ "$update_env" != "y" ] && [ "$update_env" != "Y" ]; then
        echo "Using existing .env file."
        exit 0
    fi
else
    echo "📄 Creating new .env file from template..."
    cp .env.example .env
fi

echo ""
echo "🔐 Please enter your API keys:"
echo "Note: These will be stored securely in your local .env file"
echo ""

# OpenAI API Key
echo "1. OpenAI API Key (required for GPT-based analysis)"
echo "   Get your key from: https://platform.openai.com/api-keys"
read -p "   Enter OpenAI API Key: " openai_key

if [ -n "$openai_key" ]; then
    # Update .env file
    if grep -q "OPENAI_API_KEY=" .env; then
        sed -i "s/OPENAI_API_KEY=.*/OPENAI_API_KEY=$openai_key/" .env
    else
        echo "OPENAI_API_KEY=$openai_key" >> .env
    fi
    echo "   ✅ OpenAI API key saved"
else
    echo "   ⚠️  No OpenAI API key provided. GPT features will not work."
fi

echo ""

# Mistral API Key
echo "2. Mistral AI API Key (optional, for enhanced OCR)"
echo "   Get your key from: https://console.mistral.ai/"
read -p "   Enter Mistral API Key (or press Enter to skip): " mistral_key

if [ -n "$mistral_key" ]; then
    # Update .env file
    if grep -q "MISTRAL_API_KEY=" .env; then
        sed -i "s/MISTRAL_API_KEY=.*/MISTRAL_API_KEY=$mistral_key/" .env
    else
        echo "MISTRAL_API_KEY=$mistral_key" >> .env
    fi
    echo "   ✅ Mistral API key saved"
else
    echo "   ⚠️  No Mistral API key provided. Basic OCR will be used."
fi

echo ""
echo "🎉 API key setup complete!"
echo ""
echo "Next steps:"
echo "1. Run: ./setup.sh to install dependencies"
echo "2. Run: ./start_system.sh for full system"
echo "3. Run: ./start_standalone.sh for GPT-only analysis"
echo ""
echo "Your API keys are stored securely in .env file (not tracked by git)"
echo "=============================================="
