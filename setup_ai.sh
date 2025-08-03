#!/bin/bash

echo "🤖 Setting up AI-Powered MUD Client..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp env.example .env
    echo "⚠️  Please edit .env and add your OpenAI API key!"
    echo "   You can get one from: https://platform.openai.com/api-keys"
else
    echo "✅ .env file already exists"
fi

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OpenAI API key not found in environment"
    echo "   Please add it to your .env file or set the environment variable"
fi

echo "✅ AI-Powered MUD Client setup complete!"
echo ""
echo "🚀 To run the AI client:"
echo "   python launcher_ai.py --config config_examples/sindome_config.json --duration 60"
echo ""
echo "📋 Available commands:"
echo "   python launcher_ai.py --list-configs    # List available configs"
echo "   python launcher_ai.py --help            # Show help" 