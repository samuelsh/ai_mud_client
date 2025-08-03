#!/bin/bash

# Cybersphere MUD Bot Launcher
# www.cybersphere.net:7777

echo "🤖 Cybersphere MUD Bot Launcher"
echo "================================"
echo "Server: www.cybersphere.net:7777"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup_ai.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if config file exists
if [ ! -f "config_examples/cybersphere_config.json" ]; then
    echo "❌ Cybersphere config not found: config_examples/cybersphere_config.json"
    exit 1
fi

echo "✅ Virtual environment activated"
echo "✅ Configuration loaded: config_examples/cybersphere_config.json"
echo ""

# Show options
echo "Choose your bot type:"
echo "1) Enhanced Rule-Based (Free, Fast, Recommended)"
echo "2) AI-Powered with OpenAI (Requires API key)"
echo "3) Ollama Local AI (Requires Ollama installation)"
echo ""

read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo "🚀 Starting Enhanced Rule-Based Bot..."
        echo "Strategy options: explorer, combat_focused, farmer, social, adventurer"
        read -p "Enter strategy (default: explorer): " strategy
        strategy=${strategy:-explorer}
        
        read -p "Enter duration in seconds (default: 60): " duration
        duration=${duration:-60}
        
        python enhanced_generic_mud_client.py \
            --config config_examples/cybersphere_config.json \
            --strategy $strategy \
            --duration $duration
        ;;
    2)
        echo "🤖 Starting AI-Powered Bot..."
        
        # Check for OpenAI API key
        if [ -z "$OPENAI_API_KEY" ]; then
            echo "❌ OPENAI_API_KEY not set. Please set it in your .env file"
            echo "   Example: OPENAI_API_KEY=your_api_key_here"
            exit 1
        fi
        
        read -p "Enter duration in seconds (default: 60): " duration
        duration=${duration:-60}
        
        python ai_mud_client.py \
            --config config_examples/cybersphere_config.json \
            --duration $duration
        ;;
    3)
        echo "🏠 Starting Ollama Local AI Bot..."
        
        # Check if Ollama is available
        if ! command -v ollama &> /dev/null; then
            echo "❌ Ollama not found. Please install it first:"
            echo "   curl -fsSL https://ollama.ai/install.sh | sh"
            echo "   ollama pull llama2:7b"
            exit 1
        fi
        
        read -p "Enter Ollama model (default: llama2:7b): " model
        model=${model:-llama2:7b}
        
        read -p "Enter duration in seconds (default: 60): " duration
        duration=${duration:-60}
        
        export OLLAMA_MODEL=$model
        python ollama_mud_client.py \
            --config config_examples/cybersphere_config.json \
            --duration $duration
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "✅ Bot session completed!" 