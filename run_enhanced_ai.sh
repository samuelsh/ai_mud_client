#!/bin/bash

# Enhanced AI MUD Bot Launcher
# Features: Better prompt engineering, stateful context, input filtering, safety features

echo "🤖 Enhanced AI MUD Bot Launcher"
echo "================================"
echo "Features:"
echo "✅ Better Prompt Engineering"
echo "✅ Stateful Context Management"
echo "✅ Input Filtering"
echo "✅ Safety & Loop Protection"
echo "✅ Model Swap Support"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup_ai.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if enhanced AI client exists
if [ ! -f "enhanced_ai_mud_client.py" ]; then
    echo "❌ Enhanced AI client not found: enhanced_ai_mud_client.py"
    exit 1
fi

echo "✅ Virtual environment activated"
echo "✅ Enhanced AI client loaded"
echo ""

# Show AI backend options
echo "Choose your AI backend:"
echo "1) OpenAI (Requires API key)"
echo "2) Ollama Local (Requires Ollama installation)"
echo "3) Mock (For testing, no API needed)"
echo ""

read -p "Enter your choice (1-3): " backend_choice

case $backend_choice in
    1)
        backend="openai"
        echo "🤖 Using OpenAI backend"
        
        # Check for OpenAI API key
        if [ -z "$OPENAI_API_KEY" ]; then
            echo "❌ OPENAI_API_KEY not set. Please set it in your .env file"
            echo "   Example: OPENAI_API_KEY=your_api_key_here"
            exit 1
        fi
        ;;
    2)
        backend="ollama"
        echo "🏠 Using Ollama local backend"
        
        # Check if Ollama is available
        if ! command -v ollama &> /dev/null; then
            echo "❌ Ollama not found. Please install it first:"
            echo "   curl -fsSL https://ollama.ai/install.sh | sh"
            echo "   ollama pull llama2:7b"
            exit 1
        fi
        ;;
    3)
        backend="mock"
        echo "🧪 Using Mock backend for testing"
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

# Show configuration options
echo ""
echo "Available configurations:"
echo "1) 3k.org (Enhanced AI config)"
echo "2) Cybersphere"
echo "3) Sindome"
echo "4) Custom config file"
echo ""

read -p "Enter your choice (1-4): " config_choice

case $config_choice in
    1)
        config_file="config_examples/enhanced_ai_config.json"
        echo "✅ Using 3k.org enhanced AI configuration"
        ;;
    2)
        config_file="config_examples/cybersphere_config.json"
        echo "✅ Using Cybersphere configuration"
        ;;
    3)
        config_file="config_examples/sindome_config.json"
        echo "✅ Using Sindome configuration"
        ;;
    4)
        read -p "Enter custom config file path: " custom_config
        if [ -f "$custom_config" ]; then
            config_file="$custom_config"
            echo "✅ Using custom configuration: $config_file"
        else
            echo "❌ Custom config file not found: $custom_config"
            exit 1
        fi
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

# Check if config file exists
if [ ! -f "$config_file" ]; then
    echo "❌ Configuration file not found: $config_file"
    exit 1
fi

# Get duration
read -p "Enter duration in seconds (default: 60): " duration
duration=${duration:-60}

echo ""
echo "🚀 Starting Enhanced AI MUD Bot..."
echo "Backend: $backend"
echo "Config: $config_file"
echo "Duration: $duration seconds"
echo ""

# Run the enhanced AI client
python enhanced_ai_mud_client.py \
    --config "$config_file" \
    --backend "$backend" \
    --duration "$duration"

echo ""
echo "✅ Enhanced AI bot session completed!"
echo ""
echo "📊 Check the following files for details:"
echo "   - enhanced_ai_mud.log (detailed logs)"
echo "   - game_state.db (SQLite database with game state)"
echo "   - mud_bot.log (general bot logs)" 