#!/bin/bash

# Improved Async AI MUD Bot Launcher
# Features: Non-blocking AI, better async patterns, modular architecture

echo "🚀 Improved Async AI MUD Bot Launcher"
echo "====================================="
echo "Features:"
echo "✅ Non-blocking AI calls"
echo "✅ Better async patterns"
echo "✅ Modular architecture"
echo "✅ Command validation"
echo "✅ Connection management with backoff"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup_ai.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if improved async client exists
if [ ! -f "improved_async_client.py" ]; then
    echo "❌ Improved async client not found: improved_async_client.py"
    exit 1
fi

echo "✅ Virtual environment activated"
echo "✅ Improved async client loaded"
echo ""

# Show configuration options
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
echo "🚀 Starting Improved Async AI MUD Bot..."
echo "Config: $config_file"
echo "Duration: $duration seconds"
echo ""

# Run the improved async client
python improved_async_client.py \
    --config "$config_file" \
    --duration "$duration"

echo ""
echo "✅ Improved async bot session completed!"
echo ""
echo "📊 Check the following files for details:"
echo "   - improved_async_mud.log (detailed logs)"
echo "   - game_context.json (context persistence)"
echo "   - mud_bot.log (general bot logs)" 