#!/bin/bash
# Quick activation script for MUD Bot virtual environment

if [ -d "venv" ]; then
    echo "🎮 Activating MUD Bot virtual environment..."
    source venv/bin/activate
    echo "✅ Virtual environment activated!"
    echo ""
    echo "Available commands:"
    echo "  python launcher.py          # Run the bot"
    echo "  python test_connection.py   # Test server connection"
    echo "  python example_usage.py     # Run examples"
    echo "  deactivate                  # Deactivate virtual environment"
    echo ""
else
    echo "❌ Virtual environment not found!"
    echo "Run './setup.sh' first to create the virtual environment."
    exit 1
fi 