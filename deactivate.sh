#!/bin/bash
# Deactivation script for MUD Bot virtual environment

if [ "$VIRTUAL_ENV" ]; then
    echo "🎮 Deactivating MUD Bot virtual environment..."
    deactivate
    echo "✅ Virtual environment deactivated!"
    echo ""
    echo "To reactivate later:"
    echo "  source venv/bin/activate"
    echo "  # or"
    echo "  source activate.sh"
else
    echo "ℹ️  No virtual environment is currently active."
fi 