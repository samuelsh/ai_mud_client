#!/bin/bash
# Quick launch script for Achaea using generic client

echo "🚀 Launching Generic MUD Client for Achaea"
echo "==========================================="

source venv/bin/activate
python launcher_generic.py --config config_examples/achaea_config.json --mode auto --duration 60 