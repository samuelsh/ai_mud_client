#!/bin/bash
# Quick launch script for Aetolia using generic client

echo "🚀 Launching Generic MUD Client for Aetolia"
echo "============================================"

source venv/bin/activate
python launcher_generic.py --config config_examples/aetolia_config.json --mode auto --duration 60 