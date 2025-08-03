#!/bin/bash
# Quick launch script for Sindome using generic client

echo "🚀 Launching Generic MUD Client for Sindome"
echo "============================================="

source venv/bin/activate
python launcher_generic.py --config config_examples/sindome_config.json --mode auto --duration 60 