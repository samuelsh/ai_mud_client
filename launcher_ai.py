#!/usr/bin/env python3
"""
AI-Powered MUD Client Launcher
Launches the ChatGPT-powered MUD bot
"""

import asyncio
import argparse
import os
import sys
from ai_mud_client import AIPoweredMUDClient

def list_configs():
    """List available configuration files"""
    config_dir = "config_examples"
    if os.path.exists(config_dir):
        configs = [f for f in os.listdir(config_dir) if f.endswith('.json')]
        print("Available configurations:")
        for config in configs:
            print(f"  - {config}")
    else:
        print("No config_examples directory found")

def validate_config(config_file):
    """Validate configuration file exists"""
    if not os.path.exists(config_file):
        print(f"❌ Error: Configuration file '{config_file}' not found!")
        return False
    return True

async def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="AI-Powered MUD Client Launcher")
    parser.add_argument("--config", default="config_examples/sindome_config.json", 
                       help="Configuration file path")
    parser.add_argument("--duration", type=int, default=60, 
                       help="Duration to run in seconds")
    parser.add_argument("--list-configs", action="store_true", 
                       help="List available configurations")
    parser.add_argument("--interactive", action="store_true", 
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # List configurations if requested
    if args.list_configs:
        list_configs()
        return
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("1. Copy env.example to .env")
        print("2. Add your OpenAI API key to .env")
        print("3. Or set the environment variable: export OPENAI_API_KEY=your_key_here")
        return
    
    # Validate configuration
    if not validate_config(args.config):
        return
    
    print("🤖 AI-Powered MUD Client")
    print(f"📁 Config: {args.config}")
    print(f"⏱️  Duration: {args.duration} seconds")
    print(f"🧠 AI Model: {os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview')}")
    print("-" * 50)
    
    # Create and run client
    client = AIPoweredMUDClient(args.config)
    
    try:
        if await client.connect():
            print("✅ Connected successfully!")
            
            if args.interactive:
                print("🤖 Starting interactive AI mode...")
                await client.run_ai_mode(args.duration)
            else:
                print("🤖 Starting AI mode...")
                await client.run_ai_mode(args.duration)
            
            client.print_character_info()
        else:
            print("❌ Failed to connect to server")
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main()) 