#!/usr/bin/env python3
"""
Generic MUD Client Launcher
"""

import asyncio
import sys
import os
import argparse
import json

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generic_mud_client import GenericMUDClient

def list_configs():
    """List available configuration files"""
    config_dir = "config_examples"
    configs = []
    
    if os.path.exists(config_dir):
        for file in os.listdir(config_dir):
            if file.endswith('.json'):
                configs.append(os.path.join(config_dir, file))
    
    return configs

def load_config(config_file):
    """Load and validate configuration"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Validate required fields
        required_fields = ['server', 'bot', 'ai', 'game_specific', 'logging']
        for field in required_fields:
            if field not in config:
                print(f"❌ Error: Missing required field '{field}' in {config_file}")
                return None
        
        return config
    except FileNotFoundError:
        print(f"❌ Error: Configuration file {config_file} not found")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {config_file}: {e}")
        return None

async def run_client(config_file, mode="auto", duration=None):
    """Run the generic MUD client"""
    print(f"🚀 Launching Generic MUD Client")
    print(f"📁 Config: {config_file}")
    print(f"🎮 Mode: {mode}")
    if duration:
        print(f"⏱️  Duration: {duration} seconds")
    print("=" * 50)
    
    # Create client
    client = GenericMUDClient(config_file=config_file)
    
    # Connect
    print("🔌 Connecting...")
    if await client.connect():
        print("✅ Connected successfully!")
        
        if mode == "auto":
            if duration:
                print(f"🤖 Starting auto mode for {duration} seconds...")
                # Run for specified duration
                await asyncio.wait_for(client.run_auto_mode(), timeout=duration)
            else:
                print("🤖 Starting auto mode (press Ctrl+C to stop)...")
                await client.run_auto_mode()
        elif mode == "interactive":
            print("🎯 Starting interactive mode...")
            await run_interactive_mode(client)
        else:
            print(f"❌ Unknown mode: {mode}")
            return
        
        # Disconnect
        await client.disconnect()
        print("✅ Disconnected")
    else:
        print("❌ Failed to connect")

async def run_interactive_mode(client):
    """Run interactive mode"""
    print("\n🎮 Interactive Mode Commands:")
    print("  look, north, south, east, west, up, down")
    print("  inventory, equipment, status")
    print("  say <message>, tell <player> <message>")
    print("  quit - Exit the client")
    print("  help - Show this help")
    print("-" * 30)
    
    while True:
        try:
            command = input("🤖 Command: ").strip()
            
            if not command:
                continue
            
            if command.lower() == "quit":
                print("👋 Goodbye!")
                break
            elif command.lower() == "help":
                print("\n🎮 Interactive Mode Commands:")
                print("  look, north, south, east, west, up, down")
                print("  inventory, equipment, status")
                print("  say <message>, tell <player> <message>")
                print("  quit - Exit the client")
                print("  help - Show this help")
                print("-" * 30)
                continue
            
            # Send command
            await client.send_command(command)
            
            # Wait for response
            response = await client.read_response()
            if response:
                print(f"📡 Server: {response.strip()}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generic MUD Client Launcher")
    parser.add_argument(
        "--config", "-c",
        default="config_examples/generic_mud_config.json",
        help="Configuration file path"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["auto", "interactive"],
        default="auto",
        help="Client mode (auto or interactive)"
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        help="Duration in seconds for auto mode"
    )
    parser.add_argument(
        "--list-configs", "-l",
        action="store_true",
        help="List available configuration files"
    )
    
    args = parser.parse_args()
    
    # List configurations if requested
    if args.list_configs:
        configs = list_configs()
        if configs:
            print("📁 Available configurations:")
            for config in configs:
                print(f"  {config}")
        else:
            print("❌ No configuration files found in config_examples/")
        return
    
    # Validate configuration
    config = load_config(args.config)
    if not config:
        return
    
    # Show server info
    server_config = config.get("server", {})
    print(f"🎮 Server: {server_config.get('name', 'Unknown')}")
    print(f"🌐 Host: {server_config.get('host', 'localhost')}:{server_config.get('port', 23)}")
    print(f"🤖 Bot: {config.get('bot', {}).get('name', 'Unknown')}")
    print(f"🧠 AI Strategy: {config.get('ai', {}).get('strategy', 'explorer')}")
    print()
    
    # Run client
    try:
        asyncio.run(run_client(args.config, args.mode, args.duration))
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 