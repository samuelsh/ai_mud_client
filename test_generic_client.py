#!/usr/bin/env python3
"""
Test the generic MUD client with different configurations
"""

import asyncio
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generic_mud_client import GenericMUDClient

async def test_generic_client(config_file: str = "config_examples/generic_mud_config.json"):
    """Test the generic MUD client with a specific configuration"""
    print(f"🧪 Testing Generic MUD Client with {config_file}")
    print("=" * 50)
    
    # Create client with specified config
    client = GenericMUDClient(config_file=config_file)
    print("✅ Client created")
    
    # Connect
    print("🔌 Connecting...")
    if await client.connect():
        print("✅ Connected successfully!")
        
        # Start auto mode for 30 seconds
        print("🤖 Starting auto mode for 30 seconds...")
        await client.run_auto_mode()
        
        # Disconnect
        await client.disconnect()
        print("✅ Disconnected")
    else:
        print("❌ Failed to connect")

async def main():
    """Main function to test different configurations"""
    configs = [
        "config_examples/sindome_config.json",
        "config_examples/aetolia_config.json",
        "config_examples/achaea_config.json"
    ]
    
    for config in configs:
        try:
            await test_generic_client(config)
            print(f"\n✅ Test completed for {config}")
        except Exception as e:
            print(f"\n❌ Test failed for {config}: {e}")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(main()) 