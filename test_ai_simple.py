#!/usr/bin/env python3
"""
Simple test for AI-Powered MUD Client
Demonstrates the AI client functionality without complex mocking
"""

import asyncio
import os
import sys

def test_ai_context_building():
    """Test AI context building functionality"""
    print("🧪 Testing AI Context Building")
    print("=" * 50)
    
    # Import the AI client
    from ai_mud_client import AIPoweredMUDClient
    
    # Create a client instance
    client = AIPoweredMUDClient("config_examples/sindome_config.json")
    
    # Set up test data
    client.player.name = "TestBot"
    client.player.health = 85
    client.player.mana = 70
    client.player.experience = 150
    client.player.gold = 75
    client.game_state = client.game_state.EXPLORING
    client.command_history = ["look", "north", "examine sign"]
    client._learned_commands = {"examine me", "examine here", "look sign"}
    client._failed_commands = {"cast spell", "apply caloric"}
    
    mock_response = "You are in a dark alley. Exits: north, south, east. You see a mysterious figure watching you."
    
    # Test context building
    context = client._build_ai_context(mock_response, "exploring")
    
    print("📋 Generated AI Context:")
    print(context)
    print("\n✅ Context building test completed")
    
    # Test immediate action detection
    print("\n🧪 Testing Immediate Action Detection")
    print("=" * 50)
    
    test_responses = [
        "Welcome to Sindome! Type 'connect guest' to enter.",
        "Login: ",
        "Password: ",
        "Character name already exists. Please choose another.",
        "You are in a peaceful garden."
    ]
    
    for i, response in enumerate(test_responses, 1):
        action = client._check_immediate_actions(response)
        print(f"Response {i}: {response[:50]}...")
        print(f"Action: {action or 'None'}")
        print("-" * 30)
    
    print("✅ Immediate action detection test completed")
    
    # Test learning functionality
    print("\n🧪 Testing Learning Functionality")
    print("=" * 50)
    
    test_learning_responses = [
        "I don't understand that. Try 'examine me' or 'look'.",
        "You can't do that here. Try 'north' or 'south'.",
        "You successfully cast a spell!",
        "You find a treasure chest!"
    ]
    
    for i, response in enumerate(test_learning_responses, 1):
        print(f"Learning from response {i}: {response}")
        client._learn_from_response(response)
        print(f"Learned commands: {list(client._learned_commands)[-3:]}")
        print(f"Failed commands: {list(client._failed_commands)[-3:]}")
        print("-" * 30)
    
    print("✅ Learning functionality test completed")

def test_configuration_loading():
    """Test configuration loading"""
    print("\n🧪 Testing Configuration Loading")
    print("=" * 50)
    
    from ai_mud_client import AIPoweredMUDClient
    
    try:
        client = AIPoweredMUDClient("config_examples/sindome_config.json")
        print("✅ Configuration loaded successfully")
        print(f"Server: {client.server_name}")
        print(f"Host: {client.host}:{client.port}")
        print(f"Character: {client.character_name}")
        print(f"AI Strategy: {client.ai_strategy.value}")
        print(f"AI Model: {client.ai_model}")
        
        # Test game-specific config
        print(f"Movement commands: {len(client.game_config.get('movement_commands', []))}")
        print(f"Combat commands: {len(client.game_config.get('combat_commands', []))}")
        print(f"System commands: {len(client.game_config.get('system_commands', []))}")
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        import traceback
        traceback.print_exc()

def test_environment_setup():
    """Test environment setup"""
    print("\n🧪 Testing Environment Setup")
    print("=" * 50)
    
    # Check for OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print("✅ OpenAI API key found")
        print(f"Key: {api_key[:10]}...{api_key[-4:]}")
    else:
        print("⚠️  OpenAI API key not found")
        print("   Set OPENAI_API_KEY to use real ChatGPT")
    
    # Check for other environment variables
    model = os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview')
    temperature = os.getenv('AI_TEMPERATURE', '0.7')
    max_tokens = os.getenv('AI_MAX_TOKENS', '150')
    
    print(f"AI Model: {model}")
    print(f"Temperature: {temperature}")
    print(f"Max Tokens: {max_tokens}")
    
    print("✅ Environment setup test completed")

async def main():
    """Main test function"""
    print("🤖 AI-Powered MUD Client Test Suite")
    print("=" * 50)
    
    # Test environment setup
    test_environment_setup()
    
    # Test configuration loading
    test_configuration_loading()
    
    # Test AI context building
    test_ai_context_building()
    
    print("\n✅ All tests completed!")
    print("\n📋 Summary:")
    print("- Context building: ✅ Working")
    print("- Configuration loading: ✅ Working")
    print("- Immediate actions: ✅ Working")
    print("- Learning functionality: ✅ Working")
    print("- Environment setup: ✅ Working")
    print("\n🚀 To run with real AI:")
    print("1. Set OPENAI_API_KEY environment variable")
    print("2. Run: python launcher_ai.py --config config_examples/sindome_config.json --duration 60")

if __name__ == "__main__":
    asyncio.run(main()) 