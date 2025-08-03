#!/usr/bin/env python3
"""
Test script for AI-Powered MUD Client
Demonstrates the AI client functionality without connecting to a real server
"""

import asyncio
import os
import sys

# Mock the OpenAI client for testing
class MockOpenAI:
    def __init__(self, api_key=None):
        self.api_key = api_key or "mock_key"
    
    def chat(self):
        return self
    
    def completions(self):
        return self
    
    def create(self, **kwargs):
        # Return a mock response that matches OpenAI's structure
        class MockChoice:
            def __init__(self):
                class MockMessage:
                    def __init__(self):
                        self.content = 'look'
                self.message = MockMessage()
        
        class MockResponse:
            def __init__(self):
                self.choices = [MockChoice()]
        
        return MockResponse()

# Mock the openai module
sys.modules['openai'] = type('openai', (), {
    'OpenAI': MockOpenAI,
    'OpenAIError': Exception
})

from ai_mud_client import AIPoweredMUDClient

class MockAIPoweredMUDClient(AIPoweredMUDClient):
    """Mock AI client for testing without real server connection"""
    
    def __init__(self, config_file: str = "config_examples/sindome_config.json"):
        # Create a proper mock OpenAI client
        self.mock_openai_client = MockOpenAI()
        
        # Call parent constructor first
        super().__init__(config_file)
        
        # Override the OpenAI client with our mock
        self.openai_client = self.mock_openai_client
        
        self.mock_responses = [
            "Welcome to Sindome! Type 'connect guest' to enter.",
            "You are in a dark alley. Exits: north, south, east. You see a mysterious figure.",
            "The figure approaches you menacingly. You feel threatened.",
            "You successfully cast a spell! The figure retreats.",
            "You are in a busy marketplace. Many vendors call out to you.",
            "You find a treasure chest! It's locked.",
            "You successfully pick the lock. Inside you find 50 gold coins!",
            "You are in a peaceful garden. The air is fresh and calming.",
            "You feel rested and your health is restored.",
            "You encounter a group of bandits! They attack!",
            "You defeat the bandits! You gain 100 experience points.",
            "You are in a library. Books line the walls.",
            "You find a rare spellbook! Your magical knowledge increases.",
            "You are in a tavern. The bartender greets you warmly.",
            "You order a drink and hear interesting rumors from other patrons."
        ]
        self.response_index = 0
        self.connected = True
        self.connection_lost = False
    
    async def connect(self) -> bool:
        """Mock connection"""
        print("🔗 Mock connection established")
        return True
    
    async def read_response(self) -> str:
        """Return mock server responses"""
        if self.response_index >= len(self.mock_responses):
            self.response_index = 0  # Loop back
        
        response = self.mock_responses[self.response_index]
        self.response_index += 1
        
        print(f"📡 Mock Server: {response}")
        print("-" * 50)
        
        return response
    
    async def send_command(self, command: str):
        """Mock command sending"""
        print(f"🤖 AI Bot: {command}")
        print(f"📝 Command logged: {command}")
        
        # Simulate some character state changes
        if "cast" in command.lower():
            self.player.mana -= 10
        elif "rest" in command.lower():
            self.player.health = min(100, self.player.health + 20)
        elif "experience" in command.lower():
            self.player.experience += 50
        elif "gold" in command.lower():
            self.player.gold += 25

async def test_ai_client():
    """Test the AI client with mock responses"""
    print("🧪 Testing AI-Powered MUD Client")
    print("=" * 50)
    
    print("⚠️  Using mock OpenAI client for testing.")
    print("   Set OPENAI_API_KEY to test with real ChatGPT.")
    
    # Create mock client
    client = MockAIPoweredMUDClient("config_examples/sindome_config.json")
    
    try:
        if await client.connect():
            print("✅ Mock connection successful!")
            print(f"🤖 AI Model: {client.ai_model}")
            print(f"🎮 Server: {client.server_name}")
            print("-" * 50)
            
            # Test AI decision making
            print("🧠 Testing AI decision making...")
            
            for i in range(5):  # Test 5 interactions
                print(f"\n--- Interaction {i+1} ---")
                
                # Read mock response
                response = await client.read_response()
                
                # Get AI decision
                ai_command = await client._get_ai_decision(response)
                
                if ai_command:
                    await client.send_command(ai_command)
                else:
                    print("❌ AI decision failed, using fallback")
                    await client.send_command("look")
                
                # Update character info
                client._learn_from_response(response)
                
                await asyncio.sleep(1)  # Brief pause
            
            # Show final character info
            client.print_character_info()
            
        else:
            print("❌ Mock connection failed")
    
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def test_ai_context_building():
    """Test AI context building"""
    print("\n🧪 Testing AI Context Building")
    print("=" * 50)
    
    client = MockAIPoweredMUDClient()
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
    
    context = client._build_ai_context(mock_response, "exploring")
    
    print("📋 Generated AI Context:")
    print(context)
    print("\n✅ Context building test completed")

async def main():
    """Main test function"""
    print("🤖 AI-Powered MUD Client Test Suite")
    print("=" * 50)
    
    # Test context building
    test_ai_context_building()
    
    # Test AI client
    await test_ai_client()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(main()) 