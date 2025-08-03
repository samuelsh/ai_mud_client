#!/usr/bin/env python3
"""
AI-Powered MUD Client with ChatGPT Integration
Uses OpenAI's GPT models for intelligent game decision making
"""

import asyncio
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import telnetlib3
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('MUD_BOT_LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.getenv('MUD_BOT_LOG_FILE', 'mud_bot.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GameState(Enum):
    """Game state enumeration"""
    CONNECTING = "connecting"
    LOGIN = "login"
    CHARACTER_CREATION = "character_creation"
    PLAYING = "playing"
    COMBAT = "combat"
    INVENTORY = "inventory"
    SHOPPING = "shopping"
    EXPLORING = "exploring"
    RESTING = "resting"
    DEAD = "dead"

class AIStrategy(Enum):
    """AI strategy types"""
    EXPLORER = "explorer"
    COMBAT_FOCUSED = "combat_focused"
    FARMER = "farmer"
    SOCIAL = "social"
    ADVENTURER = "adventurer"

@dataclass
class Room:
    """Represents a game room"""
    name: str
    description: str
    exits: Dict[str, str]
    items: List[str]
    npcs: List[str]
    visited: bool = False
    last_visited: Optional[datetime] = None

@dataclass
class PlayerState:
    """Player state information"""
    name: str = ""
    level: int = 1
    health: int = 100
    max_health: int = 100
    mana: int = 100
    max_mana: int = 100
    experience: int = 0
    gold: int = 0
    location: str = ""
    inventory: List[str] = None
    equipment: Dict[str, str] = None
    skills: Dict[str, int] = None
    spells: List[str] = None
    
    def __post_init__(self):
        if self.inventory is None:
            self.inventory = []
        if self.equipment is None:
            self.equipment = {}
        if self.skills is None:
            self.skills = {}
        if self.spells is None:
            self.spells = []

class AIPoweredMUDClient:
    """AI-powered MUD client using ChatGPT for intelligent decision making"""
    
    def __init__(self, config_file: str = "config_examples/generic_mud_config.json"):
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize OpenAI client
        try:
            self.openai_client = openai.OpenAI(
                api_key=os.getenv('OPENAI_API_KEY')
            )
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}")
            self.openai_client = None
        
        # AI configuration
        self.ai_model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        self.ai_temperature = float(os.getenv('AI_TEMPERATURE', '0.7'))
        self.ai_max_tokens = int(os.getenv('AI_MAX_TOKENS', '150'))
        
        # Server configuration
        server_config = self.config.get("server", {})
        self.host = server_config.get("host", "localhost")
        self.port = server_config.get("port", 23)
        self.server_name = server_config.get("name", "Unknown")
        
        # Bot configuration
        bot_config = self.config.get("bot", {})
        self.character_name = bot_config.get("character_name", "mudbot")
        self.need_character_creation = bot_config.get("need_character_creation", False)
        self.max_command_history = bot_config.get("max_command_history", 50)
        self.max_reconnect_attempts = bot_config.get("max_reconnect_attempts", 3)
        
        # AI configuration
        ai_config = self.config.get("ai", {})
        self.ai_strategy = AIStrategy(ai_config.get("strategy", "explorer"))
        self.behavior_weights = ai_config.get("behavior_weights", {})
        
        # Game-specific configuration
        self.game_config = self.config.get("game_specific", {})
        
        # Connection state
        self.connected = False
        self.connection_lost = False
        self.reader = None
        self.writer = None
        
        # Game state
        self.game_state = GameState.CONNECTING
        self.player = PlayerState()
        
        # Command tracking
        self.command_history = []
        self.last_command = None
        self.character_creation_attempts = 0
        
        # AI conversation context
        self.conversation_history = []
        self.max_conversation_history = 20
        
        # Learning tracking
        self._learned_commands = set()
        self._failed_commands = set()
        self._successful_commands = set()
        
        # Cooldowns
        self._last_command_time = 0
        self._cooldowns = {}
        
        logger.info(f"AI MUD Client initialized for {self.server_name} ({self.host}:{self.port})")
        if self.openai_client:
            logger.info(f"Using AI model: {self.ai_model}")
        else:
            logger.warning("OpenAI client not available - will use fallback commands")

    def _load_config(self, config_file: str) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_file}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return {}

    def _build_ai_context(self, server_response: str, game_state: str) -> str:
        """Build context for AI decision making"""
        context = f"""
You are an AI playing a MUD (Multi-User Dungeon) game called {self.server_name}.

CURRENT GAME STATE:
- Game State: {game_state}
- Character: {self.player.name or 'Unknown'}
- Health: {self.player.health}/{self.player.max_health}
- Mana: {self.player.mana}/{self.player.max_mana}
- Experience: {self.player.experience}
- Gold: {self.player.gold}
- Location: {self.player.location or 'Unknown'}

RECENT COMMANDS:
{self.command_history[-5:] if self.command_history else 'None'}

LEARNED COMMANDS:
{list(self._learned_commands)[-10:] if self._learned_commands else 'None'}

FAILED COMMANDS (avoid these):
{list(self._failed_commands)[-5:] if self._failed_commands else 'None'}

AVAILABLE COMMANDS:
- Movement: {self.game_config.get('movement_commands', [])}
- Combat: {self.game_config.get('combat_commands', [])}
- System: {self.game_config.get('system_commands', [])}
- Special: {self.game_config.get('special_commands', [])}

LATEST SERVER RESPONSE:
{server_response}

INSTRUCTIONS:
1. Analyze the server response carefully
2. Consider the current game state and character status
3. Choose the most appropriate action based on the context
4. Avoid commands that have failed recently
5. Prefer learned commands that have worked before
6. If the response indicates an error or misunderstanding, try to learn from it
7. If character creation is needed, follow the prompts appropriately
8. If in combat, prioritize survival and effective combat commands
9. If exploring, move around and examine interesting things
10. Keep responses concise - just the command, no explanation

RESPOND WITH ONLY THE COMMAND TO SEND, nothing else.
"""
        return context

    async def _get_ai_decision(self, server_response: str) -> Optional[str]:
        """Get AI decision using ChatGPT"""
        try:
            # Check if OpenAI client is available
            if not self.openai_client:
                logger.warning("OpenAI client not available, using fallback")
                return None
            
            # Build context
            context = self._build_ai_context(server_response, self.game_state.value)
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "system",
                "content": context
            })
            
            # Keep conversation history manageable
            if len(self.conversation_history) > self.max_conversation_history:
                self.conversation_history = self.conversation_history[-self.max_conversation_history:]
            
            # Get AI response
            response = self.openai_client.chat.completions.create(
                model=self.ai_model,
                messages=self.conversation_history,
                temperature=self.ai_temperature,
                max_tokens=self.ai_max_tokens,
                stop=None
            )
            
            ai_command = response.choices[0].message.content.strip()
            
            # Log AI decision
            logger.info(f"AI Decision: {ai_command}")
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": ai_command
            })
            
            return ai_command
            
        except Exception as e:
            logger.error(f"Failed to get AI decision: {e}")
            return None

    async def connect(self) -> bool:
        """Connect to the MUD server"""
        try:
            logger.info(f"Connecting to {self.host}:{self.port}...")
            
            self.reader, self.writer = await telnetlib3.open_connection(
                self.host, self.port
            )
            
            self.connected = True
            self.connection_lost = False
            
            # Read initial response
            initial_response = await self.read_response()
            logger.info(f"Initial response received ({len(initial_response)} chars)")
            
            if self.need_character_creation:
                logger.info("Character creation needed, starting creation process...")
                self.game_state = GameState.CHARACTER_CREATION
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        """Disconnect from the server"""
        if self.writer:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except AttributeError:
                pass  # Some writers don't have wait_closed
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
        
        self.connected = False
        self.connection_lost = True
        logger.info("Disconnected from server")

    async def send_command(self, command: str):
        """Send a command to the server"""
        if not self.connected or self.connection_lost:
            logger.warning("Connection lost, cannot send command")
            return

        try:
            # Check cooldown for repeated commands
            current_time = time.time()
            if (self.last_command == command and
                current_time - self._last_command_time < 5):  # 5 second cooldown
                logger.debug(f"Command {command} on cooldown, skipping...")
                return

            self.writer.write(f"{command}\n")
            await self.writer.drain()

            # Add to command history
            self.command_history.append(command)
            if len(self.command_history) > self.max_command_history:
                self.command_history.pop(0)

            self.last_command = command
            self._last_command_time = current_time
            logger.info(f"AI Command: {command}")
            print(f"🤖 AI Bot: {command}")

        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            self.connection_lost = True

    async def read_response(self) -> str:
        """Read response from the server"""
        if not self.connected or self.connection_lost:
            return ""

        try:
            data = await asyncio.wait_for(self.reader.read(1024), timeout=5.0)

            if isinstance(data, bytes):
                response = data.decode('utf-8', errors='ignore')
            elif isinstance(data, str):
                response = data
            else:
                response = str(data)

            # Log the server response
            if response.strip():
                logger.info(f"Server Response: {response.strip()}")
                print(f"📡 Server: {response.strip()}")
                print("-" * 50)  # Separator for readability

            return response
        except asyncio.TimeoutError:
            logger.warning("Timeout reading response")
            return ""
        except Exception as e:
            logger.error(f"Failed to read response: {e}")
            self.connection_lost = True
            return ""

    def _check_immediate_actions(self, response_text: str) -> Optional[str]:
        """Check for immediate actions that don't need AI"""
        response_lower = response_text.lower()
        
        # Character creation errors
        error_messages = [
            "character name already exists", "name is taken", "character exists", 
            "name is either taken", "sadly, that name is either taken", "name is not available"
        ]
        
        if any(msg in response_lower for msg in error_messages):
            logger.warning("Character name already exists, trying different name...")
            import random
            import string
            suffix = ''.join(random.choices(string.ascii_lowercase, k=3))
            return f"{self.character_name}{suffix}"
        
        # Login prompts
        login_prompts = self.game_config.get("login_prompts", [])
        if any(prompt in response_lower for prompt in login_prompts):
            excluded_words = self.game_config.get("excluded_words", [])
            if not any(word in response_lower for word in excluded_words):
                logger.info("Login prompt detected, using character name...")
                return self.character_name
        
        # Password prompts
        password_prompts = self.game_config.get("password_prompts", [])
        if any(prompt in response_lower for prompt in password_prompts):
            default_password = self.game_config.get("default_password", "password123")
            logger.info("Password prompt detected...")
            return default_password
        
        # Character creation prompts
        char_prompts = self.game_config.get("character_creation_prompts", [])
        if any(prompt in response_lower for prompt in char_prompts):
            if "character creation (part" not in response_lower:
                char_option = self.game_config.get("character_creation_menu_option", "2")
                logger.info("Character creation prompt detected...")
                return char_option
        
        # Welcome messages
        welcome_messages = self.game_config.get("welcome_messages", [])
        if any(msg in response_lower for msg in welcome_messages):
            if not hasattr(self, '_welcome_detected'):
                self._welcome_detected = True
                logger.info("Welcome message detected, starting exploration...")
                self.game_state = GameState.EXPLORING
                return "look"
        
        return None

    def _learn_from_response(self, response_text: str):
        """Learn from server response"""
        response_lower = response_text.lower()
        
        # Track failed commands
        if self.last_command and any(error in response_lower for error in 
                                   ["don't understand", "can't do that", "invalid", "not allowed", "i don't understand"]):
            self._failed_commands.add(self.last_command)
            logger.debug(f"Learned failed command: {self.last_command}")
        
        # Track successful commands
        elif self.last_command and not any(error in response_lower for error in 
                                         ["don't understand", "can't do that", "invalid", "not allowed"]):
            self._successful_commands.add(self.last_command)
        
        # Parse command hints
        try_patterns = [
            r"try '([^']+)'",
            r"try \"([^\"]+)\"",
            r"try ([a-zA-Z_]+)",
            r"([a-zA-Z_]+) <[^>]+>",
            r"([a-zA-Z_]+) <anything>"
        ]
        
        for pattern in try_patterns:
            matches = re.findall(pattern, response_text)
            for match in matches:
                if match and len(match) > 2:
                    self._learned_commands.add(match)
                    logger.info(f"Learned new command from hints: {match}")

    async def run_ai_mode(self, duration: int = 60):
        """Run the bot in AI mode"""
        logger.info(f"Starting AI mode for {duration} seconds...")
        print(f"🤖 Starting AI mode for {duration} seconds...")
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            if not self.connected or self.connection_lost:
                logger.warning("Connection lost, attempting to reconnect...")
                if not await self.reconnect():
                    logger.error("Failed to reconnect, exiting...")
                    break
            
            # Read server response
            response = await self.read_response()
            if not response:
                await asyncio.sleep(1)
                continue
            
            # Check for immediate actions first
            immediate_action = self._check_immediate_actions(response)
            if immediate_action:
                await self.send_command(immediate_action)
                await asyncio.sleep(1)
                continue
            
            # Learn from response
            self._learn_from_response(response)
            
            # Get AI decision
            ai_command = await self._get_ai_decision(response)
            if ai_command:
                await self.send_command(ai_command)
                await asyncio.sleep(2)  # Give AI time to process
            else:
                # Fallback to basic commands if AI fails
                fallback_commands = self.game_config.get("movement_commands", ["look"])
                fallback_command = random.choice(fallback_commands)
                await self.send_command(fallback_command)
                await asyncio.sleep(1)
        
        logger.info("AI mode completed")
        print("✅ AI mode completed")

    async def reconnect(self) -> bool:
        """Attempt to reconnect to the server"""
        for attempt in range(self.max_reconnect_attempts):
            logger.info(f"Reconnection attempt {attempt + 1}/{self.max_reconnect_attempts}")
            
            try:
                await self.disconnect()
                await asyncio.sleep(2)
                
                if await self.connect():
                    logger.info("Reconnected successfully")
                    return True
                    
            except Exception as e:
                logger.error(f"Reconnection attempt {attempt + 1} failed: {e}")
        
        logger.error("All reconnection attempts failed")
        return False

    def print_character_info(self):
        """Print current character information"""
        print("\n" + "=" * 50)
        print("=== AI Character Information ===")
        print(f"Name: {self.player.name or 'Unknown'}")
        print(f"Level: {self.player.level}")
        print(f"Health: {self.player.health}/{self.player.max_health}")
        print(f"Mana: {self.player.mana}/{self.player.max_mana}")
        print(f"Experience: {self.player.experience}")
        print(f"Gold: {self.player.gold}")
        print(f"Location: {self.player.location or 'Unknown'}")
        print(f"Current Room: {getattr(self, 'current_room', 'Unknown')}")
        print(f"Game State: {self.game_state.value}")
        print(f"AI Strategy: {self.ai_strategy.value}")
        print(f"Server: {self.server_name}")
        print(f"AI Model: {self.ai_model}")
        print(f"Learned Commands: {len(self._learned_commands)}")
        print(f"Failed Commands: {len(self._failed_commands)}")
        print("=" * 50 + "\n")

async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI-Powered MUD Client")
    parser.add_argument("--config", default="config_examples/sindome_config.json", 
                       help="Configuration file path")
    parser.add_argument("--duration", type=int, default=60, 
                       help="Duration to run in seconds")
    parser.add_argument("--interactive", action="store_true", 
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key in the .env file or environment.")
        return
    
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
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main()) 