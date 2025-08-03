#!/usr/bin/env python3
"""
Enhanced Generic MUD Client
Advanced rule-based AI with sophisticated decision making
No external AI APIs required - completely free and unlimited
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

class EnhancedGenericMUDClient:
    """Enhanced generic MUD client with sophisticated rule-based AI"""
    
    def __init__(self, config_file: str = "config_examples/generic_mud_config.json"):
        # Load configuration
        self.config = self._load_config(config_file)
        
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
        
        # Enhanced learning tracking
        self._learned_commands = set()
        self._failed_commands = set()
        self._successful_commands = set()
        self._command_scores = {}  # Track command effectiveness
        self._context_patterns = {}  # Track successful patterns
        
        # Cooldowns
        self._last_command_time = 0
        self._cooldowns = {}
        
        # Enhanced state tracking
        self._current_room = None
        self._room_history = []
        self._combat_state = False
        self._last_health = 100
        self._last_mana = 100
        
        logger.info(f"Enhanced Generic MUD Client initialized for {self.server_name} ({self.host}:{self.port})")

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

    def _analyze_context(self, response_text: str) -> Dict[str, Any]:
        """Advanced context analysis"""
        context = {
            "combat": False,
            "danger": False,
            "opportunity": False,
            "rest_needed": False,
            "exploration": True,
            "social": False,
            "items": [],
            "npcs": [],
            "exits": [],
            "health_low": False,
            "mana_low": False
        }
        
        response_lower = response_text.lower()
        
        # Combat detection
        combat_keywords = ["attack", "fight", "battle", "enemy", "monster", "damage", "hit", "kill"]
        if any(keyword in response_lower for keyword in combat_keywords):
            context["combat"] = True
            context["danger"] = True
            self._combat_state = True
        
        # Danger detection
        danger_keywords = ["danger", "threat", "warning", "caution", "deadly", "poison", "trap"]
        if any(keyword in response_lower for keyword in danger_keywords):
            context["danger"] = True
        
        # Opportunity detection
        opportunity_keywords = ["treasure", "chest", "gold", "loot", "item", "weapon", "armor", "potion"]
        if any(keyword in response_lower for keyword in opportunity_keywords):
            context["opportunity"] = True
        
        # Health/Mana detection
        if "health" in response_lower and "low" in response_lower:
            context["health_low"] = True
            context["rest_needed"] = True
        
        if "mana" in response_lower and "low" in response_lower:
            context["mana_low"] = True
            context["rest_needed"] = True
        
        # Social detection
        social_keywords = ["say", "tell", "chat", "conversation", "npc", "merchant", "innkeeper"]
        if any(keyword in response_lower for keyword in social_keywords):
            context["social"] = True
        
        # Exit detection
        exit_patterns = [
            r"exits?:?\s*([^\.]+)",
            r"you can go\s+([^\.]+)",
            r"paths?:?\s*([^\.]+)"
        ]
        for pattern in exit_patterns:
            match = re.search(pattern, response_lower)
            if match:
                exits_text = match.group(1)
                context["exits"] = [exit.strip() for exit in exits_text.split(",")]
        
        return context

    def _get_enhanced_decision(self, response_text: str) -> Optional[str]:
        """Get enhanced AI decision using sophisticated rules"""
        try:
            # Analyze context
            context = self._analyze_context(response_text)
            
            # Update game state based on context
            if context["combat"]:
                self.game_state = GameState.COMBAT
            elif context["rest_needed"]:
                self.game_state = GameState.RESTING
            elif context["opportunity"]:
                self.game_state = GameState.INVENTORY
            else:
                self.game_state = GameState.EXPLORING
            
            # Decision making based on context and strategy
            strategy = self.ai_strategy.value
            
            # Combat priority
            if context["combat"]:
                combat_commands = self.game_config.get("combat_commands", [])
                healing_commands = self.game_config.get("healing_commands", [])
                
                if context["health_low"]:
                    return random.choice(healing_commands)
                elif context["mana_low"]:
                    return "rest"
                else:
                    return random.choice(combat_commands)
            
            # Rest priority
            if context["rest_needed"]:
                rest_commands = ["rest", "sleep", "meditate"]
                return random.choice(rest_commands)
            
            # Opportunity priority
            if context["opportunity"]:
                inventory_commands = self.game_config.get("inventory_commands", [])
                return random.choice(inventory_commands)
            
            # Social priority
            if context["social"]:
                social_commands = self.game_config.get("social_commands", [])
                return random.choice(social_commands)
            
            # Exploration (default)
            if strategy == "explorer":
                # Prefer learned commands
                if self._learned_commands:
                    learned_list = list(self._learned_commands)
                    # Weight by success rate
                    weighted_commands = []
                    for cmd in learned_list:
                        score = self._command_scores.get(cmd, 0.5)
                        weighted_commands.extend([cmd] * int(score * 10))
                    
                    if weighted_commands:
                        return random.choice(weighted_commands)
                
                # Use movement commands
                movement_commands = self.game_config.get("movement_commands", [])
                if movement_commands:
                    return random.choice(movement_commands)
            
            elif strategy == "combat_focused":
                combat_commands = self.game_config.get("combat_commands", [])
                if combat_commands:
                    return random.choice(combat_commands)
            
            elif strategy == "farmer":
                # Focus on gathering and crafting
                farming_commands = ["gather", "harvest", "craft", "mine", "fish"]
                return random.choice(farming_commands)
            
            # Fallback to system commands
            system_commands = self.game_config.get("system_commands", ["look"])
            return random.choice(system_commands)
            
        except Exception as e:
            logger.error(f"Failed to get enhanced decision: {e}")
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
            logger.info(f"Enhanced AI Command: {command}")
            print(f"🤖 Enhanced Bot: {command}")

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
                # Check for Cybersphere-specific login
                if "connect guest" in response_lower or "connect <username>" in response_lower:
                    logger.info("Cybersphere login detected, using connect guest...")
                    return "connect guest"
                else:
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
        
        # Check for Cybersphere login prompt
        if "valid commands at this point are" in response_lower and "connect" in response_lower:
            if not hasattr(self, '_cybersphere_login_sent'):
                self._cybersphere_login_sent = True
                logger.info("Cybersphere login prompt detected, sending connect guest...")
                return "connect guest"
        
        return None

    def _learn_from_response(self, response_text: str):
        """Enhanced learning from server response"""
        response_lower = response_text.lower()
        
        # Track failed commands
        if self.last_command and any(error in response_lower for error in 
                                   ["don't understand", "can't do that", "invalid", "not allowed", "i don't understand"]):
            self._failed_commands.add(self.last_command)
            self._command_scores[self.last_command] = max(0.1, self._command_scores.get(self.last_command, 0.5) - 0.1)
            logger.debug(f"Learned failed command: {self.last_command}")
        
        # Track successful commands
        elif self.last_command and not any(error in response_lower for error in 
                                         ["don't understand", "can't do that", "invalid", "not allowed"]):
            self._successful_commands.add(self.last_command)
            self._command_scores[self.last_command] = min(1.0, self._command_scores.get(self.last_command, 0.5) + 0.1)
        
        # Parse command hints with enhanced patterns
        try_patterns = [
            r"try '([^']+)'",
            r"try \"([^\"]+)\"",
            r"try ([a-zA-Z_]+)",
            r"([a-zA-Z_]+) <[^>]+>",
            r"([a-zA-Z_]+) <anything>",
            r"you can ([a-zA-Z_]+)",
            r"use ([a-zA-Z_]+)",
            r"command ([a-zA-Z_]+)"
        ]
        
        for pattern in try_patterns:
            matches = re.findall(pattern, response_text)
            for match in matches:
                if match and len(match) > 2:
                    self._learned_commands.add(match)
                    self._command_scores[match] = 0.8  # High initial score for learned commands
                    logger.info(f"Learned new command from hints: {match}")
        
        # Learn context patterns
        if "combat" in response_lower or "fight" in response_lower:
            self._context_patterns["combat"] = True
        if "treasure" in response_lower or "loot" in response_lower:
            self._context_patterns["opportunity"] = True
        if "danger" in response_lower or "threat" in response_lower:
            self._context_patterns["danger"] = True

    async def run_enhanced_mode(self, duration: int = 60):
        """Run the bot in enhanced mode"""
        logger.info(f"Starting enhanced mode for {duration} seconds...")
        print(f"🤖 Starting enhanced mode for {duration} seconds...")
        
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
            
            # Get enhanced AI decision
            ai_command = self._get_enhanced_decision(response)
            if ai_command:
                await self.send_command(ai_command)
                await asyncio.sleep(2)  # Give AI time to process
            else:
                # Fallback to basic commands if AI fails
                fallback_commands = self.game_config.get("movement_commands", ["look"])
                fallback_command = random.choice(fallback_commands)
                await self.send_command(fallback_command)
                await asyncio.sleep(1)
        
        logger.info("Enhanced mode completed")
        print("✅ Enhanced mode completed")

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
        print("=== Enhanced Character Information ===")
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
        print(f"Learned Commands: {len(self._learned_commands)}")
        print(f"Failed Commands: {len(self._failed_commands)}")
        print(f"Successful Commands: {len(self._successful_commands)}")
        print(f"Command Scores: {len(self._command_scores)}")
        print("=" * 50 + "\n")

async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Generic MUD Client")
    parser.add_argument("--config", default="config_examples/sindome_config.json", 
                       help="Configuration file path")
    parser.add_argument("--duration", type=int, default=60, 
                       help="Duration to run in seconds")
    parser.add_argument("--strategy", default="explorer", 
                       help="AI strategy (explorer, combat_focused, farmer, social, adventurer)")
    
    args = parser.parse_args()
    
    # Create and run client
    client = EnhancedGenericMUDClient(args.config)
    client.ai_strategy = AIStrategy(args.strategy)
    
    try:
        if await client.connect():
            print("✅ Connected successfully!")
            print("🤖 Starting enhanced mode...")
            await client.run_enhanced_mode(args.duration)
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