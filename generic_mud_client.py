#!/usr/bin/env python3
"""
Generic MUD Client - Configuration-based MUD bot
"""

import asyncio
import telnetlib3
import json
import logging
import random
import string
import re
import time
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG)
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

class GenericMUDClient:
    """Generic MUD client that uses configuration-based parsing"""
    
    def __init__(self, config_file: str = "config_examples/generic_mud_config.json"):
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Server settings
        server_config = self.config.get("server", {})
        self.host = server_config.get("host", "localhost")
        self.port = server_config.get("port", 23)
        self.server_name = server_config.get("name", "Generic MUD")
        self.server_description = server_config.get("description", "")
        
        # Bot settings
        bot_config = self.config.get("bot", {})
        self.bot_name = bot_config.get("name", "genericbot")
        self.character_name = bot_config.get("character_name", "genericbot")
        self.auto_login = bot_config.get("auto_login", True)
        self.auto_delay_min = bot_config.get("auto_delay_min", 1.0)
        self.auto_delay_max = bot_config.get("auto_delay_max", 3.0)
        self.max_command_history = bot_config.get("max_command_history", 50)
        
        # AI settings
        ai_config = self.config.get("ai", {})
        self.ai_strategy = AIStrategy(ai_config.get("strategy", "explorer"))
        self.behavior_weights = ai_config.get("weights", {})
        
        # Game-specific settings
        self.game_config = self.config.get("game_specific", {})
        
        # Connection state
        self.reader = None
        self.writer = None
        self.connected = False
        self.connection_lost = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3
        self.need_character_creation = False
        
        # Game state
        self.state = GameState.CONNECTING
        self.player = PlayerState()
        self.current_room = None
        self.rooms_visited = set()
        self.combat_target = None
        self.last_command = ""
        self.command_history = []
        self.auto_mode = False
        
        # Cooldowns
        self.last_hint_time = 0
        self.last_newbie_help_time = 0
        self.character_creation_attempts = 0
        
        # Memory
        self.memory = {
            "dangerous_areas": set(),
            "safe_areas": set(),
            "item_locations": {},
            "npc_locations": {},
            "combat_history": []
        }
        
        # Setup logging
        logging_config = self.config.get("logging", {})
        log_level = getattr(logging, logging_config.get("level", "DEBUG").upper())
        log_file = logging_config.get("file", "mud_bot.log")
        log_format = logging_config.get("format", "%(asctime)s - %(levelname)s - %(message)s")
        
        # Configure file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
        logger.setLevel(log_level)

    def _load_config(self, config_file: str) -> dict:
        """Load configuration from file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_file} not found, using defaults")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing config file: {e}")
            return {}

    async def connect(self) -> bool:
        """Connect to the MUD server"""
        try:
            logger.info(f"Connecting to {self.server_name} at {self.host}:{self.port}")
            self.reader, self.writer = await telnetlib3.open_connection(
                self.host, self.port
            )
            self.connection_lost = False
            self.connected = True
            self.state = GameState.CONNECTING
            logger.info(f"Connected successfully to {self.server_name}")
            
            # Wait for initial server response
            logger.info("Waiting for initial server response...")
            initial_response = await self.read_response()
            if initial_response:
                logger.info(f"Initial response received ({len(initial_response)} chars)")
                logger.debug(f"Initial response: {initial_response[:200]}...")
                
                # Parse initial response for login prompts
                parsed = self.parse_response(initial_response)
                
                # If we need to create a character, send the menu option
                if self.need_character_creation:
                    menu_option = self.game_config.get("character_creation_menu_option", "2")
                    logger.info(f"Need to create character, selecting option {menu_option}...")
                    await self.send_command(menu_option)
                    self.need_character_creation = False
                else:
                    immediate_action = self._check_immediate_actions(parsed)
                    if immediate_action:
                        logger.info(f"Sending immediate action: {immediate_action}")
                        await self.send_command(immediate_action)
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        """Disconnect from the server"""
        if self.writer:
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except AttributeError:
                pass
        self.connected = False
        self.connection_lost = True
        logger.info("Disconnected from server")

    async def send_command(self, command: str):
        """Send a command to the server"""
        if not self.connected or self.connection_lost:
            logger.warning("Connection lost, cannot send command")
            return
        
        try:
            import time
            
            # Check cooldown for repeated commands
            if hasattr(self, '_last_command_time') and hasattr(self, '_last_command'):
                current_time = time.time()
                if (self._last_command == command and 
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
            self._last_command = command
            self._last_command_time = time.time()
            logger.info(f"AI Command: {command}")
            print(f"🤖 Bot: {command}")
            
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

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse server response using configuration-based patterns"""
        parsed = {
            "text": response,
            "room_name": None,
            "room_type": None,
            "exits": [],
            "items": [],
            "npcs": [],
            "health": None,
            "mana": None,
            "experience": None,
            "gold": None
        }
        
        response_lower = response.lower()
        
        # Parse room information
        room_patterns = self.game_config.get("room_patterns", {})
        
        # Room name and type
        room_name_pattern = room_patterns.get("room_name")
        if room_name_pattern:
            match = re.search(room_name_pattern, response)
            if match:
                parsed["room_name"] = match.group(1)
                parsed["room_type"] = match.group(2)
        
        # Exits
        exits_pattern = room_patterns.get("exits")
        if exits_pattern:
            match = re.search(exits_pattern, response_lower)
            if match:
                exits_text = match.group(1)
                # Parse exits (this is a simplified version)
                parsed["exits"] = [exit.strip() for exit in exits_text.split(",")]
        
        # NPCs
        npc_patterns = room_patterns.get("npcs", [])
        for pattern in npc_patterns:
            matches = re.findall(pattern, response)
            parsed["npcs"].extend(matches)
        
        # Status patterns
        status_patterns = self.game_config.get("status_patterns", {})
        
        # Health and mana
        health_mana_patterns = status_patterns.get("health_mana", [])
        for pattern in health_mana_patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    health, mana = match.groups()
                    parsed["health"] = int(health)
                    parsed["mana"] = int(mana)
                    break
                except (ValueError, IndexError):
                    continue
        
        # Experience
        exp_patterns = status_patterns.get("experience", [])
        for pattern in exp_patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    parsed["experience"] = int(match.group(1))
                    break
                except (ValueError, IndexError):
                    continue
        
        # Gold
        gold_patterns = status_patterns.get("gold", [])
        for pattern in gold_patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    parsed["gold"] = int(match.group(1))
                    break
                except (ValueError, IndexError):
                    continue
        
        return parsed

    def _check_immediate_actions(self, parsed_response: Dict[str, Any]) -> Optional[str]:
        """Check for immediate actions based on configuration"""
        response_text = parsed_response.get("text", "").lower()
        
        # Character creation errors (check first)
        error_messages = [
            "character name already exists", "name is taken", "character exists", 
            "name is either taken", "sadly, that name is either taken", "name is not available"
        ]
        
        if any(msg in response_text for msg in error_messages):
            logger.warning("Character name already exists, trying different name...")
            import random
            import string
            suffix = ''.join(random.choices(string.ascii_lowercase, k=3))
            return f"{self.character_name}{suffix}"
        
        # Login prompts
        login_prompts = self.game_config.get("login_prompts", [])
        if any(prompt in response_text for prompt in login_prompts):
            excluded_words = self.game_config.get("excluded_words", [])
            if not any(word in response_text for word in excluded_words):
                logger.info("Login prompt detected, using character name...")
                return self.character_name
        
        # Character creation prompts
        char_prompts = self.game_config.get("character_creation_prompts", [])
        if any(prompt in response_text for prompt in char_prompts):
            if "character creation (part" not in response_text:
                # Limit character creation attempts
                if self.character_creation_attempts >= 3:
                    logger.info("Too many character creation attempts, stopping...")
                    return "look"
                
                self.character_creation_attempts += 1
                logger.info(f"Character creation menu prompt detected... (attempt {self.character_creation_attempts}/3)")
                
                # Try login first, then character creation if needed
                if not self.need_character_creation:
                    logger.info("Trying login first...")
                    return self.character_name
                else:
                    logger.info("Creating new character...")
                    return self.game_config.get("character_creation_menu_option", "2")
        
        # Name selection
        name_prompts = self.game_config.get("name_selection_prompts", [])
        if any(prompt in response_text for prompt in name_prompts):
            logger.info("Character name selection prompt detected...")
            # Use fantasy names from config
            fantasy_names = self.game_config.get("fantasy_names", [])
            if fantasy_names:
                name = random.choice(fantasy_names)
                suffix = ''.join(random.choices(string.ascii_lowercase, k=2))
                name_with_suffix = f"{name}{suffix}"
                logger.info(f"Using character name: {name_with_suffix}")
                return name_with_suffix
            else:
                return self.character_name
        
        # Password prompts
        password_prompts = self.game_config.get("password_prompts", [])
        if any(prompt in response_text for prompt in password_prompts):
            logger.info("Password prompt detected, using default password...")
            return self.game_config.get("default_password", "password123")
        
        # Check for Sindome-specific connection prompts (check these first)
        if "type 'connect guest'" in response_text.lower():
            logger.info("Sindome guest connection prompt detected...")
            return "connect guest"
        
        if "connect <username> <password>" in response_text.lower():
            logger.info("Sindome login prompt detected...")
            return f"connect {self.character_name} {self.game_config.get('default_password', 'password123')}"
        
        # Check if we're in a limited command mode (like connection phase)
        if "valid commands are:" in response_text.lower():
            logger.info("Limited command mode detected, trying to connect...")
            # Try different connection commands based on what's available
            if "welcome" in response_text.lower():
                return "welcome"
            elif "uptime" in response_text.lower():
                return "uptime"
            elif "connect" in response_text.lower():
                return "connect"
            else:
                return "welcome"  # Default to welcome
        
        # Password setup (check this last to avoid false positives)
        password_setup_prompts = self.game_config.get("password_setup_prompts", [])
        if any(prompt in response_text for prompt in password_setup_prompts):
            # Only trigger if we're not in the initial connection phase
            if "connect guest" not in response_text.lower() and "connect <username>" not in response_text.lower():
                logger.info("Character creation password prompt detected...")
                return self.game_config.get("default_password", "password123")
        
        # Password confirmation
        password_confirmation_prompts = self.game_config.get("password_confirmation_prompts", [])
        if any(prompt in response_text for prompt in password_confirmation_prompts):
            logger.info("Password confirmation prompt detected...")
            return self.game_config.get("default_password", "password123")
        
        # Gender selection
        gender_prompts = self.game_config.get("gender_selection_prompts", [])
        if any(prompt in response_text for prompt in gender_prompts):
            logger.info("Gender selection prompt detected...")
            gender_options = self.game_config.get("gender_options", {})
            default_gender = self.game_config.get("default_gender", "male")
            return gender_options.get(default_gender, "1")
        
        # Race selection
        race_prompts = self.game_config.get("race_selection_prompts", [])
        if any(prompt in response_text for prompt in race_prompts):
            logger.info("Race selection prompt detected...")
            return self.game_config.get("default_race", "human")
        
        # Class selection
        class_prompts = self.game_config.get("class_selection_prompts", [])
        if any(prompt in response_text for prompt in class_prompts):
            logger.info("Class selection prompt detected...")
            return self.game_config.get("default_class", "monk")
        
        # City selection
        city_prompts = self.game_config.get("city_selection_prompts", [])
        if any(prompt in response_text for prompt in city_prompts):
            logger.info("City selection prompt detected...")
            return self.game_config.get("default_city", "ashtan")
        
        # Service agreement
        service_prompts = self.game_config.get("service_agreement_prompts", [])
        if any(prompt in response_text for prompt in service_prompts):
            logger.info("Service agreement prompt detected...")
            return "yes"
        
        # Screenreader tools
        screenreader_prompts = self.game_config.get("screenreader_prompts", [])
        if any(prompt in response_text for prompt in screenreader_prompts):
            logger.info("Screenreader tools prompt detected...")
            return "no"
        
        # Character creation success
        success_messages = self.game_config.get("character_creation_success", [])
        if any(msg in response_text for msg in success_messages):
            logger.info("Character created successfully, starting gameplay...")
            self.print_character_info()
            return "look"
        
        # Welcome messages (more specific detection)
        welcome_messages = self.game_config.get("welcome_messages", [])
        if any(msg in response_text for msg in welcome_messages):
            # Check if it's actually a welcome message, not just part of server banner
            if ("welcome" in response_text and 
                ("to" in response_text or "back" in response_text or "login" in response_text)):
                # Only trigger once per session to avoid cycling
                if not hasattr(self, '_welcome_detected'):
                    logger.info("Welcome message detected, starting exploration...")
                    self._welcome_detected = True
                    self.print_character_info()
                    return "look"
                else:
                    logger.debug("Welcome message already detected, skipping...")
        
        # Try to detect character name from server responses
        self._detect_character_info(response_text)
        
        # Password incorrect
        if "password incorrect" in response_text or "invalid password" in response_text:
            logger.info("Password incorrect, connection will be closed. Will create new character on next connection.")
            self.connection_lost = True
            self.need_character_creation = True
            return None
        
        # Affliction handling
        affliction_remedies = self.game_config.get("affliction_remedies", {})
        for affliction, remedy in affliction_remedies.items():
            if affliction in response_text:
                logger.info(f"{affliction.title()} affliction detected, applying {remedy}...")
                return remedy
        
        return None

    def _detect_character_info(self, response_text: str):
        """Detect character information from server responses"""
        response_lower = response_text.lower()
        
        # Try to detect character name from various patterns
        name_patterns = [
            r"logging you in as `([^`']+)'",
            r"welcome, ([a-zA-Z_]+)",
            r"you are ([a-zA-Z_]+)",
            r"character name: ([a-zA-Z_]+)",
            r"playing as ([a-zA-Z_]+)"
        ]
        
        for pattern in name_patterns:
            import re
            match = re.search(pattern, response_text)
            if match:
                detected_name = match.group(1)
                # Filter out common false positives
                if (detected_name and 
                    detected_name != "guest" and 
                    len(detected_name) > 2 and
                    not detected_name.lower() in ["about", "welcome", "you", "the", "and", "are", "for", "can", "not", "all", "any", "one", "two", "three", "four", "five"]):
                    self.player.name = detected_name
                    logger.info(f"Detected character name: {detected_name}")
                    break
        
        # Try to detect health/mana from status patterns
        status_patterns = self.game_config.get("status_patterns", {})
        
        # Health and mana
        health_mana_patterns = status_patterns.get("health_mana", [])
        for pattern in health_mana_patterns:
            match = re.search(pattern, response_text)
            if match:
                try:
                    health, mana = match.groups()
                    self.player.health = int(health)
                    self.player.mana = int(mana)
                    logger.info(f"Detected health: {health}, mana: {mana}")
                    break
                except (ValueError, IndexError):
                    continue
        
        # Experience
        exp_patterns = status_patterns.get("experience", [])
        for pattern in exp_patterns:
            match = re.search(pattern, response_text)
            if match:
                try:
                    self.player.experience = int(match.group(1))
                    logger.info(f"Detected experience: {self.player.experience}")
                    break
                except (ValueError, IndexError):
                    continue
        
        # Gold
        gold_patterns = status_patterns.get("gold", [])
        for pattern in gold_patterns:
            match = re.search(pattern, response_text)
            if match:
                try:
                    self.player.gold = int(match.group(1))
                    logger.info(f"Detected gold: {self.player.gold}")
                    break
                except (ValueError, IndexError):
                    continue

    def _parse_command_hints(self, response_text: str):
        """Parse command hints from server error messages"""
        import re
        
        # Initialize command hints if not exists
        if not hasattr(self, '_learned_commands'):
            self._learned_commands = set()
        
        # Look for "try 'command'" patterns
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
        
        # Look for command descriptions
        command_desc_patterns = [
            r"([a-zA-Z_]+) <[^>]+> ([^.]+)",
            r"([a-zA-Z_]+) ([^.]+)",
        ]
        
        for pattern in command_desc_patterns:
            matches = re.findall(pattern, response_text)
            for match in matches:
                if match[0] and len(match[0]) > 2:
                    self._learned_commands.add(match[0])
                    logger.info(f"Learned new command from description: {match[0]} - {match[1]}")
        
        # Look for specific command suggestions
        if "examine me" in response_text.lower():
            self._learned_commands.add("examine me")
            logger.info("Learned command: examine me")
        if "examine here" in response_text.lower():
            self._learned_commands.add("examine here")
            logger.info("Learned command: examine here")
        if "look sign" in response_text.lower():
            self._learned_commands.add("look sign")
            logger.info("Learned command: look sign")
        if "tour" in response_text.lower():
            self._learned_commands.add("tour")
            logger.info("Learned command: tour")
        if "roles" in response_text.lower():
            self._learned_commands.add("roles")
            logger.info("Learned command: roles")
        
        # Track role selection state
        if "roles" in response_text.lower() and "character concepts" in response_text.lower():
            if not hasattr(self, '_role_selection_started'):
                self._role_selection_started = True
                logger.info("Role selection process started")
        
        # Track tour state
        if "tour" in response_text.lower() and "check out the world" in response_text.lower():
            if not hasattr(self, '_tour_started'):
                self._tour_started = True
                logger.info("Tour process started")

    def generate_ai_response(self, parsed_response: Dict[str, Any]) -> Optional[str]:
        """Generate AI response based on current state and configuration"""
        response_text = parsed_response.get("text", "").lower()
        
        # Initialize learning tracking
        if not hasattr(self, '_failed_commands'):
            self._failed_commands = set()
        if not hasattr(self, '_successful_commands'):
            self._successful_commands = set()
        
        # Check for immediate actions first
        immediate_action = self._check_immediate_actions(parsed_response)
        if immediate_action:
            return immediate_action
        
        # Handle role selection and tour processes
        role_action = self._handle_role_selection(parsed_response)
        if role_action:
            return role_action
        
        # Learn from previous responses
        if self.last_command and response_text:
            if any(error in response_text for error in ["don't understand", "can't do that", "invalid", "not allowed"]):
                self._failed_commands.add(self.last_command)
                logger.debug(f"Learned failed command: {self.last_command}")
                
                # Parse command hints from server response
                self._parse_command_hints(response_text)
            else:
                self._successful_commands.add(self.last_command)
        
        # Get current strategy weights
        strategy_name = self.ai_strategy.value
        weights = self.behavior_weights.get(strategy_name, self.behavior_weights.get("explorer", {}))
        
        # Generate response based on weights
        action = random.choices(
            list(weights.keys()),
            weights=list(weights.values())
        )[0]
        
        # Get commands for this action
        if action == "explore":
            return self._exploration_ai(parsed_response, weights)
        elif action == "combat":
            return self._combat_ai(parsed_response)
        elif action == "loot":
            return self._loot_ai(parsed_response)
        elif action == "rest":
            return self._resting_ai(parsed_response)
        else:
            return self._general_ai(parsed_response)

    def _handle_role_selection(self, parsed_response: Dict[str, Any]) -> Optional[str]:
        """Handle role selection and tour processes"""
        response_text = parsed_response.get("text", "").lower()
        
        # Check if we're in role selection mode
        if hasattr(self, '_role_selection_started') and self._role_selection_started:
            if "roles" in response_text and "character concepts" in response_text:
                logger.info("Starting role selection process...")
                return "roles"
            elif "role" in response_text and "select" in response_text:
                # If we see role selection options, pick one
                if "corporate" in response_text:
                    return "1"  # Select first option
                elif "cop" in response_text:
                    return "2"  # Select second option
                elif "criminal" in response_text:
                    return "3"  # Select third option
                else:
                    return "1"  # Default to first option
        
        # Check if we're in tour mode
        if hasattr(self, '_tour_started') and self._tour_started:
            if "tour" in response_text and "check out the world" in response_text:
                logger.info("Starting tour process...")
                return "tour"
            elif "tour" in response_text and "guide" in response_text:
                return "yes"  # Accept tour
            elif "tour" in response_text and "continue" in response_text:
                return "continue"  # Continue tour
        
        # Check for sign reading
        if "look sign" in response_text.lower():
            logger.info("Reading sign for commands...")
            return "look sign"
        
        return None

    def _exploration_ai(self, parsed_response: Dict[str, Any], weights: Dict[str, float]) -> Optional[str]:
        """Exploration AI logic"""
        # Get movement commands from config
        movement_commands = self.game_config.get("movement_commands", ["north", "south", "east", "west"])
        
        # Get special commands from config
        special_commands = self.game_config.get("special_commands", [])
        
        # Get learned commands from server hints
        learned_commands = getattr(self, '_learned_commands', set())
        
        # Filter out failed commands
        available_movement = [cmd for cmd in movement_commands if cmd not in self._failed_commands]
        available_special = [cmd for cmd in special_commands if cmd not in self._failed_commands]
        available_learned = [cmd for cmd in learned_commands if cmd not in self._failed_commands]
        
        # If no available movement commands, fall back to successful ones or look
        if not available_movement:
            available_movement = [cmd for cmd in movement_commands if cmd in self._successful_commands]
            if not available_movement:
                available_movement = ["look"]
        
        # Prioritize role selection and tour commands
        priority_commands = ["roles", "tour", "look sign"]
        available_priority = [cmd for cmd in priority_commands if cmd not in self._failed_commands]
        
        if available_priority and random.random() < 0.6:  # 60% chance to try priority commands
            return random.choice(available_priority)
        
        # Prioritize learned commands from server hints
        if available_learned and random.random() < 0.4:  # 40% chance to try learned commands
            return random.choice(list(available_learned))
        
        # Random exploration actions
        if random.random() < 0.5:  # 50% chance for movement
            return random.choice(available_movement)
        elif random.random() < 0.1 and available_special:  # 10% chance for special commands
            return random.choice(available_special)
        else:  # 40% chance for look
            return "look"

    def _combat_ai(self, parsed_response: Dict[str, Any]) -> Optional[str]:
        """Combat AI logic"""
        combat_commands = self.game_config.get("combat_commands", ["attack"])
        return random.choice(combat_commands)

    def _loot_ai(self, parsed_response: Dict[str, Any]) -> Optional[str]:
        """Loot AI logic"""
        inventory_commands = self.game_config.get("inventory_commands", ["inventory"])
        return random.choice(inventory_commands)

    def _resting_ai(self, parsed_response: Dict[str, Any]) -> Optional[str]:
        """Resting AI logic"""
        healing_commands = self.game_config.get("healing_commands", ["rest"])
        return random.choice(healing_commands)

    def _general_ai(self, parsed_response: Dict[str, Any]) -> Optional[str]:
        """General AI logic"""
        system_commands = self.game_config.get("system_commands", ["look"])
        return random.choice(system_commands)

    async def run_auto_mode(self):
        """Run the bot in automatic mode"""
        logger.info("Starting auto mode")
        self.auto_mode = True
        
        try:
            while self.auto_mode:
                if self.connection_lost:
                    if self.reconnect_attempts < self.max_reconnect_attempts:
                        logger.warning(f"Connection lost, attempting to reconnect... (attempt {self.reconnect_attempts + 1}/{self.max_reconnect_attempts})")
                        await self.reconnect()
                    else:
                        logger.error("Max reconnection attempts reached, stopping auto mode")
                        break
                
                # Read response
                response = await self.read_response()
                if not response:
                    logger.warning("No response, waiting...")
                    await asyncio.sleep(1)
                    continue
                
                # Parse response
                parsed_response = self.parse_response(response)
                
                # Generate AI response
                ai_command = self.generate_ai_response(parsed_response)
                
                if ai_command:
                    await self.send_command(ai_command)
                
                # Random delay
                delay = random.uniform(self.auto_delay_min, self.auto_delay_max)
                await asyncio.sleep(delay)
                
        except KeyboardInterrupt:
            logger.info("Auto mode interrupted by user")
        except Exception as e:
            logger.error(f"Error in auto mode: {e}")
        finally:
            self.auto_mode = False

    async def reconnect(self):
        """Attempt to reconnect to the server"""
        self.reconnect_attempts += 1
        logger.info("Attempting to reconnect...")
        
        await self.disconnect()
        await asyncio.sleep(2)
        
        if await self.connect():
            logger.info("Reconnected successfully")
            self.reconnect_attempts = 0
        else:
            logger.error("Failed to reconnect")

    def print_character_info(self):
        """Print character information and stats"""
        print("\n=== Character Information ===")
        print(f"Name: {self.player.name or self.character_name or 'Unknown'}")
        print(f"Level: {self.player.level}")
        print(f"Health: {self.player.health}/{self.player.max_health}")
        print(f"Mana: {self.player.mana}/{self.player.max_mana}")
        print(f"Experience: {self.player.experience}")
        print(f"Gold: {self.player.gold}")
        print(f"Location: {self.player.location or 'Unknown'}")
        print(f"Current Room: {self.current_room or 'Unknown'}")
        print(f"Game State: {self.state.value}")
        print(f"AI Strategy: {self.ai_strategy.value}")
        print(f"Server: {self.server_name}")
        print("=" * 40)

async def main():
    """Main function"""
    client = GenericMUDClient()
    
    if await client.connect():
        try:
            await client.run_auto_mode()
        finally:
            await client.disconnect()
    else:
        logger.error("Failed to connect to MUD server")

if __name__ == "__main__":
    asyncio.run(main()) 