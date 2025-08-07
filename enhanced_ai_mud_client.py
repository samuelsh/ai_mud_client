#!/usr/bin/env python3
"""
Enhanced AI MUD Client with:
- Better prompt engineering
- Stateful context management
- Input filtering
- Safety & loop protection
- Model swap support
"""

import asyncio
import json
import logging
import random
import re
import sqlite3
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import argparse
import os
from pathlib import Path

# AI Backends
try:
    import openai
except ImportError:
    openai = None

try:
    import requests
except ImportError:
    requests = None

import telnetlib3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_ai_mud.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GameState(Enum):
    """Enhanced game state enumeration"""
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
    DIALOGUE = "dialogue"

class AIBackend(Enum):
    """AI backend types"""
    OPENAI = "openai"
    OLLAMA = "ollama"
    CLAUDE = "claude"
    MOCK = "mock"

@dataclass
class GameContext:
    """Enhanced game context with stateful information"""
    current_location: str = ""
    inventory: List[str] = None
    last_events: List[str] = None
    dialogue_history: List[str] = None
    health: int = 100
    max_health: int = 100
    mana: int = 100
    max_mana: int = 100
    experience: int = 0
    gold: int = 0
    level: int = 1
    game_state: GameState = GameState.CONNECTING
    last_command: str = ""
    command_history: List[str] = None
    failed_commands: set = None
    successful_commands: set = None
    cooldowns: Dict[str, float] = None
    
    def __post_init__(self):
        if self.inventory is None:
            self.inventory = []
        if self.last_events is None:
            self.last_events = []
        if self.dialogue_history is None:
            self.dialogue_history = []
        if self.command_history is None:
            self.command_history = []
        if self.failed_commands is None:
            self.failed_commands = set()
        if self.successful_commands is None:
            self.successful_commands = set()
        if self.cooldowns is None:
            self.cooldowns = {}

class MessageFilter:
    """Filter and categorize game messages"""
    
    def __init__(self):
        # Patterns for different message types
        self.room_patterns = [
            r"^[A-Z][^.]*\.\s*\([A-Za-z]+\)",  # Room names
            r"exits?:?\s*[A-Za-z,\s]+",  # Exits
            r"you see:\s*[^.]*",  # Items/NPCs
        ]
        
        self.system_patterns = [
            r"^>",  # Prompts
            r"^\d+/\d+",  # Health/Mana
            r"^[A-Z][a-z]+ says,",  # NPC dialogue
            r"^You say,",  # Player dialogue
        ]
        
        self.ansi_pattern = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
    
    def filter_message(self, message: str) -> Dict[str, Any]:
        """Filter and categorize a message"""
        # Remove ANSI codes
        clean_message = self.ansi_pattern.sub('', message)
        
        result = {
            'original': message,
            'clean': clean_message,
            'type': 'unknown',
            'content': clean_message,
            'is_prompt': False,
            'is_room': False,
            'is_system': False,
            'is_dialogue': False
        }
        
        # Check for prompts
        if clean_message.strip().endswith('>') or ':' in clean_message and len(clean_message.strip()) < 50:
            result['type'] = 'prompt'
            result['is_prompt'] = True
            return result
        
        # Check for room descriptions
        for pattern in self.room_patterns:
            if re.search(pattern, clean_message, re.IGNORECASE):
                result['type'] = 'room'
                result['is_room'] = True
                return result
        
        # Check for system messages
        for pattern in self.system_patterns:
            if re.search(pattern, clean_message, re.IGNORECASE):
                result['type'] = 'system'
                result['is_system'] = True
                return result
        
        # Check for dialogue
        if any(keyword in clean_message.lower() for keyword in ['says,', 'tells you,', 'shouts,']):
            result['type'] = 'dialogue'
            result['is_dialogue'] = True
            return result
        
        return result

class StateManager:
    """Manage game state with SQLite persistence"""
    
    def __init__(self, db_path: str = "game_state.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS game_context (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    location TEXT,
                    health INTEGER,
                    mana INTEGER,
                    experience INTEGER,
                    gold INTEGER,
                    level INTEGER,
                    game_state TEXT,
                    last_command TEXT,
                    context_data TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    event_type TEXT,
                    content TEXT,
                    context TEXT
                )
            """)
    
    def save_context(self, context: GameContext):
        """Save current context to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO game_context 
                (timestamp, location, health, mana, experience, gold, level, game_state, last_command, context_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                context.current_location,
                context.health,
                context.mana,
                context.experience,
                context.gold,
                context.level,
                context.game_state.value,
                context.last_command,
                json.dumps({
                    'inventory': context.inventory,
                    'last_events': context.last_events[-10:],  # Keep last 10
                    'dialogue_history': context.dialogue_history[-5:],  # Keep last 5
                    'command_history': context.command_history[-20:],  # Keep last 20
                    'failed_commands': list(context.failed_commands),
                    'successful_commands': list(context.successful_commands)
                })
            ))
    
    def load_context(self) -> Optional[GameContext]:
        """Load the most recent context from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM game_context 
                ORDER BY timestamp DESC 
                LIMIT 1
            """)
            row = cursor.fetchone()
            
            if row:
                context_data = json.loads(row[10]) if row[10] else {}
                return GameContext(
                    current_location=row[2] or "",
                    health=row[3] or 100,
                    mana=row[4] or 100,
                    experience=row[5] or 0,
                    gold=row[6] or 0,
                    level=row[7] or 1,
                    game_state=GameState(row[8]) if row[8] else GameState.CONNECTING,
                    last_command=row[9] or "",
                    inventory=context_data.get('inventory', []),
                    last_events=context_data.get('last_events', []),
                    dialogue_history=context_data.get('dialogue_history', []),
                    command_history=context_data.get('command_history', []),
                    failed_commands=set(context_data.get('failed_commands', [])),
                    successful_commands=set(context_data.get('successful_commands', []))
                )
        return None

class AIBackendManager:
    """Manage different AI backends"""
    
    def __init__(self, backend: AIBackend, config: Dict[str, Any]):
        self.backend = backend
        self.config = config
        self.setup_backend()
    
    def setup_backend(self):
        """Setup the AI backend"""
        if self.backend == AIBackend.OPENAI:
            if not openai:
                raise ImportError("OpenAI library not available")
            self.client = openai.OpenAI(api_key=self.config.get('api_key'))
        elif self.backend == AIBackend.OLLAMA:
            self.base_url = self.config.get('base_url', 'http://localhost:11434')
        elif self.backend == AIBackend.CLAUDE:
            # Claude implementation would go here
            pass
        elif self.backend == AIBackend.MOCK:
            self.client = None
    
    def generate_response(self, prompt: str, context: GameContext) -> str:
        """Generate AI response based on backend"""
        try:
            if self.backend == AIBackend.OPENAI:
                return self._openai_generate(prompt, context)
            elif self.backend == AIBackend.OLLAMA:
                return self._ollama_generate(prompt, context)
            elif self.backend == AIBackend.MOCK:
                return self._mock_generate(prompt, context)
            else:
                return "look"
        except Exception as e:
            logger.error(f"AI generation failed: {e}")
            return "look"
    
    def _openai_generate(self, prompt: str, context: GameContext) -> str:
        """Generate response using OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.get('model', 'gpt-3.5-turbo'),
                messages=[
                    {
                        "role": "system",
                        "content": self._build_system_prompt(context)
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.config.get('max_tokens', 50),
                temperature=self.config.get('temperature', 0.7)
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return "look"
    
    def _ollama_generate(self, prompt: str, context: GameContext) -> str:
        """Generate response using Ollama"""
        try:
            model = self.config.get('model', 'llama2:7b')
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": self._build_system_prompt(context) + "\n\nUser: " + prompt + "\nAssistant:",
                    "stream": False
                }
            )
            if response.status_code == 200:
                return response.json()['response'].strip()
            else:
                logger.error(f"Ollama request failed: {response.status_code}")
                return "look"
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return "look"
    
    def _mock_generate(self, prompt: str, context: GameContext) -> str:
        """Generate mock response for testing"""
        commands = ["look", "north", "south", "east", "west", "examine", "inventory"]
        return random.choice(commands)
    
    def _build_system_prompt(self, context: GameContext) -> str:
        """Build a structured system prompt"""
        return f"""You are a player in a MUD game. Respond only with text that would be typed by a real player. Don't explain.

Current State:
- Location: {context.current_location}
- Health: {context.health}/{context.max_health}
- Mana: {context.mana}/{context.max_mana}
- Level: {context.level}
- Experience: {context.experience}
- Gold: {context.gold}
- Game State: {context.game_state.value}

Recent Events: {', '.join(context.last_events[-3:])}
Inventory: {', '.join(context.inventory[-5:])}
Last Commands: {', '.join(context.command_history[-3:])}

Available Commands: look, north, south, east, west, up, down, examine, inventory, get, drop, attack, cast, rest, say, tell

Rules:
1. Respond with only the command, no explanations
2. Don't repeat failed commands
3. Be contextually appropriate
4. Don't spam the same command repeatedly
5. Use short, clear commands"""

class SafetyManager:
    """Manage safety features and loop protection"""
    
    def __init__(self):
        self.command_cooldowns = {}
        self.repeated_commands = {}
        self.max_repeats = 3
        self.cooldown_time = 5.0
    
    def check_safety(self, command: str, context: GameContext) -> bool:
        """Check if command is safe to execute"""
        current_time = time.time()
        
        # Check cooldown
        if command in self.command_cooldowns:
            if current_time - self.command_cooldowns[command] < self.cooldown_time:
                logger.warning(f"Command {command} is on cooldown")
                return False
        
        # Check for repeated commands
        if command in self.repeated_commands:
            count = self.repeated_commands[command]
            if count >= self.max_repeats:
                logger.warning(f"Command {command} repeated too many times")
                return False
            self.repeated_commands[command] = count + 1
        else:
            self.repeated_commands[command] = 1
        
        # Reset repeat counter for other commands
        for cmd in self.repeated_commands:
            if cmd != command:
                self.repeated_commands[cmd] = 0
        
        return True
    
    def update_cooldown(self, command: str):
        """Update command cooldown"""
        self.command_cooldowns[command] = time.time()
    
    def reset_cooldowns(self):
        """Reset all cooldowns"""
        self.command_cooldowns.clear()
        self.repeated_commands.clear()

class EnhancedAIMUDClient:
    """Enhanced AI-powered MUD client"""
    
    def __init__(self, config_file: str):
        self.config = self._load_config(config_file)
        self.server_config = self.config['server']
        self.bot_config = self.config['bot']
        self.ai_config = self.config['ai']
        
        # Initialize components
        self.message_filter = MessageFilter()
        self.state_manager = StateManager()
        self.safety_manager = SafetyManager()
        
        # Load or create context
        self.context = self.state_manager.load_context() or GameContext()
        
        # Setup AI backend
        backend_type = AIBackend(self.ai_config.get('backend', 'openai'))
        self.ai_backend = AIBackendManager(backend_type, self.ai_config)
        
        # Connection state
        self.connected = False
        self.connection_lost = False
        self.reader = None
        self.writer = None
        
        logger.info(f"Enhanced AI MUD Client initialized for {self.server_config['name']}")
    
    def _load_config(self, config_file: str) -> dict:
        """Load configuration from file"""
        with open(config_file, 'r') as f:
            return json.load(f)
    
    async def connect(self) -> bool:
        """Connect to the MUD server"""
        try:
            logger.info(f"Connecting to {self.server_config['host']}:{self.server_config['port']}...")
            self.reader, self.writer = await telnetlib3.open_connection(
                self.server_config['host'], 
                self.server_config['port']
            )
            self.connected = True
            self.connection_lost = False
            
            # Read initial response
            response = await self.read_response()
            if response:
                logger.info("Connected successfully!")
                return True
            else:
                logger.error("Failed to read initial response")
                return False
                
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from the server"""
        if self.connected:
            try:
                self.writer.close()
                try:
                    await self.writer.wait_closed()
                except AttributeError:
                    pass
                self.connected = False
                logger.info("Disconnected from server")
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
    
    async def send_command(self, command: str):
        """Send command to server with safety checks"""
        if not self.connected:
            return
        
        # Safety check
        if not self.safety_manager.check_safety(command, self.context):
            logger.warning(f"Command {command} blocked by safety manager")
            return
        
        try:
            # Add newline if not present
            if not command.endswith('\n'):
                command += '\n'
            
            self.writer.write(command)
            await self.writer.drain()
            
            # Update context
            self.context.last_command = command.strip()
            self.context.command_history.append(command.strip())
            
            # Update cooldown
            self.safety_manager.update_cooldown(command.strip())
            
            logger.info(f"Sent command: {command.strip()}")
            print(f"🤖 AI Bot: {command.strip()}")
            
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            self.connection_lost = True
    
    async def read_response(self) -> str:
        """Read response from server"""
        if not self.connected:
            return ""
        
        try:
            data = await asyncio.wait_for(self.reader.read(1024), timeout=5.0)
            
            if isinstance(data, bytes):
                response = data.decode('utf-8', errors='ignore')
            else:
                response = str(data)
            
            if response.strip():
                # Filter and categorize message
                filtered = self.message_filter.filter_message(response)
                
                # Update context based on message type
                self._update_context_from_message(filtered)
                
                # Log the response
                logger.info(f"Server Response: {response.strip()}")
                print(f"📡 Server: {response.strip()}")
                print("-" * 50)
            
            return response
            
        except asyncio.TimeoutError:
            logger.warning("Timeout reading response")
            return ""
        except Exception as e:
            logger.error(f"Failed to read response: {e}")
            self.connection_lost = True
            return ""
    
    def _update_context_from_message(self, filtered_message: Dict[str, Any]):
        """Update context based on filtered message"""
        content = filtered_message['clean']
        
        # Update last events
        self.context.last_events.append(content)
        if len(self.context.last_events) > 10:
            self.context.last_events.pop(0)
        
        # Update based on message type
        if filtered_message['is_room']:
            # Extract location from room description
            location_match = re.search(r'^([A-Z][^.]*?)\s*\(', content)
            if location_match:
                self.context.current_location = location_match.group(1)
        
        elif filtered_message['is_dialogue']:
            self.context.dialogue_history.append(content)
            if len(self.context.dialogue_history) > 5:
                self.context.dialogue_history.pop(0)
        
        # Extract health/mana from system messages
        health_match = re.search(r'(\d+)/(\d+)\s*health', content.lower())
        if health_match:
            self.context.health = int(health_match.group(1))
            self.context.max_health = int(health_match.group(2))
        
        mana_match = re.search(r'(\d+)/(\d+)\s*mana', content.lower())
        if mana_match:
            self.context.mana = int(mana_match.group(1))
            self.context.max_mana = int(mana_match.group(2))
    
    def _check_immediate_actions(self, response_text: str) -> Optional[str]:
        """Check for immediate actions that don't need AI"""
        response_lower = response_text.lower()
        
        # Login prompts
        login_prompts = self.config.get('game_specific', {}).get('login_prompts', [])
        if any(prompt in response_lower for prompt in login_prompts):
            existing_account = self.bot_config.get('existing_account')
            if existing_account and existing_account.get('username'):
                logger.info(f"Login prompt detected, using existing account: {existing_account['username']}")
                return existing_account['username']
            else:
                logger.info("Login prompt detected, using character name...")
                return self.bot_config.get('character_name', 'player')
        
        # Password prompts
        password_prompts = self.config.get('game_specific', {}).get('password_prompts', [])
        if any(prompt in response_lower for prompt in password_prompts):
            existing_account = self.bot_config.get('existing_account')
            if existing_account and existing_account.get('password'):
                logger.info("Password prompt detected, using stored password...")
                return existing_account['password']
            else:
                logger.info("Password prompt detected, using default password...")
                return self.config.get('game_specific', {}).get('default_password', 'password123')
        
        # Welcome messages
        welcome_messages = self.config.get('game_specific', {}).get('welcome_messages', [])
        if any(msg in response_lower for msg in welcome_messages):
            self.context.game_state = GameState.EXPLORING
            return "look"
        
        return None
    
    def _generate_ai_prompt(self, response_text: str) -> str:
        """Generate a structured prompt for AI"""
        filtered = self.message_filter.filter_message(response_text)
        
        prompt = f"Game Output: {filtered['clean']}\n\n"
        prompt += f"Current State: {self.context.game_state.value}\n"
        prompt += f"Location: {self.context.current_location}\n"
        prompt += f"Health: {self.context.health}/{self.context.max_health}\n"
        prompt += f"Mana: {self.context.mana}/{self.context.max_mana}\n"
        
        if self.context.last_events:
            prompt += f"Recent Events: {', '.join(self.context.last_events[-2:])}\n"
        
        prompt += "\nWhat should I do next? Respond with only the command:"
        
        return prompt
    
    async def run_ai_mode(self, duration: int = 60):
        """Run the bot in AI mode with enhanced features"""
        logger.info(f"Starting enhanced AI mode for {duration} seconds...")
        print(f"🤖 Starting enhanced AI mode for {duration} seconds...")
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            if not self.connected or self.connection_lost:
                logger.warning("Connection lost, attempting to reconnect...")
                if not await self.connect():
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
            
            # Generate AI response
            prompt = self._generate_ai_prompt(response)
            ai_command = self.ai_backend.generate_response(prompt, self.context)
            
            if ai_command:
                await self.send_command(ai_command)
                await asyncio.sleep(2)
            else:
                # Fallback
                await self.send_command("look")
                await asyncio.sleep(1)
            
            # Save context periodically
            if int(time.time()) % 30 == 0:  # Every 30 seconds
                self.state_manager.save_context(self.context)
        
        # Final save
        self.state_manager.save_context(self.context)
        logger.info("Enhanced AI mode completed")
        print("✅ Enhanced AI mode completed")
    
    def print_context_info(self):
        """Print current context information"""
        print("\n" + "=" * 50)
        print("=== Enhanced AI Context Information ===")
        print(f"Location: {self.context.current_location}")
        print(f"Health: {self.context.health}/{self.context.max_health}")
        print(f"Mana: {self.context.mana}/{self.context.max_mana}")
        print(f"Level: {self.context.level}")
        print(f"Experience: {self.context.experience}")
        print(f"Gold: {self.context.gold}")
        print(f"Game State: {self.context.game_state.value}")
        print(f"AI Backend: {self.ai_backend.backend.value}")
        print(f"Recent Events: {len(self.context.last_events)}")
        print(f"Dialogue History: {len(self.context.dialogue_history)}")
        print(f"Command History: {len(self.context.command_history)}")
        print("=" * 50 + "\n")

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Enhanced AI MUD Client")
    parser.add_argument("--config", required=True, help="Configuration file path")
    parser.add_argument("--duration", type=int, default=60, help="Duration to run in seconds")
    parser.add_argument("--backend", default="openai", help="AI backend (openai, ollama, mock)")
    
    args = parser.parse_args()
    
    # Create client
    client = EnhancedAIMUDClient(args.config)
    
    # Override backend if specified
    if args.backend:
        client.ai_backend.backend = AIBackend(args.backend)
    
    try:
        # Connect
        if await client.connect():
            # Run AI mode
            await client.run_ai_mode(args.duration)
            
            # Print final info
            client.print_context_info()
        else:
            logger.error("Failed to connect to server")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main()) 