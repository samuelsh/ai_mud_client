#!/usr/bin/env python3
"""
Improved Async AI MUD Client
Implements architectural improvements for better performance and modularity
"""

import asyncio
import json
import logging
import time
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import argparse
import telnetlib3

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameState(Enum):
    CONNECTING = "connecting"
    LOGIN = "login"
    EXPLORING = "exploring"
    COMBAT = "combat"
    DIALOGUE = "dialogue"

@dataclass
class GameContext:
    current_location: str = ""
    health: int = 100
    max_health: int = 100
    mana: int = 100
    max_mana: int = 100
    game_state: GameState = GameState.CONNECTING
    last_command: str = ""
    command_history: List[str] = None
    failed_commands: set = None
    successful_commands: set = None
    
    def __post_init__(self):
        if self.command_history is None:
            self.command_history = []
        if self.failed_commands is None:
            self.failed_commands = set()
        if self.successful_commands is None:
            self.successful_commands = set()

class PromptEngineer:
    """Improved prompt engineering for command-only output"""
    
    def __init__(self):
        self.command_templates = {
            'movement': ['north', 'south', 'east', 'west', 'up', 'down', 'n', 's', 'e', 'w', 'u', 'd'],
            'action': ['look', 'examine', 'get', 'drop', 'inventory', 'status'],
            'combat': ['attack', 'cast', 'flee', 'rest'],
            'social': ['say', 'tell', 'emote', 'who']
        }
    
    def build_command_only_prompt(self, context: GameContext, game_output: str) -> str:
        """Build prompt that forces command-only output"""
        valid_commands = self._get_valid_commands(context)
        
        return f"""You are a MUD player. Respond with ONLY a valid command, no explanations.

CONTEXT:
Location: {context.current_location}
Health: {context.health}/{context.max_health}
State: {context.game_state.value}
Last Command: {context.last_command}

GAME OUTPUT:
{game_output}

VALID COMMANDS: {', '.join(valid_commands)}

RESPOND WITH ONLY THE COMMAND:"""
    
    def _get_valid_commands(self, context: GameContext) -> List[str]:
        """Get contextually appropriate commands"""
        commands = []
        
        # Always available
        commands.extend(self.command_templates['action'])
        
        # State-specific commands
        if context.game_state == GameState.EXPLORING:
            commands.extend(self.command_templates['movement'])
        elif context.game_state == GameState.COMBAT:
            commands.extend(self.command_templates['combat'])
        elif context.game_state == GameState.DIALOGUE:
            commands.extend(self.command_templates['social'])
        
        return commands

class AsyncAIBackend:
    """Non-blocking AI backend with thread pool"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend_type = config.get('backend', 'mock')
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.session = None
        
        if self.backend_type == 'ollama':
            self.session = aiohttp.ClientSession()
    
    async def generate_response_async(self, prompt: str, context: GameContext) -> str:
        """Non-blocking AI response generation"""
        try:
            # Submit to thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(
                self.executor, 
                self._generate_sync, 
                prompt, 
                context
            )
            
            # Add timeout
            result = await asyncio.wait_for(future, timeout=10.0)
            return result
        except asyncio.TimeoutError:
            logger.warning("AI generation timed out, using fallback")
            return "look"
        except Exception as e:
            logger.error(f"AI generation failed: {e}")
            return "look"
    
    def _generate_sync(self, prompt: str, context: GameContext) -> str:
        """Synchronous AI generation (runs in thread pool)"""
        if self.backend_type == 'mock':
            return self._mock_generate(prompt, context)
        elif self.backend_type == 'ollama':
            return self._ollama_generate_sync(prompt, context)
        else:
            return "look"
    
    def _mock_generate(self, prompt: str, context: GameContext) -> str:
        """Mock AI generation for testing"""
        import random
        commands = ['look', 'north', 'south', 'east', 'west', 'examine', 'inventory']
        return random.choice(commands)
    
    def _ollama_generate_sync(self, prompt: str, context: GameContext) -> str:
        """Synchronous Ollama generation"""
        import requests
        try:
            payload = {
                "model": self.config.get('model', 'llama2:7b'),
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 50,
                    "stop": ["\n", "Game:", "CONTEXT:", "EXAMPLES:"]
                }
            }
            
            response = requests.post(
                f"{self.config.get('base_url', 'http://localhost:11434')}/api/generate",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['response'].strip()
            else:
                logger.error(f"Ollama request failed: {response.status_code}")
                return "look"
        
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return "look"
    
    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
        if self.executor:
            self.executor.shutdown(wait=False)

class ConnectionManager:
    """Manages connection with exponential backoff"""
    
    def __init__(self, host: str, port: int, max_retries: int = 5, base_delay: float = 1.0):
        self.host = host
        self.port = port
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.retry_count = 0
        self.reader = None
        self.writer = None
        self.connected = False
    
    async def connect_with_backoff(self) -> bool:
        """Connect with exponential backoff"""
        while self.retry_count < self.max_retries:
            try:
                logger.info(f"Connecting to {self.host}:{self.port} (attempt {self.retry_count + 1})")
                self.reader, self.writer = await telnetlib3.open_connection(self.host, self.port)
                self.connected = True
                self.retry_count = 0  # Reset on success
                logger.info("Connected successfully!")
                return True
            except Exception as e:
                logger.warning(f"Connection attempt {self.retry_count + 1} failed: {e}")
            
            # Exponential backoff
            delay = self.base_delay * (2 ** self.retry_count)
            await asyncio.sleep(delay)
            self.retry_count += 1
        
        logger.error("Max connection retries exceeded")
        return False
    
    async def send_command(self, command: str) -> bool:
        """Send command to server"""
        if not self.connected:
            return False
        
        try:
            # Check if writer is valid
            if self.writer is None:
                logger.error("Writer is None, cannot send command")
                return False
            
            # Ensure command is a string
            if command is None:
                logger.error("Cannot send None command")
                return False
            
            command = str(command).strip()
            if not command:
                logger.error("Cannot send empty command")
                return False
            
            if not command.endswith('\n'):
                command += '\n'
            
            # Check if writer is still valid
            if hasattr(self.writer, 'write'):
                # Check if write method is async
                import inspect
                if inspect.iscoroutinefunction(self.writer.write):
                    await self.writer.write(command)
                else:
                    self.writer.write(command)
            else:
                logger.error("Writer does not have write method")
                return False
            
            await self.writer.drain()
            logger.info(f"Sent command: {command.strip()}")
            return True
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            self.connected = False
            return False
    
    async def read_response(self) -> Optional[str]:
        """Read response from server"""
        if not self.connected:
            return None
        
        try:
            data = await asyncio.wait_for(self.reader.read(1024), timeout=5.0)
            
            if isinstance(data, bytes):
                response = data.decode('utf-8', errors='ignore')
            else:
                response = str(data)
            
            if response.strip():
                logger.info(f"Server Response: {response.strip()}")
                return response
            
        except asyncio.TimeoutError:
            logger.warning("Timeout reading response")
            return None
        except Exception as e:
            logger.error(f"Failed to read response: {e}")
            self.connected = False
            return None
    
    async def disconnect(self):
        """Disconnect from server"""
        if self.writer:
            try:
                self.writer.close()
                try:
                    await self.writer.wait_closed()
                except AttributeError:
                    pass
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
        
        self.connected = False
        logger.info("Disconnected from server")

class CommandValidator:
    """Validates and sanitizes commands"""
    
    def __init__(self, valid_commands: List[str]):
        self.valid_commands = set(valid_commands)
    
    def validate_command(self, command: str) -> Tuple[bool, str]:
        """Validate command and return (is_valid, sanitized_command)"""
        if not command:
            return False, ""
        
        # Clean command
        clean_command = command.strip().lower()
        
        # Check if command is valid
        if clean_command in self.valid_commands:
            return True, clean_command
        
        # Try to find similar command
        for valid_cmd in self.valid_commands:
            if clean_command in valid_cmd or valid_cmd in clean_command:
                return True, valid_cmd
        
        return False, clean_command
    
    def suggest_alternative(self, invalid_command: str) -> Optional[str]:
        """Suggest alternative for invalid command"""
        # Simple fallback to common commands
        fallbacks = ['look', 'north', 'south', 'east', 'west']
        for fallback in fallbacks:
            if fallback in self.valid_commands:
                return fallback
        return None

class ImprovedMUDClient:
    """Improved async MUD client with better architecture"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.server_config = config['server']
        self.ai_config = config['ai']
        
        # Initialize components
        self.connection_manager = ConnectionManager(
            self.server_config['host'], 
            self.server_config['port']
        )
        self.ai_backend = AsyncAIBackend(self.ai_config)
        self.prompt_engineer = PromptEngineer()
        
        # Get all valid commands from config
        game_specific = self.config.get('game_specific', {})
        all_commands = []
        all_commands.extend(game_specific.get('movement_commands', []))
        all_commands.extend(game_specific.get('combat_commands', []))
        all_commands.extend(game_specific.get('inventory_commands', []))
        all_commands.extend(game_specific.get('healing_commands', []))
        all_commands.extend(game_specific.get('social_commands', []))
        all_commands.extend(game_specific.get('system_commands', []))
        all_commands.extend(game_specific.get('special_commands', []))
        
        self.command_validator = CommandValidator(all_commands)
        
        # State
        self.context = GameContext()
        self.last_ai_call = 0
        self.min_ai_interval = 2.0  # Minimum seconds between AI calls
        
        # Queues for async processing
        self._ai_queue = asyncio.Queue()
        self._response_queue = asyncio.Queue()
    
    async def _ai_worker(self):
        """Dedicated AI processing worker"""
        while True:
            try:
                prompt = await self._ai_queue.get()
                if prompt is None:  # Shutdown signal
                    break
                
                response = await self.ai_backend.generate_response_async(
                    prompt, self.context
                )
                await self._response_queue.put(response)
            except Exception as e:
                logger.error(f"AI worker error: {e}")
                await self._response_queue.put("look")
    
    async def _process_response(self, response: str):
        """Process server response and update context"""
        # Update context based on response
        if "health" in response.lower() and "low" in response.lower():
            self.context.game_state = GameState.COMBAT
        elif "exits" in response.lower():
            self.context.game_state = GameState.EXPLORING
        
        # Extract location if present
        if "(" in response and ")" in response:
            location_match = response.split("(")[0].strip()
            if location_match:
                self.context.current_location = location_match
    
    async def _handle_reconnection(self):
        """Handle reconnection logic"""
        logger.warning("Connection lost, attempting to reconnect...")
        if await self.connection_manager.connect_with_backoff():
            logger.info("Reconnected successfully")
        else:
            logger.error("Failed to reconnect")
    
    async def _save_context_async(self):
        """Async context saving"""
        try:
            # Simple context saving for now
            context_data = {
                'location': self.context.current_location,
                'health': self.context.health,
                'game_state': self.context.game_state.value,
                'command_history': self.context.command_history[-10:],  # Keep last 10
                'timestamp': time.time()
            }
            
            with open('game_context.json', 'w') as f:
                json.dump(context_data, f)
            
            logger.debug("Context saved")
        except Exception as e:
            logger.error(f"Failed to save context: {e}")
    
    async def run(self, duration: int = 60):
        """Main run loop with improved async architecture"""
        logger.info(f"Starting improved async MUD client for {duration} seconds...")
        
        # Connect
        if not await self.connection_manager.connect_with_backoff():
            logger.error("Failed to connect to server")
            return
        
        # Start AI worker
        ai_task = asyncio.create_task(self._ai_worker())
        
        try:
            start_time = time.time()
            while time.time() - start_time < duration:
                try:
                    if not self.connection_manager.connected:
                        await self._handle_reconnection()
                        continue
                    
                    # Read with timeout
                    response = await asyncio.wait_for(
                        self.connection_manager.read_response(), 
                        timeout=5.0
                    )
                    
                    if not response:
                        continue  # Continue polling instead of sleeping
                    
                    # Process response
                    await self._process_response(response)
                    
                    # Rate limit AI calls
                    current_time = time.time()
                    if current_time - self.last_ai_call >= self.min_ai_interval:
                        prompt = self.prompt_engineer.build_command_only_prompt(
                            self.context, response
                        )
                        await self._ai_queue.put(prompt)
                        self.last_ai_call = current_time
                    
                    # Process AI responses - Use proper async polling
                    try:
                        # Process all available responses
                        while not self._response_queue.empty():
                            ai_command = self._response_queue.get_nowait()
                            
                            # Skip None or empty commands
                            if ai_command is None or not ai_command.strip():
                                logger.warning(f"Skipping empty command: {ai_command}")
                                continue
                            
                            # Validate command
                            is_valid, sanitized_command = self.command_validator.validate_command(ai_command)
                            if is_valid:
                                await self.connection_manager.send_command(sanitized_command)
                                self.context.last_command = sanitized_command
                                self.context.command_history.append(sanitized_command)
                                self.context.successful_commands.add(sanitized_command)
                            else:
                                logger.warning(f"Invalid command generated: {ai_command}")
                                # Try alternative
                                alternative = self.command_validator.suggest_alternative(ai_command)
                                if alternative:
                                    await self.connection_manager.send_command(alternative)
                                    self.context.last_command = alternative
                                    self.context.command_history.append(alternative)
                                    self.context.successful_commands.add(alternative)
                                else:
                                    # Fallback
                                    await self.connection_manager.send_command("look")
                                    self.context.last_command = "look"
                                    self.context.command_history.append("look")
                                    self.context.successful_commands.add("look")
                    
                    except asyncio.QueueEmpty:
                        pass  # No AI response ready - continue polling
                    
                    # Small delay to prevent overwhelming the loop
                    await asyncio.sleep(0.01)
                
                except Exception as e:
                    logger.error(f"Error in main loop iteration: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    await asyncio.sleep(1)  # Brief pause before retry
        
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            # Cleanup
            await self._ai_queue.put(None)  # Shutdown signal
            await ai_task
            await self.ai_backend.close()
            await self.connection_manager.disconnect()
        
        logger.info("Improved async MUD client completed")
    
    def print_context_info(self):
        """Print current context information"""
        print("\n" + "=" * 50)
        print("=== Improved Async Context Information ===")
        print(f"Location: {self.context.current_location}")
        print(f"Health: {self.context.health}/{self.context.max_health}")
        print(f"Game State: {self.context.game_state.value}")
        print(f"Last Command: {self.context.last_command}")
        print(f"Command History: {len(self.context.command_history)}")
        print(f"Successful Commands: {len(self.context.successful_commands)}")
        print(f"Failed Commands: {len(self.context.failed_commands)}")
        print("=" * 50 + "\n")

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Improved Async AI MUD Client")
    parser.add_argument("--config", required=True, help="Configuration file path")
    parser.add_argument("--duration", type=int, default=60, help="Duration to run in seconds")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create client
    client = ImprovedMUDClient(config)
    
    try:
        # Run client
        await client.run(args.duration)
        
        # Print final info
        client.print_context_info()
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await client.connection_manager.disconnect()

if __name__ == "__main__":
    asyncio.run(main()) 