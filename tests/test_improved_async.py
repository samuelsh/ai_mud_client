import pytest
import asyncio
import json
import time
from unittest.mock import patch, AsyncMock, Mock
from improved_async_client import (
    GameContext, GameState, PromptEngineer, AsyncAIBackend,
    ConnectionManager, CommandValidator, ImprovedMUDClient
)

class TestGameContext:
    def test_default_initialization(self):
        """Test default context initialization"""
        context = GameContext()
        
        assert context.current_location == ""
        assert context.health == 100
        assert context.max_health == 100
        assert context.mana == 100
        assert context.max_mana == 100
        assert context.game_state == GameState.CONNECTING
        assert context.last_command == ""
        assert context.command_history == []
        assert context.failed_commands == set()
        assert context.successful_commands == set()
    
    def test_custom_initialization(self):
        """Test custom context initialization"""
        context = GameContext(
            current_location="Test Room",
            health=80,
            game_state=GameState.EXPLORING
        )
        
        assert context.current_location == "Test Room"
        assert context.health == 80
        assert context.game_state == GameState.EXPLORING

class TestPromptEngineer:
    def setup_method(self):
        self.engineer = PromptEngineer()
    
    def test_command_only_prompt(self):
        """Test command-only prompt generation"""
        context = GameContext(
            current_location="Test Room",
            game_state=GameState.EXPLORING
        )
        game_output = "You see exits: north, south"
        
        prompt = self.engineer.build_command_only_prompt(context, game_output)
        
        assert "CONTEXT:" in prompt
        assert "GAME OUTPUT:" in prompt
        assert "VALID COMMANDS:" in prompt
        assert "north" in prompt
        assert "south" in prompt
    
    def test_get_valid_commands_exploring(self):
        """Test valid commands for exploring state"""
        context = GameContext(game_state=GameState.EXPLORING)
        
        commands = self.engineer._get_valid_commands(context)
        
        assert "look" in commands
        assert "north" in commands
        assert "south" in commands
        assert "east" in commands
        assert "west" in commands
    
    def test_get_valid_commands_combat(self):
        """Test valid commands for combat state"""
        context = GameContext(game_state=GameState.COMBAT)
        
        commands = self.engineer._get_valid_commands(context)
        
        assert "attack" in commands
        assert "flee" in commands
    
    def test_get_valid_commands_dialogue(self):
        """Test valid commands for dialogue state"""
        context = GameContext(game_state=GameState.DIALOGUE)
        
        commands = self.engineer._get_valid_commands(context)
        
        assert "say" in commands
        assert "tell" in commands

class TestAsyncAIBackend:
    @pytest.fixture
    def config(self):
        return {
            'backend': 'mock',
            'base_url': 'http://localhost:11434',
            'model': 'test-model'
        }
    
    @pytest.fixture
    def context(self):
        return GameContext(
            current_location='Test Room',
            game_state=GameState.EXPLORING
        )
    
    def test_mock_generation(self, config, context):
        """Test mock generation"""
        backend = AsyncAIBackend(config)
        
        response = backend._mock_generate("test prompt", context)
        
        assert response in ['look', 'north', 'south', 'east', 'west', 'examine', 'inventory']
    
    @pytest.mark.asyncio
    async def test_async_generation_timeout(self, config, context):
        """Test async generation with timeout"""
        backend = AsyncAIBackend(config)
        
        # Mock a slow generation that actually sleeps
        def slow_generation(*args):
            import time
            time.sleep(15)  # This will cause the async wrapper to timeout
            return "slow response"
        
        with patch.object(backend, '_generate_sync', side_effect=slow_generation):
            response = await backend.generate_response_async("test prompt", context)
        
        assert response == "look"  # Fallback due to timeout
    
    @pytest.mark.asyncio
    async def test_async_generation_success(self, config, context):
        """Test successful async generation"""
        backend = AsyncAIBackend(config)
        
        response = await backend.generate_response_async("test prompt", context)
        
        assert response in ['look', 'north', 'south', 'east', 'west', 'examine', 'inventory']
    
    @pytest.mark.asyncio
    async def test_ollama_generation_sync(self, config, context):
        """Test Ollama sync generation"""
        config['backend'] = 'ollama'
        backend = AsyncAIBackend(config)
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {'response': 'look'}
            
            response = backend._ollama_generate_sync("test prompt", context)
            
            assert response == "look"
            mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ollama_generation_failure(self, config, context):
        """Test Ollama generation failure"""
        config['backend'] = 'ollama'
        backend = AsyncAIBackend(config)
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 500
            
            response = backend._ollama_generate_sync("test prompt", context)
            
            assert response == "look"  # Fallback

class TestConnectionManager:
    @pytest.fixture
    def manager(self):
        return ConnectionManager("test.host", 23)
    
    @pytest.mark.asyncio
    async def test_connect_with_backoff_success(self, manager):
        """Test successful connection with backoff"""
        with patch('telnetlib3.open_connection') as mock_connect:
            mock_connect.return_value = (Mock(), Mock())
            
            result = await manager.connect_with_backoff()
            
            assert result == True
            assert manager.connected == True
            assert manager.retry_count == 0
    
    @pytest.mark.asyncio
    async def test_connect_with_backoff_failure(self, manager):
        """Test connection failure with backoff"""
        with patch('telnetlib3.open_connection', side_effect=Exception("Connection failed")):
            result = await manager.connect_with_backoff()
            
            assert result == False
            assert manager.connected == False
            assert manager.retry_count == manager.max_retries
    
    @pytest.mark.asyncio
    async def test_send_command_success(self, manager):
        """Test successful command sending"""
        manager.connected = True
        manager.writer = AsyncMock()
        manager.writer.write = AsyncMock()
        manager.writer.drain = AsyncMock()
        
        result = await manager.send_command("look")
        
        assert result is True
        manager.writer.write.assert_called_once_with("look\n")
        manager.writer.drain.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_command_failure(self, manager):
        """Test command sending failure"""
        manager.connected = True
        manager.writer = AsyncMock()
        manager.writer.write = AsyncMock(side_effect=Exception("Write failed"))
        
        result = await manager.send_command("look")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_read_response_success(self, manager):
        """Test successful response reading"""
        manager.connected = True
        manager.reader = AsyncMock()
        manager.reader.read.return_value = b"test response\n"
        
        response = await manager.read_response()
        
        assert response == "test response\n"
    
    @pytest.mark.asyncio
    async def test_read_response_timeout(self, manager):
        """Test response reading timeout"""
        manager.connected = True
        manager.reader = AsyncMock()
        manager.reader.read.side_effect = asyncio.TimeoutError()
        
        response = await manager.read_response()
        
        assert response is None

class TestCommandValidator:
    def setup_method(self):
        self.validator = CommandValidator(['look', 'north', 'south', 'east', 'west'])
    
    def test_validate_command_valid(self):
        """Test valid command validation"""
        is_valid, command = self.validator.validate_command("look")
        
        assert is_valid == True
        assert command == "look"
    
    def test_validate_command_invalid(self):
        """Test invalid command validation"""
        is_valid, command = self.validator.validate_command("invalid")
        
        assert is_valid == False
        assert command == "invalid"
    
    def test_validate_command_empty(self):
        """Test empty command validation"""
        is_valid, command = self.validator.validate_command("")
        
        assert is_valid == False
        assert command == ""
    
    def test_validate_command_with_whitespace(self):
        """Test command validation with whitespace"""
        is_valid, command = self.validator.validate_command("  look  ")
        
        assert is_valid == True
        assert command == "look"
    
    def test_suggest_alternative(self):
        """Test command alternative suggestion"""
        alternative = self.validator.suggest_alternative("invalid")
        
        assert alternative in ['look', 'north', 'south', 'east', 'west']

class TestImprovedMUDClient:
    @pytest.fixture
    def config(self):
        return {
            'server': {
                'host': 'test.host',
                'port': 23
            },
            'ai': {
                'backend': 'mock',
                'model': 'test-model'
            },
            'game_specific': {
                'movement_commands': ['north', 'south', 'east', 'west', 'look']
            }
        }
    
    @pytest.fixture
    def client(self, config):
        return ImprovedMUDClient(config)
    
    def test_client_initialization(self, client):
        """Test client initialization"""
        assert client.context is not None
        assert client.ai_backend is not None
        assert client.prompt_engineer is not None
        assert client.command_validator is not None
        assert client.connection_manager is not None
    
    @pytest.mark.asyncio
    async def test_process_response_exploring(self, client):
        """Test response processing for exploring state"""
        response = "You see exits: north, south"
        
        await client._process_response(response)
        
        assert client.context.game_state == GameState.EXPLORING
    
    @pytest.mark.asyncio
    async def test_process_response_combat(self, client):
        """Test response processing for combat state"""
        response = "Your health is low and you are bleeding"
        
        await client._process_response(response)
        
        assert client.context.game_state == GameState.COMBAT
    
    @pytest.mark.asyncio
    async def test_process_response_location_extraction(self, client):
        """Test location extraction from response"""
        response = "High Street Terminal (Cybersphere)\nYou see exits: north, south"
        
        await client._process_response(response)
        
        assert client.context.current_location == "High Street Terminal"
    
    @pytest.mark.asyncio
    async def test_ai_worker(self, client):
        """Test AI worker functionality"""
        # Add a test prompt to the queue
        await client._ai_queue.put("test prompt")
        await client._ai_queue.put(None)  # Shutdown signal
        
        # Run the worker
        await client._ai_worker()
        
        # Check that a response was generated
        response = await client._response_queue.get()
        assert response in ['look', 'north', 'south', 'east', 'west', 'examine', 'inventory']
    
    @pytest.mark.asyncio
    async def test_save_context_async(self, client):
        """Test async context saving"""
        client.context.current_location = "Test Room"
        client.context.health = 80
        client.context.command_history = ["look", "north"]
        
        await client._save_context_async()
        
        # Check that file was created
        with open('game_context.json', 'r') as f:
            data = json.load(f)
        
        assert data['location'] == "Test Room"
        assert data['health'] == 80
        assert data['command_history'] == ["look", "north"]
    
    @pytest.mark.asyncio
    async def test_run_with_mock_connection(self, client):
        """Test main run loop with mocked connection"""
        # Mock connection manager to avoid real connection attempts
        client.connection_manager.connected = True
        client.connection_manager.read_response = AsyncMock()
        client.connection_manager.send_command = AsyncMock(return_value=True)
        
        # Mock the connect_with_backoff method to return True
        client.connection_manager.connect_with_backoff = AsyncMock(return_value=True)
        
        # Mock AI backend to return commands
        client.ai_backend.generate_response_async = AsyncMock(return_value="look")
        
        # Bypass rate limiting
        client.min_ai_interval = 0.0
        client.last_ai_call = 0
        
        # Create a simple response generator that provides more responses
        responses = ["Welcome to the game!", "You see exits: north, south", "You are in a room.", "You see exits: east, west"]
        response_index = 0
        
        async def mock_read_response():
            nonlocal response_index
            if response_index < len(responses):
                response = responses[response_index]
                response_index += 1
                return response
            # Return a default response instead of None to keep the loop going
            return "You are in a room."
        
        client.connection_manager.read_response = mock_read_response
        
        # Mock the AI worker to stay alive and process prompts
        async def mock_ai_worker():
            print("AI worker started!")
            while True:
                try:
                    prompt = await client._ai_queue.get()
                    print(f"AI worker received prompt: {prompt[:50]}...")
                    if prompt is None:  # Shutdown signal
                        break
                    # Put response in queue immediately
                    await client._response_queue.put("look")
                    print("AI worker put response in queue")
                except Exception as e:
                    print(f"AI worker exception: {e}")
                    break
        
        # Replace the AI worker
        client._ai_worker = mock_ai_worker
        
        # Run for a longer duration to ensure processing
        await client.run(duration=3)
        
        # Print debug info
        print(f"AI queue size: {client._ai_queue.qsize()}")
        print(f"Response queue size: {client._response_queue.qsize()}")
        
        # Verify commands were sent
        assert client.connection_manager.send_command.called
    
    def test_print_context_info(self, client, capsys):
        """Test context info printing"""
        client.context.current_location = "Test Room"
        client.context.health = 80
        
        client.print_context_info()
        
        captured = capsys.readouterr()
        assert "Test Room" in captured.out
        assert "80" in captured.out

class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_message_processing_cycle(self):
        """Test full message processing cycle"""
        config = {
            'server': {'host': 'test.host', 'port': 23},
            'ai': {'backend': 'mock'},
            'game_specific': {'movement_commands': ['look', 'north']}
        }
        
        client = ImprovedMUDClient(config)
        
        # Mock AI backend
        client.ai_backend.generate_response_async = AsyncMock(return_value="look")
        
        # Test processing a response
        response = "You see exits: north, south"
        await client._process_response(response)
        
        assert client.context.game_state == GameState.EXPLORING
    
    @pytest.mark.asyncio
    async def test_command_validation_integration(self):
        """Test command validation integration"""
        validator = CommandValidator(['look', 'north', 'south'])
        
        # Test valid command
        is_valid, command = validator.validate_command("look")
        assert is_valid == True
        
        # Test invalid command
        is_valid, command = validator.validate_command("invalid")
        assert is_valid == False
        
        # Test alternative suggestion
        alternative = validator.suggest_alternative("invalid")
        assert alternative in ['look', 'north', 'south'] 