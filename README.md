# MUD Bot AI Client

A generic, configuration-driven MUD (Multi-User Dungeon) bot client that can connect to any MUD server and play automatically using AI-driven decision making.

## 🎯 **Features**

- **Generic Architecture**: Works with any MUD server through configuration files
- **AI-Driven Gameplay**: Intelligent decision making based on game state and learned commands
- **Self-Learning**: Bot learns from server responses and adapts its behavior
- **Character Creation**: Automated character creation and role selection
- **Multiple AI Strategies**: Explorer, Combat, Farmer, Social, and Adventurer modes
- **Comprehensive Logging**: Full conversation logging for debugging and analysis
- **Virtual Environment**: Isolated Python environment for clean dependency management
- **🤖 ChatGPT Integration**: Real AI-powered decision making using OpenAI's GPT models

## 🚀 **Quick Start**

### 1. Setup Environment
```bash
# Activate virtual environment
source venv/bin/activate

# Or create new environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run with Generic Launcher
```bash
# List available configurations
python launcher_generic.py --list-configs

# Run with specific configuration
python launcher_generic.py --config config_examples/sindome_config.json --mode auto --duration 60

# Interactive mode
python launcher_generic.py --config config_examples/sindome_config.json --mode interactive
```

### 3. Quick Launch Scripts
```bash
# Sindome server
./run_sindome_generic.sh

# Aetolia server  
./run_aetolia_generic.sh

# Achaea server
./run_achaea_generic.sh
```

## 🤖 **AI-Powered Client (ChatGPT Integration)**

### Setup AI Client
```bash
# Setup AI client with OpenAI integration
./setup_ai.sh

# Or manually:
cp env.example .env
# Edit .env and add your OpenAI API key
pip install openai
```

### Run AI-Powered Bot
```bash
# Run with ChatGPT integration
python launcher_ai.py --config config_examples/sindome_config.json --duration 60

# List available configurations
python launcher_ai.py --list-configs

# Interactive AI mode
python launcher_ai.py --config config_examples/sindome_config.json --interactive --duration 120
```

### AI Features
- **🧠 Real AI Decision Making**: Uses GPT-4 or GPT-3.5-turbo for intelligent responses
- **📚 Context Awareness**: Maintains conversation history and game state
- **🎯 Adaptive Learning**: Learns from server responses and avoids failed commands
- **⚡ Smart Fallbacks**: Falls back to basic commands if AI fails
- **🔧 Configurable AI**: Adjustable temperature, token limits, and model selection

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
OPENAI_MODEL=gpt-3.5-turbo  # or gpt-4 if you have access
AI_TEMPERATURE=0.7                # 0.0 to 1.0
AI_MAX_TOKENS=150                 # Response length limit
```

## 📁 **Project Structure**

```
mud-ia-client/
├── generic_mud_client.py      # Main generic MUD client
├── ai_mud_client.py           # AI-powered client with ChatGPT
├── launcher_generic.py        # Generic launcher with CLI
├── launcher_ai.py             # AI-powered launcher
├── test_generic_client.py     # Test script for generic client
├── config_examples/           # Server configuration files
│   ├── generic_mud_config.json
│   ├── sindome_config.json
│   └── README.md
├── run_*_generic.sh          # Quick launch scripts
├── setup_ai.sh               # AI client setup script
├── env.example               # Environment variables template
├── requirements.txt           # Python dependencies
├── mud_bot.log              # Bot activity log
└── README.md                # This file
```

## ⚙️ **Configuration**

The bot uses JSON configuration files to adapt to different MUD servers. Each config contains:

### Server Settings
```json
{
  "server": {
    "host": "moo.sindome.org",
    "port": 5555,
    "name": "Sindome"
  }
}
```

### Bot Settings
```json
{
  "bot": {
    "name": "genericbot",
    "character_name": "genericbot",
    "max_command_history": 50,
    "reconnect_attempts": 3
  }
}
```

### AI Strategy
```json
{
  "ai": {
    "strategy": "explorer",
    "weights": {
      "explore": 0.7,
      "combat": 0.1,
      "loot": 0.1,
      "rest": 0.1
    }
  }
}
```

### Game-Specific Commands and Patterns
```json
{
  "game_specific": {
    "movement_commands": ["north", "south", "east", "west"],
    "special_commands": ["look", "inventory", "examine"],
    "login_prompts": ["login:", "username:", "password:"],
    "welcome_messages": ["welcome", "logged in"],
    "patterns": {
      "room": "You are in (.+)",
      "health": "Health: (\\d+)/(\\d+)",
      "mana": "Mana: (\\d+)/(\\d+)"
    }
  }
}
```

## 🧠 **AI Features**

### Self-Learning System
- **Command Learning**: Bot learns valid commands from server error messages
- **Failed Command Tracking**: Avoids repeating failed commands
- **Hint Parsing**: Extracts command suggestions from server responses
- **Cooldown System**: Prevents command spamming

### AI Strategies
- **Explorer**: Focuses on movement and discovery
- **Combat Focused**: Prioritizes combat and fighting
- **Farmer**: Focuses on resource gathering
- **Social**: Emphasizes interaction with NPCs
- **Adventurer**: Balanced approach to all activities

### State Management
- **Game States**: Connecting, Login, Playing, Combat, Inventory, etc.
- **Character States**: Health, mana, experience, location tracking
- **Learning States**: Command success/failure tracking

## 🔧 **Usage Examples**

### Basic Auto Mode
```bash
python launcher_generic.py --config config_examples/sindome_config.json --mode auto --duration 120
```

### Interactive Mode
```bash
python launcher_generic.py --config config_examples/sindome_config.json --mode interactive
```

### Test Different Configurations
```bash
python test_generic_client.py
```

## 📊 **Logging and Monitoring**

The bot provides comprehensive logging:

- **Console Output**: Real-time bot commands and server responses
- **File Logging**: Detailed logs in `mud_bot.log`
- **Character Information**: Stats, location, and game state
- **Learning Progress**: Command learning and adaptation

## 🎮 **Supported MUD Servers**

The generic client can work with any MUD server through configuration:

- **Sindome** (`moo.sindome.org:5555`) - Cyberpunk MUD
- **Aetolia** (`aetolia.com:23`) - Fantasy MUD  
- **Achaea** (`achaea.com:23`) - Fantasy MUD

## 🔄 **Development Workflow**

1. **Create Configuration**: Add new server config in `config_examples/`
2. **Test Connection**: Use interactive mode to verify connection
3. **Tune Commands**: Adjust game-specific commands and patterns
4. **Run Auto Mode**: Test AI behavior and learning
5. **Monitor Logs**: Analyze bot performance and adapt

## 🛠️ **Technical Details**

- **Async I/O**: Uses `asyncio` and `telnetlib3` for non-blocking connections
- **State Machine**: Robust state management for game flow
- **Regex Parsing**: Configurable pattern matching for game responses
- **Virtual Environment**: Isolated Python environment
- **Configuration-Driven**: No hardcoded server-specific logic

## 📝 **License**

This project is for educational and research purposes. Please respect the terms of service of any MUD servers you connect to. 