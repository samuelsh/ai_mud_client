# Configuration Examples

This directory contains configuration files for different MUD servers that work with the generic MUD client.

## Available Configurations

### `generic_mud_config.json`
A template configuration that can be adapted for any MUD server. Contains all the necessary sections:
- Server connection settings
- Bot behavior configuration  
- AI strategy and weights
- Game-specific commands and patterns

### `sindome_config.json`
Configuration for the Sindome MUD server (`moo.sindome.org:5555`).
- Cyberpunk-themed MUD
- Guest account system
- Role-based character creation
- Comprehensive command learning

## Creating New Configurations

To create a configuration for a new MUD server:

1. **Copy the generic template:**
   ```bash
   cp generic_mud_config.json my_server_config.json
   ```

2. **Update server settings:**
   ```json
   {
     "server": {
       "host": "your.mud.server.com",
       "port": 4000,
       "name": "Your MUD Server"
     }
   }
   ```

3. **Configure game-specific settings:**
   - Add movement commands
   - Define login prompts
   - Set welcome messages
   - Configure response patterns

4. **Test the configuration:**
   ```bash
   python launcher_generic.py --config config_examples/my_server_config.json --mode interactive
   ```

## Configuration Structure

Each configuration file contains these sections:

### Server Section
```json
{
  "server": {
    "host": "server.address.com",
    "port": 4000,
    "name": "Server Name"
  }
}
```

### Bot Section
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

### AI Section
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

### Game-Specific Section
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

## Best Practices

1. **Start with the generic template** - It contains all necessary sections
2. **Test incrementally** - Start with basic connection, then add commands
3. **Use interactive mode** - Test commands manually before auto mode
4. **Monitor logs** - Check `mud_bot.log` for learning progress
5. **Adapt patterns** - Adjust regex patterns to match server responses

## Troubleshooting

### Connection Issues
- Verify server address and port
- Check if server requires specific telnet options
- Test with `telnet` command first

### Command Learning Issues
- Check if commands are being learned from server hints
- Verify patterns match server response format
- Use interactive mode to test commands manually

### AI Behavior Issues
- Adjust AI strategy weights
- Modify command lists in game_specific section
- Check for failed command tracking 