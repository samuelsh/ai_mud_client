# OpenAI Models Guide for AI-Powered MUD Client

## 🤖 **Available Models**

### **GPT-3.5 Models (Recommended)**
- **`gpt-3.5-turbo`** - Fast, cost-effective, good for most use cases
- **`gpt-3.5-turbo-16k`** - Same as turbo but with 16k context window
- **`gpt-3.5-turbo-instruct`** - Optimized for instruction following

### **GPT-4 Models (Premium)**
- **`gpt-4`** - Most capable model, requires GPT-4 access
- **`gpt-4-turbo`** - Faster and cheaper than GPT-4
- **`gpt-4-32k`** - GPT-4 with 32k context window

### **Legacy Models (Not Recommended)**
- **`text-davinci-003`** - Legacy model, more expensive
- **`text-curie-001`** - Legacy model, limited capabilities

## ⚙️ **Configuration**

### **Environment Variable**
```bash
# Set in .env file or environment
OPENAI_MODEL=gpt-3.5-turbo
```

### **Model Selection Guide**

| Use Case | Recommended Model | Cost | Speed | Capability |
|----------|------------------|------|-------|------------|
| **Testing/Development** | `gpt-3.5-turbo` | Low | Fast | Good |
| **Production Gaming** | `gpt-3.5-turbo` | Low | Fast | Good |
| **Complex MUDs** | `gpt-4` | High | Slow | Excellent |
| **Budget Conscious** | `gpt-3.5-turbo` | Low | Fast | Good |

## 💰 **Cost Comparison**

### **GPT-3.5-turbo**
- **Input**: $0.0015 per 1K tokens
- **Output**: $0.002 per 1K tokens
- **Typical MUD session**: ~$0.01-0.05 per hour

### **GPT-4**
- **Input**: $0.03 per 1K tokens
- **Output**: $0.06 per 1K tokens
- **Typical MUD session**: ~$0.10-0.30 per hour

## 🚀 **Quick Setup**

### **1. Get OpenAI API Key**
```bash
# Visit https://platform.openai.com/api-keys
# Create a new API key
```

### **2. Set Environment Variables**
```bash
# Copy example file
cp env.example .env

# Edit .env file
echo "OPENAI_API_KEY=your_actual_api_key_here" >> .env
echo "OPENAI_MODEL=gpt-3.5-turbo" >> .env
```

### **3. Test Configuration**
```bash
# Run the AI client
python launcher_ai.py --config config_examples/sindome_config.json --duration 30
```

## 🔧 **Model-Specific Settings**

### **GPT-3.5-turbo (Recommended)**
```bash
OPENAI_MODEL=gpt-3.5-turbo
AI_TEMPERATURE=0.7
AI_MAX_TOKENS=150
```

### **GPT-4 (Premium)**
```bash
OPENAI_MODEL=gpt-4
AI_TEMPERATURE=0.5
AI_MAX_TOKENS=200
```

### **GPT-4-turbo (Latest)**
```bash
OPENAI_MODEL=gpt-4-turbo
AI_TEMPERATURE=0.6
AI_MAX_TOKENS=200
```

## ⚠️ **Common Issues**

### **Model Not Found Error**
```
Error: The model `gpt-4-turbo-preview` does not exist
```
**Solution**: Use `gpt-3.5-turbo` or `gpt-4` instead

### **Access Denied Error**
```
Error: You do not have access to this model
```
**Solution**: 
1. Check your OpenAI account has access to the model
2. Use `gpt-3.5-turbo` which is available to all users
3. Upgrade your OpenAI plan if needed

### **Rate Limit Error**
```
Error: Rate limit exceeded
```
**Solution**: 
1. Reduce request frequency
2. Use a model with higher rate limits
3. Implement exponential backoff

## 🎯 **Best Practices**

1. **Start with GPT-3.5-turbo** - It's fast, cheap, and capable
2. **Monitor costs** - Set up billing alerts on OpenAI
3. **Test thoroughly** - Use short sessions first
4. **Have fallbacks** - The client works without API access
5. **Respect rate limits** - Don't spam requests

## 📊 **Performance Tips**

- **Lower temperature** (0.3-0.5) for more consistent responses
- **Higher temperature** (0.7-0.9) for more creative responses
- **Adjust max_tokens** based on your needs
- **Use conversation history** for better context

## 🔒 **Security Notes**

- **Never commit API keys** to version control
- **Use environment variables** for sensitive data
- **Rotate API keys** regularly
- **Monitor usage** to prevent unexpected charges 