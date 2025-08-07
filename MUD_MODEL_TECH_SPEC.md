# MUD AI Model Technical Specification

## Model Architecture

### Input Format
```
[CLS] Game Context [SEP] Server Response [SEP]
```

### Model Structure
```python
class MUDCommandClassifier(nn.Module):
    def __init__(self, num_commands: int):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, num_commands)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled)
```

## Training Data Structure

### Example Training Pair
```json
{
    "input": "Location: Dark Forest | Health: 75/100 | State: exploring | Last Command: look [SEP] You see a path leading north. A troll blocks the way south. [SEP]",
    "output": "look",
    "context": "exploring"
}
```

## Implementation Phases

### Phase 1: Data Collection
- Parse existing MUD logs
- Generate synthetic training data
- Create command-response pairs
- Validate data quality

### Phase 2: Model Training
- Fine-tune DistilBERT on MUD data
- Train custom classification head
- Optimize for command accuracy
- Validate on test set

### Phase 3: Integration
- Export model to ONNX/TorchScript
- Replace OpenAI backend
- Add confidence thresholds
- Implement fallback logic

## Performance Targets

- **Accuracy**: >90% command accuracy
- **Latency**: <50ms inference time
- **Memory**: <1GB RAM usage
- **Model Size**: <200MB quantized

## Integration with Current Client

```python
class LocalAIBackend:
    def __init__(self, model_path: str):
        self.model = torch.load(model_path)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    async def generate_response_async(self, prompt: str, context: GameContext) -> str:
        encoding = self.tokenizer(prompt, return_tensors='pt')
        with torch.no_grad():
            logits = self.model(**encoding)
            command_id = torch.argmax(logits).item()
        return self.command_mapping[command_id]
```

## Benefits Over External APIs

- **Cost**: One-time training vs. ongoing API costs
- **Latency**: Local inference eliminates network delays
- **Privacy**: All data stays local
- **Reliability**: No rate limits or service outages
- **Customization**: Model trained specifically for MUD patterns 