# MUD AI Model Technical Specification
## Detailed Implementation Guide

### Model Architecture Details

## 1. Input Processing

### Tokenization Strategy
```python
# Input format: [CLS] Game Context [SEP] Server Response [SEP]
def tokenize_input(game_context: dict, server_response: str) -> torch.Tensor:
    """
    Tokenize game context and server response for DistilBERT
    """
    context_text = f"Location: {game_context['location']} | "
    context_text += f"Health: {game_context['health']}/{game_context['max_health']} | "
    context_text += f"State: {game_context['game_state']} | "
    context_text += f"Last Command: {game_context['last_command']}"
    
    full_text = f"[CLS] {context_text} [SEP] {server_response} [SEP]"
    
    return tokenizer(
        full_text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
```

### Context Encoding
```python
class GameContextEncoder:
    """Encode game context into model input"""
    
    def __init__(self, max_length: int = 512):
        self.max_length = max_length
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    def encode(self, context: GameContext, response: str) -> Dict[str, torch.Tensor]:
        # Combine context and response
        context_str = self._format_context(context)
        full_text = f"[CLS] {context_str} [SEP] {response} [SEP]"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return encoding
```

## 2. Model Architecture

### Custom DistilBERT Head
```python
class MUDCommandClassifier(nn.Module):
    """Custom classifier for MUD command prediction"""
    
    def __init__(self, num_commands: int, dropout: float = 0.1):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Freeze DistilBERT layers (optional)
        for param in self.distilbert.parameters():
            param.requires_grad = False
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_commands)
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask):
        # Get DistilBERT embeddings
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Classification
        logits = self.classifier(pooled_output)
        confidence = self.confidence_head(pooled_output)
        
        return logits, confidence
```

### Alternative: Sequence-to-Sequence Approach
```python
class MUDCommandGenerator(nn.Module):
    """Sequence-to-sequence model for command generation"""
    
    def __init__(self, vocab_size: int, max_length: int = 20):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.decoder = nn.TransformerDecoder(
            num_layers=2,
            d_model=768,
            nhead=8,
            dim_feedforward=2048
        )
        self.output_projection = nn.Linear(768, vocab_size)
        self.max_length = max_length
    
    def forward(self, input_ids, attention_mask, target_ids=None):
        # Encode input
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        if self.training and target_ids is not None:
            # Training: teacher forcing
            decoder_outputs = self.decoder(
                target_ids,
                encoder_outputs.last_hidden_state
            )
        else:
            # Inference: auto-regressive generation
            decoder_outputs = self._generate(encoder_outputs.last_hidden_state)
        
        return self.output_projection(decoder_outputs)
```

## 3. Training Pipeline

### DataLoader Implementation
```python
class MUDDataset(Dataset):
    """Custom dataset for MUD command training"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize input
        encoding = self.tokenizer(
            item['input_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create target
        target = torch.tensor(item['command_id'], dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'target': target,
            'confidence': torch.tensor(item['success'], dtype=torch.float)
        }
```

### Training Loop
```python
class MUDTrainer:
    """Custom trainer for MUD model"""
    
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Forward pass
            logits, confidence = self.model(input_ids, attention_mask)
            
            # Calculate loss
            loss = self.criterion(logits, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
```

## 4. Data Generation

### Synthetic Data Generator
```python
class MUDDataGenerator:
    """Generate synthetic training data for MUD interactions"""
    
    def __init__(self):
        self.commands = {
            'movement': ['north', 'south', 'east', 'west', 'up', 'down'],
            'action': ['look', 'examine', 'take', 'drop', 'inventory'],
            'combat': ['attack', 'flee', 'cast', 'heal'],
            'social': ['say', 'tell', 'emote', 'who']
        }
        
        self.locations = [
            'Dark Forest', 'Castle Hall', 'Dungeon Cell', 'Mountain Pass',
            'Village Square', 'Ancient Temple', 'Crystal Cave', 'Desert Oasis'
        ]
        
        self.enemies = ['troll', 'goblin', 'dragon', 'skeleton', 'orc']
    
    def generate_example(self) -> Dict:
        """Generate a single training example"""
        context = self._generate_context()
        response = self._generate_response(context)
        command = self._generate_command(context, response)
        
        return {
            'game_context': context,
            'server_response': response,
            'chosen_command': command,
            'success': True
        }
    
    def _generate_context(self) -> Dict:
        return {
            'location': random.choice(self.locations),
            'health': random.randint(20, 100),
            'max_health': 100,
            'game_state': random.choice(['exploring', 'combat', 'dialogue']),
            'last_command': random.choice(self.commands['movement'])
        }
```

### Real Data Parser
```python
class MUDLogParser:
    """Parse real MUD logs to extract training data"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.command_pattern = re.compile(r'^> (.+)$')
        self.response_pattern = re.compile(r'^([^>].+)$')
    
    def parse_log(self) -> List[Dict]:
        """Parse log file and extract command-response pairs"""
        pairs = []
        current_context = {}
        
        with open(self.log_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Check for command
                command_match = self.command_pattern.match(line)
                if command_match:
                    command = command_match.group(1)
                    current_context['last_command'] = command
                    continue
                
                # Check for response
                response_match = self.response_pattern.match(line)
                if response_match and current_context:
                    response = response_match.group(1)
                    pairs.append({
                        'context': current_context.copy(),
                        'response': response,
                        'command': current_context.get('last_command', '')
                    })
        
        return pairs
```

## 5. Model Optimization

### Quantization
```python
def quantize_model(model: nn.Module) -> nn.Module:
    """Quantize model for faster inference"""
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )
    return quantized_model
```

### ONNX Export
```python
def export_to_onnx(model: nn.Module, save_path: str):
    """Export model to ONNX format for deployment"""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randint(0, 1000, (1, 512))
    dummy_mask = torch.ones(1, 512)
    
    # Export
    torch.onnx.export(
        model,
        (dummy_input, dummy_mask),
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits', 'confidence'],
        dynamic_axes={
            'input_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'confidence': {0: 'batch_size'}
        }
    )
```

## 6. Integration with Existing Client

### Model Backend Implementation
```python
class LocalAIBackend:
    """Local AI backend using custom trained model"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = device
        self.model = self._load_model(model_path)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.command_mapping = self._load_command_mapping()
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained model"""
        model = MUDCommandClassifier(num_commands=len(self.command_mapping))
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    async def generate_response_async(self, prompt: str, context: GameContext) -> str:
        """Generate command using local model"""
        # Tokenize input
        encoding = self.tokenizer(
            prompt,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            logits, confidence = self.model(input_ids, attention_mask)
            predicted_id = torch.argmax(logits, dim=1).item()
            confidence_score = confidence.item()
        
        # Map to command
        command = self.command_mapping[predicted_id]
        
        # Apply confidence threshold
        if confidence_score < 0.5:
            command = "look"  # Fallback
        
        return command
```

## 7. Performance Benchmarks

### Target Performance Metrics
```python
# Performance targets
PERFORMANCE_TARGETS = {
    'inference_time': 50,  # ms
    'memory_usage': 1024,  # MB
    'accuracy': 0.90,      # 90%
    'throughput': 100      # commands/second
}

# Benchmarking function
def benchmark_model(model, test_data):
    """Benchmark model performance"""
    start_time = time.time()
    memory_before = psutil.Process().memory_info().rss / 1024 / 1024
    
    # Run inference
    for sample in test_data:
        _ = model(sample)
    
    end_time = time.time()
    memory_after = psutil.Process().memory_info().rss / 1024 / 1024
    
    return {
        'inference_time': (end_time - start_time) / len(test_data) * 1000,
        'memory_usage': memory_after - memory_before,
        'throughput': len(test_data) / (end_time - start_time)
    }
```

## 8. Deployment Strategy

### Model Serving
```python
class ModelServer:
    """Simple model server for deployment"""
    
    def __init__(self, model_path: str, port: int = 8000):
        self.model = LocalAIBackend(model_path)
        self.app = FastAPI()
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.post("/predict")
        async def predict(request: PredictionRequest):
            command = await self.model.generate_response_async(
                request.prompt, 
                request.context
            )
            return {"command": command}
    
    def run(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8000)
```

This technical specification provides the detailed implementation guide for the custom MUD AI model, covering all aspects from data processing to deployment. 