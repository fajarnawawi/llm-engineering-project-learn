# Project 5: The Transformer Block

## 🎯 Goal
Assemble the complete Transformer block by stacking Multi-Head Attention + LayerNorm + FeedForward + Residual Connections.

## 📚 Learning Objectives
- Build a complete Transformer encoder/decoder block
- Understand residual connections and their importance
- Compare Pre-LN vs Post-LN architectures
- Probe internal activations at each layer

## 🔬 The "Hard Way" Lesson
**Residual streams are the "information highway."** Without skip connections, gradients vanish in deep networks.

## 🛠️ Implementation Tasks

### Task 1: Build Transformer Block Components

#### FeedForward Network
```python
class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        # Expand to d_ff, apply ReLU, project back
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
```

#### Transformer Block (Post-LN)
```python
class TransformerBlock(nn.Module):
    """
    Transformer block with Post-Layer Normalization.

    Original "Attention is All You Need" architecture.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Post-LN: x -> Attention -> Add&Norm -> FFN -> Add&Norm
        # YOUR CODE HERE
        pass
```

#### Pre-LN Variant
```python
class PreLNTransformerBlock(nn.Module):
    """
    Transformer block with Pre-Layer Normalization.

    More stable training, used in GPT, LLaMA, etc.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # YOUR CODE HERE
        pass

    def forward(self, x, mask=None):
        # Pre-LN: x -> Norm -> Attention -> Add -> Norm -> FFN -> Add
        # YOUR CODE HERE
        pass
```

### Task 2: Activation Probing

```python
def probe_activations(model, x, layer_names=['input', 'after_attn', 'after_norm1', 'after_ffn', 'output']):
    """
    Extract and visualize activations at different points in the block.

    Returns:
        activations: Dict mapping layer name to activation tensor
    """
    # YOUR CODE HERE
    pass

def plot_activation_distributions(activations):
    """
    Plot histograms of activation values at each layer.
    """
    # YOUR CODE HERE
    pass
```

### Task 3: Ablate - Compare Pre-LN vs Post-LN

**Experiment:** Train both architectures on a small language modeling task

```python
def compare_ln_variants(seq_len=128, epochs=10):
    """
    Train Pre-LN and Post-LN models, compare convergence.

    Measure:
        - Training loss curve
        - Gradient norms
        - Activation statistics
    """
    # YOUR CODE HERE
    pass
```

**Expected Results:**
- Pre-LN converges faster and more stably
- Post-LN may need learning rate warmup
- Pre-LN has cleaner gradient flow

## 📊 Key Visualizations

1. **Residual Stream Visualization**: Show how information flows through skip connections
2. **Activation Distribution**: Plot mean/std of activations at each layer
3. **Gradient Flow**: Visualize gradient magnitudes through the network
4. **Attention + FFN Contribution**: Decompose residual stream updates

## 🎯 Deliverables

- [ ] Implemented `FeedForward` module
- [ ] Implemented `TransformerBlock` (Post-LN)
- [ ] Implemented `PreLNTransformerBlock`
- [ ] Created activation probing tool
- [ ] Compared Pre-LN vs Post-LN convergence
- [ ] Visualized residual stream flow

## 🚀 Next Steps
Move to **Project 6: Norm & Activation** to explore normalization techniques!
