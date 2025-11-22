# Project 3: Positioning in 3D (RoPE Animator)

## 🎯 Goal
Solve the "Set vs. Sequence" problem by implementing and comparing three positional embedding methods.

## 📚 Learning Objectives
- Understand why Transformers need position information
- Implement Sinusoidal, Learned, and Rotary (RoPE) positional embeddings
- Visualize how positions are encoded geometrically
- Demonstrate the failure mode when positions are removed

## 🔬 The "Hard Way" Lesson
**Absolute positions fail on long documents. Rotation generalizes better.** Without positional information, "Dog bites man" is identical to "Man bites dog."

## 🛠️ Implementation Tasks

### Task 1: Implement Three Positional Embedding Classes

#### 1a. Sinusoidal Positional Embedding (Original Transformer)
The classic "Attention is All You Need" approach:

```python
class SinusoidalPositionalEmbedding(nn.Module):
    """
    Fixed sinusoidal position encoding.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # YOUR CODE HERE
        # Create position encoding matrix of shape [max_len, d_model]
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
        Returns:
            x + positional encodings
        """
        # YOUR CODE HERE
        pass
```

#### 1b. Learned Positional Embedding (BERT-style)
Simple lookup table:

```python
class LearnedPositionalEmbedding(nn.Module):
    """
    Learned position embeddings (trainable).
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # YOUR CODE HERE
        # Use nn.Embedding to create learnable position embeddings
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
        Returns:
            x + learned positional encodings
        """
        # YOUR CODE HERE
        pass
```

#### 1c. Rotary Positional Embeddings (RoPE)
Used in LLaMA, GPT-NeoX:

```python
class RoPE(nn.Module):
    """
    Rotary Position Embedding.

    Rotates query and key vectors by angle proportional to position.
    """
    def __init__(self, d_model: int, base: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.base = base

        # Precompute theta values
        # YOUR CODE HERE
        pass

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary embeddings.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            positions: Position indices [seq_len]
        Returns:
            Rotated tensor
        """
        # YOUR CODE HERE
        # Hint: Split into even/odd dimensions and apply rotation
        pass
```

### Task 2: Visualize Position Encodings

#### 2a. Heatmap of Sinusoidal Encodings
```python
def plot_sinusoidal_heatmap(pe_module, max_pos=100):
    """
    Plot the sinusoidal position encoding pattern.
    """
    # YOUR CODE HERE
    # Generate encodings for positions 0 to max_pos
    # Display as heatmap (position x dimension)
    pass
```

#### 2b. Animate RoPE Rotation (2D)
```python
def animate_rope_rotation(rope_module, num_positions=10):
    """
    Animate how a vector rotates as position increases.

    Creates a 2D animation showing vector rotation in a plane.
    """
    # YOUR CODE HERE
    # Use matplotlib.animation or plotly
    # Show vector at position 0, 1, 2, ... rotating
    pass
```

#### 2c. Compare All Three Methods
```python
def compare_positional_methods(seq_len=50, d_model=64):
    """
    Side-by-side comparison of all three positional encoding methods.
    """
    # YOUR CODE HERE
    # Create dummy input
    # Apply each method
    # Plot resulting position encodings
    pass
```

### Task 3: Break It - Remove Positional Information

**Experiment:** Train a small sequence model (2-layer Transformer) on a simple task:
- **Task:** Predict if sentence is "Dog bites man" (label=0) or "Man bites dog" (label=1)
- **Dataset:** Generate 1000 examples of each

#### 3a. Create Dataset
```python
def create_word_order_dataset(num_examples=1000):
    """
    Generate examples where word order matters.

    Returns:
        sentences: List of tokenized sentences
        labels: 0 for "Dog bites man", 1 for "Man bites dog"
    """
    # YOUR CODE HERE
    pass
```

#### 3b. Train Three Models
```python
# Model 1: No positional encoding
model_none = SimpleTransformer(use_pos_encoding=None)

# Model 2: With sinusoidal encoding
model_sin = SimpleTransformer(use_pos_encoding='sinusoidal')

# Model 3: With RoPE
model_rope = SimpleTransformer(use_pos_encoding='rope')

# Train all three and compare accuracy
```

#### Expected Result:
- **Without positions:** ~50% accuracy (random guessing - can't distinguish order)
- **With positions:** >95% accuracy (correctly identifies word order)

## 📊 Expected Visualizations

### What You Should See:

1. **Sinusoidal Heatmap:**
   - Smooth wave patterns
   - Different frequencies for different dimensions
   - Repeating patterns at different scales

2. **RoPE Animation:**
   - Vector rotating smoothly as position increases
   - Rotation angle proportional to position index
   - Consistent rotation direction

3. **Comparison Plot:**
   - Sinusoidal: Fixed patterns
   - Learned: Potentially irregular (depends on training)
   - RoPE: Applied in attention space (different visualization)

## 🧪 Testing Your Understanding

**Check 1:** What happens to sinusoidal encodings at positions > max_len?

**Check 2:** Why does RoPE rotate pairs of dimensions instead of adding vectors?

**Check 3:** Can learned positional embeddings generalize to sequences longer than seen during training?

**Check 4:** Why is RoPE relative while sinusoidal is absolute?

## 📖 Resources
- Attention is All You Need: [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)
- RoFormer (RoPE): [Su et al. 2021](https://arxiv.org/abs/2104.09864)
- Illustrated Positional Encoding: [Blog post](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)

## ⚡ Bonus Challenges

1. **ALiBi:** Implement Attention with Linear Biases (used in BLOOM)
2. **Extrapolation Test:** Train on sequences of length 128, test on length 256
3. **Relative Positions:** Implement T5-style relative position bias
4. **Frequency Analysis:** Show how sinusoidal encoding creates different "wavelengths"
5. **3D Rotation:** Visualize RoPE in 3D space (rotate entire vector, not just 2D projection)

## 🎯 Deliverables

- [ ] Implemented `SinusoidalPositionalEmbedding`
- [ ] Implemented `LearnedPositionalEmbedding`
- [ ] Implemented `RoPE`
- [ ] Created sinusoidal encoding heatmap
- [ ] Created RoPE rotation animation
- [ ] Trained models with/without positional encodings
- [ ] Demonstrated "Dog bites man" vs "Man bites dog" failure
- [ ] Comparison analysis of all three methods

## 📝 Key Insights to Gain

1. **Why positions matter:** Sets are unordered, sequences are ordered
2. **Sinusoidal pros:** No training needed, mathematically elegant
3. **Sinusoidal cons:** Absolute positions don't extrapolate well
4. **Learned pros:** Can adapt to data
5. **Learned cons:** Can't generalize beyond training length
6. **RoPE pros:** Relative positions, excellent extrapolation
7. **RoPE cons:** More complex to implement, requires modification to attention

## 🚀 Next Steps
Once complete, move to **Project 4: The Attention Surgeon** to build the core attention mechanism!
