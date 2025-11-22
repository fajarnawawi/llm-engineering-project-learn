# Project 4: The Attention Surgeon

## 🎯 Goal
Hand-wire dot-product attention from scratch and understand every mathematical operation.

## 📚 Learning Objectives
- Implement scaled dot-product attention without libraries
- Visualize attention weight matrices
- Understand the role of scaling factor (√d_k)
- Break attention by removing components and observe failures

## 🔬 The "Hard Way" Lesson
**Scaling factors prevent softmax saturation.** Without 1/√d_k, gradients vanish for large d_k values.

## 🛠️ Implementation Tasks

### Task 1: Implement Core Attention Function

```python
def attention(Q: torch.Tensor,
              K: torch.Tensor,
              V: torch.Tensor,
              mask: Optional[torch.Tensor] = None,
              scale: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled Dot-Product Attention.

    Attention(Q, K, V) = softmax(QK^T / √d_k) V

    Args:
        Q: Query tensor [batch, seq_len, d_k]
        K: Key tensor [batch, seq_len, d_k]
        V: Value tensor [batch, seq_len, d_v]
        mask: Optional mask [batch, seq_len, seq_len]
        scale: Whether to apply 1/√d_k scaling

    Returns:
        output: Attention output [batch, seq_len, d_v]
        attention_weights: Attention matrix [batch, seq_len, seq_len]
    """
    # YOUR CODE HERE
    pass
```

**Step-by-step implementation:**
1. Compute QK^T (batch matrix multiplication)
2. Scale by 1/√d_k (if enabled)
3. Apply mask (set masked positions to -inf)
4. Apply softmax along last dimension
5. Multiply by V
6. Return output and attention weights

### Task 2: Visualize Attention Patterns

Create multiple visualization functions:

#### 2a. Single Attention Head Heatmap
```python
def plot_attention_heatmap(attention_weights: torch.Tensor,
                          tokens: List[str],
                          title: str = "Attention Weights"):
    """
    Plot attention weight matrix as heatmap.

    Args:
        attention_weights: [seq_len, seq_len]
        tokens: List of token strings
    """
    # YOUR CODE HERE
    pass
```

#### 2b. Multi-Head Attention Grid
```python
def plot_multihead_attention(attention_weights: torch.Tensor,
                             tokens: List[str],
                             num_heads: int):
    """
    Plot all attention heads in a grid.

    Args:
        attention_weights: [num_heads, seq_len, seq_len]
        tokens: List of tokens
        num_heads: Number of attention heads
    """
    # YOUR CODE HERE
    pass
```

#### 2c. Attention Pattern Analysis
```python
def analyze_attention_patterns(attention_weights: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics about attention patterns.

    Returns:
        - max_attention: Maximum attention score
        - entropy: Shannon entropy of attention distribution
        - diagonal_mass: % of attention on diagonal (self-attention)
        - local_vs_global: Ratio of local to global attention
    """
    # YOUR CODE HERE
    pass
```

### Task 3: Break Attention - Ablation Studies

#### 3a. Remove Scaling Factor
**Experiment:** Compare attention with and without 1/√d_k

```python
def experiment_scaling_factor(d_k_values=[16, 64, 256, 1024]):
    """
    Test impact of scaling across different d_k values.

    Measure:
        - Softmax saturation (max value approaching 1.0)
        - Gradient magnitude
        - Entropy of attention distribution
    """
    # YOUR CODE HERE
    pass
```

**Expected Observation:**
- Without scaling, larger d_k → sharper softmax → gradient vanishing
- With scaling, attention distribution remains stable

#### 3b. Wrong Masking Direction
**Experiment:** Mask out the *past* instead of the *future*

```python
def create_causal_mask(seq_len: int, device='cpu') -> torch.Tensor:
    """
    Create lower-triangular mask for causal attention.

    Returns:
        mask: [seq_len, seq_len] with 0s in lower triangle, -inf in upper
    """
    # YOUR CODE HERE
    pass

def create_reverse_mask(seq_len: int, device='cpu') -> torch.Tensor:
    """
    WRONG: Create upper-triangular mask (lets model cheat!).
    """
    # YOUR CODE HERE
    pass
```

Train two language models:
- Model A: Correct causal mask
- Model B: Reverse mask (can see future)

**Expected Result:**
- Model B achieves near-perfect training loss (cheating!)
- Model B fails at inference (no future tokens available)

#### 3c. No Masking at All
```python
def compare_masking():
    """
    Compare three scenarios:
        1. Causal mask (can only see past)
        2. No mask (can see everything - bidirectional)
        3. Reverse mask (can only see future - broken)
    """
    # YOUR CODE HERE
    pass
```

## 📊 Expected Results

### Attention Weight Visualizations

**What to look for:**
1. **Diagonal patterns:** Self-attention (attending to same position)
2. **Vertical lines:** Attending strongly to specific tokens (like punctuation)
3. **Horizontal lines:** Broadcasting information from important tokens
4. **Block patterns:** Phrase-level attention

### Scaling Factor Experiment

| d_k | Without Scaling | With Scaling |
|-----|----------------|--------------|
| 16  | Max ≈ 0.8      | Max ≈ 0.5    |
| 64  | Max ≈ 0.95     | Max ≈ 0.5    |
| 256 | Max ≈ 0.99     | Max ≈ 0.5    |
| 1024| Max ≈ 1.0      | Max ≈ 0.5    |

Without scaling, attention becomes a one-hot selector (peaky distribution).

## 🧪 Testing Your Understanding

**Check 1:** What happens to QK^T values as d_k increases?
- Hint: Inner product of two random vectors scales with √d_k

**Check 2:** Why use -inf for masking instead of 0?
- Hint: What does softmax(0) equal?

**Check 3:** Why is the output dimension d_v, not d_k?
- Hint: Which matrix determines output dimension?

**Check 4:** Can attention output have higher rank than inputs?
- Hint: Attention is a weighted sum (convex combination)

## 📖 Resources
- Attention is All You Need: [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)
- The Illustrated Transformer: [Jay Alammar](http://jalammar.github.io/illustrated-transformer/)
- Visualizing Attention: [Jesse Vig's BertViz](https://github.com/jessevig/bertviz)

## 💡 Key Implementation Details

### Numerical Stability
```python
# Bad: Can cause overflow
scores = torch.exp(logits)

# Good: Subtract max before exp
logits_max = logits.max(dim=-1, keepdim=True)[0]
scores = torch.exp(logits - logits_max)
```

### Masking
```python
# Apply mask BEFORE softmax
scores = scores.masked_fill(mask == 0, float('-inf'))
attention_weights = F.softmax(scores, dim=-1)
```

### Handling NaN
```python
# If entire row is masked, softmax produces NaN
# Replace NaN with 0
attention_weights = attention_weights.masked_fill(torch.isnan(attention_weights), 0.0)
```

## ⚡ Bonus Challenges

1. **Flash Attention:** Implement tiling to reduce memory usage
2. **Sparse Attention:** Only attend to k-nearest neighbors
3. **Cross-Attention:** Implement attention between two different sequences
4. **Multi-Query Attention:** Share K and V across heads
5. **Attention Dropout:** Add dropout to attention weights

## 🎯 Deliverables

- [ ] Implemented `attention()` function
- [ ] Created attention heatmap visualization
- [ ] Tested scaling factor across different d_k
- [ ] Demonstrated gradient vanishing without scaling
- [ ] Implemented causal masking
- [ ] Showed failure with reverse masking
- [ ] Computed attention pattern statistics
- [ ] Analyzed attention entropy

## 🚀 Next Steps
Once complete, move to **Project 5: The Transformer Block** to assemble the full layer!
