# Project 6: Normalization & Activations

## 🎯 Goal
Understand the unsung heroes of neural network training: normalization layers and activation functions.

## 📚 Learning Objectives
- Implement RMSNorm and SwiGLU from scratch
- Compare LayerNorm, BatchNorm, and RMSNorm
- Visualize different activation functions
- Understand why BatchNorm fails for sequences

## 🔬 The "Hard Way" Lesson
**LayerNorm centers gradients; Pre-LN converges faster.** BatchNorm is terrible for variable-length sequences.

## 🛠️ Implementation Tasks

### Task 1: Implement Normalization Layers

#### RMSNorm (Root Mean Square Normalization)
```python
class RMSNorm(nn.Module):
    """
    RMSNorm: Simpler than LayerNorm, used in LLaMA.

    RMS(x) = x / sqrt(mean(x^2) + eps)
    Output = RMS(x) * gamma
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # YOUR CODE HERE
        pass
```

#### LayerNorm (for comparison)
```python
class LayerNorm(nn.Module):
    """
    Standard Layer Normalization.

    LN(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        # YOUR CODE HERE
        pass
```

### Task 2: Implement SwiGLU Activation

```python
class SwiGLU(nn.Module):
    """
    SwiGLU activation used in LLaMA, PaLM.

    SwiGLU(x) = Swish(xW) ⊙ (xV)
    where Swish(x) = x * sigmoid(x)
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W = nn.Linear(d_model, d_ff, bias=False)
        self.V = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        # YOUR CODE HERE
        pass
```

### Task 3: Visualize Activation Functions

```python
def plot_activation_functions():
    """
    Plot and compare: ReLU, GELU, Swish, SwiGLU
    """
    x = torch.linspace(-5, 5, 1000)

    activations = {
        'ReLU': F.relu(x),
        'GELU': F.gelu(x),
        'Swish': x * torch.sigmoid(x),
        'Tanh': torch.tanh(x),
    }

    # Plot all activations
    # Plot derivatives
    # YOUR CODE HERE
```

### Task 4: Experiment - BatchNorm vs LayerNorm

**Why BatchNorm Fails:**
- Computes statistics across batch dimension
- Variable-length sequences → padding → biased statistics
- Test-time statistics don't match training

```python
def compare_normalization(sequence_lengths=[10, 20, 50, 100]):
    """
    Demonstrate BatchNorm failure on variable-length sequences.
    """
    # YOUR CODE HERE
    pass
```

## 🎯 Deliverables

- [ ] Implemented `RMSNorm`
- [ ] Implemented `SwiGLU`
- [ ] Plotted activation function comparisons
- [ ] Demonstrated BatchNorm failure
- [ ] Compared LayerNorm vs RMSNorm speed

## 🚀 Next Steps
Move to **Phase 3: Project 7 - Entropy Plotter** to explore sampling strategies!
