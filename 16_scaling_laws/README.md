# Project 16: Scaling Laws

## 🎯 Goal
Discover the "physics of AI" by empirically measuring how loss scales with model size, data, and compute.

## 📚 Learning Objectives
- Train models of different sizes (Nano, Micro, Mini, Medium)
- Measure loss as a function of parameters
- Plot log-log scaling curves
- Extrapolate to predict larger models
- Understand compute-optimal training

## 🔬 The "Hard Way" Lesson
**Loss scales linearly with the log of compute/parameters.** Chinchilla laws tell us how to allocate training budget.

## 🛠️ Implementation Tasks

### Task 1: Train Models at Different Scales

```python
model_configs = {
    'nano': {
        'n_layers': 2,
        'n_heads': 2,
        'd_model': 64,
        'd_ff': 256,
    },
    'micro': {
        'n_layers': 4,
        'n_heads': 4,
        'd_model': 128,
        'd_ff': 512,
    },
    'mini': {
        'n_layers': 6,
        'n_heads': 6,
        'd_model': 256,
        'd_ff': 1024,
    },
    'small': {
        'n_layers': 12,
        'n_heads': 8,
        'd_model': 512,
        'd_ff': 2048,
    },
}

def train_at_scale(config, train_data, epochs=10):
    """
    Train model and record:
        - Parameter count
        - Training FLOPs
        - Final loss
        - Loss curve
    """
    # YOUR CODE HERE
    pass
```

### Task 2: Compute Training FLOPs

```python
def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_training_flops(model, tokens_seen):
    """
    Approximate FLOPs for training.

    FLOPs ≈ 6 × N × D

    where:
        N = number of parameters
        D = number of tokens seen during training
    """
    # YOUR CODE HERE
    pass
```

### Task 3: Plot Scaling Curves

```python
def plot_scaling_law(results):
    """
    Plot log(Loss) vs log(Parameters).

    Expected: Linear relationship in log-log space.

    Formula:
        L(N) = (N_c / N)^α
        log L = α log(N_c) - α log(N)

    where:
        N = parameters
        N_c = critical parameter count
        α ≈ 0.076 (Chinchilla paper)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Extract data
    params = [r['params'] for r in results]
    losses = [r['loss'] for r in results]

    # Log-log plot
    plt.figure(figsize=(10, 6))
    plt.loglog(params, losses, 'o-', label='Measured')

    # Fit power law
    from scipy.optimize import curve_fit

    def power_law(N, N_c, alpha):
        return (N_c / N) ** alpha

    popt, _ = curve_fit(power_law, params, losses)
    N_c, alpha = popt

    # Plot fit
    params_fit = np.logspace(np.log10(min(params)), np.log10(max(params)*10), 100)
    losses_fit = power_law(params_fit, N_c, alpha)
    plt.loglog(params_fit, losses_fit, '--', label=f'Fit: α={alpha:.3f}')

    plt.xlabel('Parameters (N)')
    plt.ylabel('Loss (L)')
    plt.title('Scaling Law: Loss vs Parameters')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return N_c, alpha
```

### Task 4: Extrapolate and Predict

```python
def predict_loss(target_params, N_c, alpha):
    """
    Predict loss for a model size not yet trained.

    Example:
        - Trained: Nano (1M), Micro (4M), Mini (16M)
        - Predict: Small (64M)
    """
    predicted_loss = (N_c / target_params) ** alpha
    return predicted_loss

def extrapolation_experiment():
    """
    1. Train Nano, Micro, Mini
    2. Fit scaling law
    3. Predict Small loss
    4. Actually train Small
    5. Compare prediction vs actual
    """
    # YOUR CODE HERE
    pass
```

### Task 5: Compute-Optimal Training (Chinchilla)

```python
def chinchilla_optimal(compute_budget_flops):
    """
    Compute optimal model size and training tokens for a given compute budget.

    Chinchilla result:
        N_opt ∝ C^α
        D_opt ∝ C^β

    where:
        α ≈ 0.50
        β ≈ 0.50
        (Equal scaling of parameters and data)

    Args:
        compute_budget_flops: Total training FLOPs available

    Returns:
        optimal_params: Recommended model size
        optimal_tokens: Recommended training tokens
    """
    # YOUR CODE HERE
    # For C FLOPs:
    # N_opt = (C / 6)^0.5
    # D_opt = (C / 6)^0.5
    pass
```

## 📊 Expected Results

### Scaling Law Data

| Model | Parameters | Loss | FLOPs |
|-------|------------|------|-------|
| Nano | 1M | 4.5 | 6B |
| Micro | 4M | 3.8 | 24B |
| Mini | 16M | 3.2 | 96B |
| Small | 64M | 2.7 | 384B |

**Log-log plot:** Nearly linear relationship.

### Power Law Fit

```
L(N) = (N_c / N)^α

Fitted parameters:
    N_c ≈ 10^9
    α ≈ 0.08
```

### Extrapolation Test

| Model | Predicted Loss | Actual Loss | Error |
|-------|----------------|-------------|-------|
| Small (64M) | 2.7 | 2.68 | 0.7% |

**Result:** Scaling laws are remarkably predictive!

### Chinchilla Optimal

For 1e18 FLOPs (1 exaFLOP):
- Optimal model size: ~400M parameters
- Optimal training tokens: ~8B tokens

## 🧪 Testing Your Understanding

**Check 1:** Why do scaling laws work in log-log space?
**Check 2:** What does the exponent α represent?
**Check 3:** Why does Chinchilla recommend equal scaling of N and D?
**Check 4:** Can scaling laws predict emergent abilities?

## 🎯 Deliverables

- [ ] Trained 4+ models at different scales
- [ ] Counted parameters for each model
- [ ] Measured final loss for each model
- [ ] Plotted log(Loss) vs log(Parameters)
- [ ] Fitted power law scaling curve
- [ ] Extrapolated to predict unseen model size
- [ ] Validated prediction by training target model
- [ ] Computed Chinchilla-optimal allocation

## 📖 Resources

- **Scaling Laws for Neural LMs:** [Kaplan et al. 2020](https://arxiv.org/abs/2001.08361)
- **Training Compute-Optimal LLMs:** [Hoffmann et al. 2022 (Chinchilla)](https://arxiv.org/abs/2203.15556)
- **Emergent Abilities:** [Wei et al. 2022](https://arxiv.org/abs/2206.07682)

## 🎉 Congratulations!

You've completed all 16 projects of **LLM Engineering: The Hard Way**!

You now understand:
- ✅ How tokenization works (BPE)
- ✅ Embedding spaces and semantic geometry
- ✅ Positional encodings (Sinusoidal, RoPE)
- ✅ Attention mechanisms and masking
- ✅ Transformer architecture
- ✅ Normalization and activations
- ✅ Sampling strategies
- ✅ KV caching and inference optimization
- ✅ Long-context handling
- ✅ Memory-efficient architectures (GQA)
- ✅ Mixture of Experts
- ✅ Quantization
- ✅ Training objectives
- ✅ Instruction tuning
- ✅ RLHF
- ✅ Scaling laws

**You built everything from scratch. You earned this knowledge.**

## 🚀 What's Next?

1. **Implement a full model:** Combine all components into a working LLM
2. **Contribute to open source:** Apply your knowledge to real projects
3. **Research:** Push the boundaries of what's possible
4. **Build products:** Create the next generation of AI applications

**The hard way is the only way to truly understand.**
