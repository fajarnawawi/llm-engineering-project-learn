# Project 7: The Sampler Dashboard (Entropy Plotter)

## 🎯 Goal
Build an interactive dashboard to control text generation sampling and visualize the probability distributions.

## 📚 Learning Objectives
- Implement Temperature, Top-K, and Top-P (Nucleus) sampling
- Understand the relationship between temperature and entropy
- Build an interactive UI with Streamlit or Gradio
- Measure and plot distribution entropy

## 🔬 The "Hard Way" Lesson
**Temperature controls the "sharpness" of the softmax distribution.** Low temp → deterministic, high temp → random.

## 🛠️ Implementation Tasks

### Task 1: Implement Sampling Methods

```python
def temperature_sampling(logits: torch.Tensor, temperature: float = 1.0) -> int:
    """
    Sample from distribution with temperature scaling.

    Args:
        logits: [vocab_size] unnormalized logits
        temperature: Scaling factor (0.1 = sharp, 2.0 = flat)

    Returns:
        token_id: Sampled token index
    """
    # YOUR CODE HERE
    # 1. Scale logits by temperature
    # 2. Apply softmax
    # 3. Sample from categorical distribution
    pass

def top_k_sampling(logits: torch.Tensor, k: int = 50) -> int:
    """
    Sample from top-k most probable tokens.

    Args:
        logits: [vocab_size]
        k: Number of top tokens to consider

    Returns:
        token_id: Sampled token
    """
    # YOUR CODE HERE
    pass

def top_p_sampling(logits: torch.Tensor, p: float = 0.9) -> int:
    """
    Nucleus sampling: sample from smallest set with cumulative prob >= p.

    Args:
        logits: [vocab_size]
        p: Cumulative probability threshold (0.0 to 1.0)

    Returns:
        token_id: Sampled token
    """
    # YOUR CODE HERE
    # 1. Sort logits descending
    # 2. Compute cumulative probabilities
    # 3. Find cutoff where cumsum >= p
    # 4. Sample from that subset
    pass
```

### Task 2: Build Interactive Dashboard

```python
import streamlit as st
import plotly.graph_objects as go

def create_sampler_dashboard(model, tokenizer):
    """
    Streamlit app with:
        - Sliders for Temperature, Top-K, Top-P
        - Text input prompt
        - Generate button
        - Probability distribution plot
        - Entropy metric display
    """
    st.title("LLM Sampling Dashboard")

    # Sliders
    temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
    top_k = st.slider("Top-K", 1, 100, 50, 1)
    top_p = st.slider("Top-P", 0.0, 1.0, 0.9, 0.05)

    prompt = st.text_input("Prompt", "The quick brown fox")

    if st.button("Generate"):
        # YOUR CODE HERE
        # 1. Encode prompt
        # 2. Get next token logits
        # 3. Apply sampling
        # 4. Plot distribution
        # 5. Display entropy
        pass
```

### Task 3: Visualize Probability Distributions

```python
def plot_token_distribution(logits: torch.Tensor,
                           vocab: List[str],
                           temperature: float = 1.0,
                           top_k: int = 50):
    """
    Plot the next-token probability distribution.

    Show:
        - Top 20 tokens with probabilities
        - Effect of temperature on distribution shape
        - Entropy of distribution
    """
    # YOUR CODE HERE
    pass
```

### Task 4: Calculate Entropy

```python
def calculate_entropy(probs: torch.Tensor) -> float:
    """
    Shannon entropy: H(P) = -Σ p(x) log p(x)

    Higher entropy = more random
    Lower entropy = more deterministic
    """
    # YOUR CODE HERE
    # Add small epsilon to avoid log(0)
    pass

def plot_entropy_vs_temperature(logits: torch.Tensor, temps=[0.1, 0.5, 1.0, 1.5, 2.0]):
    """
    Plot how entropy changes with temperature.
    """
    # YOUR CODE HERE
    pass
```

## 📊 Expected Results

### Temperature Effects

| Temperature | Entropy | Behavior |
|------------|---------|----------|
| 0.1 | Low (~1.0) | Nearly deterministic, repetitive |
| 1.0 | Medium (~5.0) | Balanced, diverse |
| 2.0 | High (~8.0) | Very random, incoherent |

### Visualization Examples
- **Low temp (0.1):** Sharp peak on most likely token
- **Medium temp (1.0):** Balanced distribution
- **High temp (2.0):** Nearly uniform distribution

## 🎯 Deliverables

- [ ] Implemented temperature sampling
- [ ] Implemented top-k sampling
- [ ] Implemented top-p (nucleus) sampling
- [ ] Built Streamlit/Gradio dashboard
- [ ] Plotted probability distributions
- [ ] Calculated and displayed entropy
- [ ] Demonstrated temperature sweep

## 🚀 Next Steps
Move to **Project 8: KV Cache Speedrun** to optimize inference!
