# Project 11: The Mixture of Experts (MoE)

## 🎯 Goal
Implement sparse Mixture-of-Experts architecture with top-k routing.

## 📚 Learning Objectives
- Build a router network that selects experts
- Implement sparse expert selection (top-2 routing)
- Understand load balancing and expert collapse
- Visualize expert utilization

## 🔬 The "Hard Way" Lesson
**Experts collapse if you don't penalize uneven routing.** Load balancing losses are essential for MoE training.

## 🛠️ Implementation Tasks

### Task 1: Implement Expert Layer

```python
class Expert(nn.Module):
    """Single expert: a feedforward network."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))

class Router(nn.Module):
    """
    Router network that assigns tokens to experts.

    Output: scores for each expert
    """
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.linear = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        # output: [batch, seq_len, num_experts]
        return F.softmax(self.linear(x), dim=-1)
```

### Task 2: Implement MoE Layer

```python
class MixtureOfExperts(nn.Module):
    """
    Sparse MoE layer with top-k routing.
    """
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Create experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(num_experts)
        ])

        # Create router
        self.router = Router(d_model, num_experts)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            output: [batch, seq_len, d_model]
            router_logits: [batch, seq_len, num_experts] (for load balancing loss)
        """
        batch_size, seq_len, d_model = x.shape

        # Get routing probabilities
        router_logits = self.router(x)  # [batch, seq_len, num_experts]

        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        # top_k_probs: [batch, seq_len, top_k]
        # top_k_indices: [batch, seq_len, top_k]

        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # YOUR CODE HERE
        # For each token:
        # 1. Route to top-k experts
        # 2. Compute expert outputs
        # 3. Combine with routing weights

        pass
```

### Task 3: Load Balancing Loss

```python
def load_balancing_loss(router_logits: torch.Tensor, num_experts: int):
    """
    Auxiliary loss to encourage balanced expert usage.

    Loss = num_experts × Σ_i (f_i × P_i)

    where:
        f_i = fraction of tokens routed to expert i
        P_i = average routing probability to expert i
    """
    # YOUR CODE HERE
    # Compute:
    # 1. Token fraction per expert (how many tokens chose expert i)
    # 2. Average routing probability per expert
    # 3. Product and sum
    pass
```

### Task 4: Visualize Expert Utilization

```python
def plot_expert_usage(router_logits: torch.Tensor, num_experts: int = 8):
    """
    Visualize expert selection patterns.

    Show:
        1. Histogram of expert selection frequency
        2. Heatmap of token-to-expert routing
        3. Load balance metric over time
    """
    # YOUR CODE HERE
    pass
```

### Task 5: Break It - Expert Collapse

```python
def demonstrate_expert_collapse():
    """
    Train MoE with and without load balancing loss.

    Show that without load balancing:
        - 1-2 experts get all the traffic
        - Other experts remain untrained
        - Loss stagnates
    """
    # Train two models:
    # 1. With load balancing loss (alpha=0.01)
    # 2. Without load balancing loss (alpha=0.0)

    # Plot:
    # - Expert usage distribution
    # - Training loss curves
    # - Expert weight norms

    # YOUR CODE HERE
    pass
```

## 📊 Expected Results

### Expert Usage (With Load Balancing)

| Expert | Usage % |
|--------|---------|
| Expert 0 | 12.5% |
| Expert 1 | 12.3% |
| Expert 2 | 12.8% |
| ... | ... |
| Expert 7 | 12.1% |

**Balanced:** Each expert gets ~12.5% of tokens (for 8 experts).

### Expert Collapse (Without Load Balancing)

| Expert | Usage % |
|--------|---------|
| Expert 0 | 0.2% |
| Expert 1 | 87.3% |
| Expert 2 | 11.5% |
| Expert 3-7 | <1% |

**Collapsed:** Most tokens go to 1-2 experts.

## 🎯 Deliverables

- [ ] Implemented `Expert` module
- [ ] Implemented `Router` module
- [ ] Implemented `MixtureOfExperts` layer
- [ ] Implemented load balancing loss
- [ ] Visualized expert utilization
- [ ] Demonstrated expert collapse without load balancing
- [ ] Plotted usage distribution with/without balancing

## 🚀 Next Steps
Move to **Project 12: Int8 Quantization**!
