# LLM Engineering Solutions Guide

This document provides solution summaries and key implementation patterns for all 16 projects.

## 📚 Solution Notebooks Available

Complete, executable solution notebooks are provided for:
- **Project 1:** `01_byte_pair_smith/bpe_solution.ipynb`
- **Project 2:** `02_meaning_space/skipgram_solution.ipynb`
- **Project 3:** `03_rope_animator/positional_solution.ipynb`
- **Project 4:** `04_attention_lab/attention_solution.ipynb`

## 🔑 Key Implementation Patterns

### Projects 5-16: Solution Summaries

---

## Project 5: Transformer Block

### Core Implementation

```python
class TransformerBlock(nn.Module):
    """Post-LN Transformer block."""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Post-LN: Attention -> Add&Norm -> FFN -> Add&Norm
        attn_out, _ = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x

class PreLNTransformerBlock(nn.Module):
    """Pre-LN variant (more stable)."""
    def forward(self, x, mask=None):
        # Pre-LN: Norm -> Attention -> Add -> Norm -> FFN -> Add
        attn_out, _ = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)

        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)

        return x
```

### Key Insight
Pre-LN converges faster and more stably than Post-LN for deep networks.

---

## Project 6: Normalization & Activations

### RMSNorm Implementation

```python
class RMSNorm(nn.Module):
    """Simpler than LayerNorm, used in LLaMA."""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.gamma * x_norm
```

### SwiGLU Activation

```python
class SwiGLU(nn.Module):
    """SwiGLU(x) = Swish(xW) ⊙ (xV)"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W = nn.Linear(d_model, d_ff, bias=False)
        self.V = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        swish = self.W(x) * torch.sigmoid(self.W(x))  # Swish activation
        gate = self.V(x)
        return swish * gate  # Element-wise multiplication
```

---

## Project 7: Sampling Strategies

### Temperature Sampling

```python
def temperature_sampling(logits, temperature=1.0):
    """Scale logits and sample."""
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### Top-K Sampling

```python
def top_k_sampling(logits, k=50):
    """Keep only top-k tokens."""
    top_k_logits, top_k_indices = torch.topk(logits, k)
    probs = F.softmax(top_k_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, num_samples=1)
    return top_k_indices[sampled_idx]
```

### Top-P (Nucleus) Sampling

```python
def top_p_sampling(logits, p=0.9):
    """Sample from smallest set with cumulative prob >= p."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Find cutoff
    cutoff_idx = (cumulative_probs >= p).nonzero()[0]

    # Keep tokens up to cutoff
    nucleus_logits = sorted_logits[:cutoff_idx+1]
    nucleus_indices = sorted_indices[:cutoff_idx+1]

    probs = F.softmax(nucleus_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, num_samples=1)
    return nucleus_indices[sampled_idx]
```

---

## Project 8: KV Cache

### Implementation

```python
class CachedAttention(nn.Module):
    def forward(self, x, past_kv=None):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        if past_kv is not None:
            past_K, past_V = past_kv
            K = torch.cat([past_K, K], dim=1)  # Concatenate with history
            V = torch.cat([past_V, V], dim=1)

        # Compute attention with full K, V
        output = attention(Q, K, V)

        # Return new cache
        return output, (K, V)
```

### Memory Calculation

```
Cache Size = 2 × layers × batch × heads × seq_len × d_k × 4 bytes
Example (7B model, 2K context):
  = 2 × 32 × 1 × 32 × 2048 × 128 × 4
  ≈ 1 GB per sequence
```

---

## Project 9: Sliding Window Attention

### Window Mask Creation

```python
def create_sliding_window_mask(seq_len, window_size):
    """Band-diagonal mask."""
    mask = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, start:i+1] = 1
    return mask
```

### Expected Result
- Memory: O(n × window_size) instead of O(n²)
- Receptive field grows linearly with layers
- Fails on tasks requiring long-range dependencies

---

## Project 10: Grouped Query Attention (GQA)

### Implementation

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model * num_kv_heads // num_heads)
        self.W_v = nn.Linear(d_model, d_model * num_kv_heads // num_heads)

    def forward(self, x):
        Q = self.W_q(x)  # [batch, seq, d_model]
        K = self.W_k(x)  # [batch, seq, d_model * num_kv_heads / num_heads]
        V = self.W_v(x)

        # Reshape and repeat K, V to match Q heads
        # ... (repeat K, V across query groups)

        return attention(Q, K_repeated, V_repeated)
```

---

## Project 11: Mixture of Experts

### Load Balancing Loss

```python
def load_balancing_loss(router_logits, num_experts):
    """
    Encourages balanced expert usage.

    Loss = num_experts × Σ (f_i × P_i)
    where:
        f_i = fraction of tokens routed to expert i
        P_i = average routing probability to expert i
    """
    # Token fraction per expert
    routing_weights = F.softmax(router_logits, dim=-1)
    expert_mask = torch.argmax(routing_weights, dim=-1)
    expert_mask_onehot = F.one_hot(expert_mask, num_experts).float()

    tokens_per_expert = expert_mask_onehot.sum(dim=0)  # f_i
    total_tokens = expert_mask_onehot.shape[0]
    f = tokens_per_expert / total_tokens

    # Average routing probability
    P = routing_weights.mean(dim=0)

    # Load balancing loss
    loss = num_experts * torch.sum(f * P)
    return loss
```

---

## Project 12: Quantization

### AbsMax Quantization

```python
def absmax_quantize(tensor, n_bits=8):
    """Symmetric quantization."""
    # Find scale
    max_val = tensor.abs().max()
    scale = max_val / (2 ** (n_bits - 1) - 1)

    # Quantize
    quantized = torch.round(tensor / scale)
    quantized = torch.clamp(quantized, -(2**(n_bits-1)), 2**(n_bits-1) - 1)

    return quantized.to(torch.int8), scale

def dequantize(quantized, scale):
    """Restore to FP32."""
    return quantized.float() * scale
```

### Expected Degradation
- Int8: ~3% perplexity increase
- Int4: ~10-15% perplexity increase

---

## Project 13: Pretraining Objectives

### Causal LM (GPT)

```python
def causal_lm_loss(model, input_ids):
    logits = model(input_ids, causal_mask=True)
    # Shift: predict next token
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
    return loss
```

### Masked LM (BERT)

```python
def masked_lm_loss(model, input_ids, mask_prob=0.15):
    # Random masking
    mask = torch.rand(input_ids.shape) < mask_prob
    masked_ids = input_ids.clone()
    masked_ids[mask] = MASK_TOKEN_ID

    # Bidirectional encoding
    logits = model(masked_ids, causal_mask=False)

    # Loss only on masked positions
    loss = F.cross_entropy(logits[mask], input_ids[mask])
    return loss
```

---

## Project 14: Instruction Tuning

### Dataset Format

```python
def format_instruction(instruction, response):
    """Alpaca-style formatting."""
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""
```

### Loss Masking (Train only on response)

```python
def compute_loss_with_mask(logits, labels, instruction_length):
    """Mask out instruction tokens from loss."""
    loss_mask = torch.zeros_like(labels)
    loss_mask[:, instruction_length:] = 1  # Only compute loss on response

    loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), reduction='none')
    masked_loss = (loss * loss_mask.view(-1)).sum() / loss_mask.sum()
    return masked_loss
```

---

## Project 15: RLHF (PPO)

### Reward Model

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.reward_head = nn.Linear(d_model, 1)

    def forward(self, input_ids):
        hidden = self.base(input_ids)
        final_hidden = hidden[:, -1, :]  # Last token
        reward = self.reward_head(final_hidden).squeeze(-1)
        return reward
```

### KL Penalty

```python
def compute_kl_penalty(policy_logits, ref_logits):
    """KL(policy || reference)"""
    policy_logprobs = F.log_softmax(policy_logits, dim=-1)
    ref_logprobs = F.log_softmax(ref_logits, dim=-1)

    kl = (policy_logprobs.exp() * (policy_logprobs - ref_logprobs)).sum(dim=-1)
    return kl.mean()
```

### Combined Reward

```python
combined_reward = reward_model(response) - kl_coef * kl_penalty
```

---

## Project 16: Scaling Laws

### Power Law Fitting

```python
def fit_scaling_law(params, losses):
    """
    Fit: L(N) = (N_c / N)^α
    """
    from scipy.optimize import curve_fit

    def power_law(N, N_c, alpha):
        return (N_c / N) ** alpha

    popt, _ = curve_fit(power_law, params, losses)
    N_c, alpha = popt
    return N_c, alpha
```

### Chinchilla Optimal

```python
def chinchilla_optimal(compute_budget_flops):
    """
    For C FLOPs:
        N_opt = (C / 6)^0.5  # Optimal parameters
        D_opt = (C / 6)^0.5  # Optimal tokens
    """
    optimal_params = (compute_budget_flops / 6) ** 0.5
    optimal_tokens = (compute_budget_flops / 6) ** 0.5
    return optimal_params, optimal_tokens
```

---

## 🎯 Common Patterns

### Residual Connections
```python
# Standard pattern
output = x + module(x)

# With dropout
output = x + dropout(module(x))

# With normalization (Pre-LN)
output = x + module(norm(x))
```

### Attention Pattern
```python
# 1. Project to Q, K, V
# 2. Compute scores: QK^T / √d_k
# 3. Apply mask
# 4. Softmax
# 5. Multiply by V
```

### Training Loop Pattern
```python
for epoch in epochs:
    for batch in dataloader:
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
```

---

## 📊 Expected Results Summary

| Project | Key Metric | Expected Value |
|---------|------------|----------------|
| 1. BPE | Compression ratio | 2-4x |
| 2. Word2Vec | Training loss | <2.0 after 10 epochs |
| 3. Positional | Position distinguishability | 100% with PE, 0% without |
| 4. Attention | Softmax saturation (scaled) | <0.7 max weight |
| 7. Sampling | Entropy variation | 1.0 (temp=0.1) to 8.0 (temp=2.0) |
| 8. KV Cache | Speedup @ 500 tokens | 50x faster |
| 10. GQA | Memory savings (8→2 heads) | 75% reduction |
| 12. Quantization | Int8 perplexity increase | <5% |
| 16. Scaling Laws | Power law exponent α | ~0.076 |

---

## 💡 Debugging Tips

### Common Issues

**1. Gradient Vanishing**
- Check: Are you using Pre-LN?
- Fix: Switch from Post-LN to Pre-LN

**2. NaN in Attention**
- Check: Are all masked rows producing NaN softmax?
- Fix: `attention_weights = attention_weights.masked_fill(torch.isnan(attention_weights), 0.0)`

**3. Memory Explosion**
- Check: Is KV cache growing unbounded?
- Fix: Implement sliding window or cache eviction

**4. Poor Analogies in Word2Vec**
- Check: Is corpus too small?
- Fix: Use larger dataset or more epochs

**5. Tokenizer Inefficiency**
- Check: Cross-language tokenization?
- Fix: Train language-specific tokenizers

---

## 🚀 Next Steps After Completion

1. **Combine all components** into a full working LLM
2. **Optimize** with Flash Attention, fused kernels
3. **Scale up** to larger models (100M+ parameters)
4. **Deploy** with ONNX, TensorRT, or vLLM
5. **Research** novel architectures and training methods

---

**Remember:** The solutions are guides, not absolutes. Experiment, break things, and learn!

