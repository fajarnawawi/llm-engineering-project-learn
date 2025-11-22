# Project 8: KV Cache Speedrun

## 🎯 Goal
Stop re-computing history by implementing Key-Value caching for autoregressive generation.

## 📚 Learning Objectives
- Understand why naive generation is O(n²)
- Implement KV cache in attention mechanism
- Benchmark inference speed with and without cache
- Monitor memory usage during generation

## 🔬 The "Hard Way" Lesson
**The bottleneck is memory bandwidth, not compute.** KV cache turns O(n²) into O(n) but increases VRAM usage.

## 🛠️ Implementation Tasks

### Task 1: Implement KV-Cached Attention

```python
class CachedMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with KV caching.
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # Same as before
        pass

    def forward(self,
                x: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = True):
        """
        Args:
            x: [batch, seq_len, d_model] (can be seq_len=1 for generation)
            past_key_value: (past_keys, past_values) from previous steps
            use_cache: Whether to return cached K, V

        Returns:
            output: [batch, seq_len, d_model]
            new_key_value: (keys, values) for next step
        """
        # YOUR CODE HERE
        # 1. Compute Q, K, V for current step
        # 2. If past_key_value exists, concatenate with new K, V
        # 3. Compute attention using full K, V
        # 4. Return output and new cache
        pass
```

### Task 2: Benchmark Inference Speed

```python
def benchmark_generation(model, prompt, max_tokens=500, use_cache=True):
    """
    Generate tokens and measure time.

    Returns:
        - generated_text
        - total_time
        - tokens_per_second
        - memory_used
    """
    import time
    import torch.cuda

    start_time = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # YOUR CODE HERE
    # Generate tokens one by one

    elapsed = time.time() - start_time
    tokens_per_sec = max_tokens / elapsed

    if torch.cuda.is_available():
        memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        memory_mb = 0

    return {
        'time': elapsed,
        'tokens_per_sec': tokens_per_sec,
        'memory_mb': memory_mb
    }
```

### Task 3: Compare With and Without Cache

```python
def compare_cache_performance(model, prompt, sequence_lengths=[50, 100, 200, 500]):
    """
    Benchmark generation at different lengths with and without cache.

    Plot:
        - Generation time vs sequence length
        - Memory usage vs sequence length
        - Speedup factor
    """
    results_cached = []
    results_no_cache = []

    for seq_len in sequence_lengths:
        # With cache
        result_cached = benchmark_generation(model, prompt, seq_len, use_cache=True)
        results_cached.append(result_cached)

        # Without cache
        result_no_cache = benchmark_generation(model, prompt, seq_len, use_cache=False)
        results_no_cache.append(result_no_cache)

    # Plot results
    # YOUR CODE HERE
```

### Task 4: Memory Profiler

```python
def profile_kv_cache_memory(model, seq_len, batch_size=1):
    """
    Calculate theoretical KV cache memory usage.

    Formula:
        Memory = 2 * num_layers * batch_size * num_heads * seq_len * d_k * sizeof(float)

    Returns:
        - cache_size_mb
        - breakdown by layer
    """
    # YOUR CODE HERE
    pass

def plot_memory_growth(max_seq_len=2048):
    """
    Plot how VRAM usage grows linearly with sequence length.
    """
    # YOUR CODE HERE
    pass
```

## 📊 Expected Results

### Speed Comparison

| Sequence Length | Without Cache | With Cache | Speedup |
|----------------|---------------|------------|---------|
| 50 tokens | 2.5s | 0.5s | 5x |
| 100 tokens | 10s | 1.0s | 10x |
| 200 tokens | 40s | 2.0s | 20x |
| 500 tokens | 250s | 5.0s | 50x |

**Complexity:**
- Without cache: O(n²) - recompute all previous positions
- With cache: O(n) - only compute new position

### Memory Usage

```
Cache Size = 2 × layers × heads × seq_len × d_k × 4 bytes

Example (LLaMA-7B):
- Layers: 32
- Heads: 32
- d_k: 128
- Seq_len: 2048
→ Cache size ≈ 1 GB per batch item
```

## 🧪 Testing Your Understanding

**Check 1:** Why does generation become quadratic without caching?
**Check 2:** Why do we cache K and V but not Q?
**Check 3:** What happens to cache when we do batch generation?
**Check 4:** Can we apply KV cache to bidirectional models like BERT?

## 🎯 Deliverables

- [ ] Implemented `CachedMultiHeadAttention`
- [ ] Benchmarked generation with and without cache
- [ ] Plotted time vs sequence length comparison
- [ ] Profiled memory usage
- [ ] Demonstrated speedup factor

## ⚡ Bonus Challenges

1. **Multi-query Attention**: Share K/V across queries for memory efficiency
2. **PagedAttention**: Implement vLLM-style paged KV cache
3. **Cache Eviction**: Implement sliding window cache (drop oldest keys)

## 🚀 Next Steps
Move to **Project 9: Sliding Window** to handle infinite context!
