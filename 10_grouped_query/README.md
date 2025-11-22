# Project 10: Grouped Query Attention (GQA)

## 🎯 Goal
Implement GQA to reduce KV cache memory usage by sharing keys and values across query heads.

## 📚 Learning Objectives
- Understand Multi-Query Attention (MQA) and Grouped Query Attention (GQA)
- Implement GQA with configurable num_kv_heads
- Measure memory savings vs standard MHA
- Evaluate perplexity trade-offs

## 🔬 The "Hard Way" Lesson
**Sharing KV heads saves massive VRAM with minimal perplexity loss.** Used in LLaMA-2, Mistral, etc.

## 🛠️ Implementation Tasks

### Task 1: Implement GQA

```python
class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention.

    Instead of num_heads separate K/V heads, use num_kv_heads < num_heads.
    Query heads are grouped to share K/V heads.

    Example:
        num_heads = 32
        num_kv_heads = 8
        → Each K/V head is shared by 4 Q heads
    """
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        assert num_heads % num_kv_heads == 0

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.d_k = d_model // num_heads

        # Q has full num_heads, but K/V only have num_kv_heads
        # YOUR CODE HERE
        pass

    def forward(self, x, mask=None, use_cache=False, past_kv=None):
        # YOUR CODE HERE
        # 1. Project to Q, K, V
        # 2. Reshape Q to [batch, num_heads, seq_len, d_k]
        # 3. Reshape K, V to [batch, num_kv_heads, seq_len, d_k]
        # 4. Repeat K, V to match num_heads
        # 5. Apply attention
        pass
```

### Task 2: Compare MHA vs MQA vs GQA

```python
def compare_attention_variants(d_model=512, num_heads=8):
    """
    Compare three architectures:
        1. Multi-Head Attention (MHA): num_kv_heads = num_heads
        2. Multi-Query Attention (MQA): num_kv_heads = 1
        3. Grouped Query Attention (GQA): num_kv_heads = 2 or 4

    Measure:
        - Parameter count
        - KV cache size
        - Inference speed
        - Perplexity (if trained)
    """
    configs = {
        'MHA': {'num_kv_heads': 8},
        'GQA-4': {'num_kv_heads': 4},
        'GQA-2': {'num_kv_heads': 2},
        'MQA': {'num_kv_heads': 1},
    }

    # YOUR CODE HERE
    pass
```

### Task 3: Measure Memory Savings

```python
def calculate_kv_cache_size(num_layers, num_kv_heads, seq_len, d_k, batch_size=1):
    """
    KV cache size = 2 × num_layers × batch_size × num_kv_heads × seq_len × d_k × 4 bytes

    Compare GQA vs MHA memory usage.
    """
    # YOUR CODE HERE
    pass

def plot_memory_savings(seq_lengths=[512, 1024, 2048, 4096]):
    """
    Plot KV cache size for different sequence lengths.

    Show:
        - MHA (32 heads)
        - GQA-8 (8 KV heads)
        - GQA-4 (4 KV heads)
        - MQA (1 KV head)
    """
    # YOUR CODE HERE
    pass
```

### Task 4: Evaluate Quality Trade-off

```python
def evaluate_perplexity_vs_kv_heads(model_variants, test_data):
    """
    Train models with different num_kv_heads.

    Plot: Perplexity vs num_kv_heads
    """
    # YOUR CODE HERE
    # Expected: Minimal perplexity increase even with 8→4 reduction
    pass
```

## 📊 Expected Results

### Memory Comparison (LLaMA-7B scale)

| Architecture | KV Heads | Cache Size (2K ctx) | Savings |
|-------------|----------|--------------------:|---------|
| MHA | 32 | 1024 MB | 0% |
| GQA-8 | 8 | 256 MB | 75% |
| GQA-4 | 4 | 128 MB | 87.5% |
| MQA | 1 | 32 MB | 96.9% |

### Quality Trade-off

| Architecture | Perplexity | Memory | Speed |
|-------------|------------|--------|-------|
| MHA | 10.5 | 100% | 1.0x |
| GQA-8 | 10.6 | 25% | 1.2x |
| GQA-4 | 10.8 | 12.5% | 1.3x |
| MQA | 11.5 | 3.1% | 1.5x |

## 🧪 Testing Your Understanding

**Check 1:** Why does reducing KV heads save memory but not computation?
**Check 2:** How do you "repeat" K/V to match Q heads?
**Check 3:** Why is MQA fastest but lowest quality?

## 🎯 Deliverables

- [ ] Implemented `GroupedQueryAttention`
- [ ] Compared MHA, GQA, MQA parameter counts
- [ ] Calculated KV cache memory savings
- [ ] Plotted memory vs num_kv_heads
- [ ] Measured perplexity trade-off

## 🚀 Next Steps
Move to **Phase 4: Project 11 - Expert Router (MoE)**!
