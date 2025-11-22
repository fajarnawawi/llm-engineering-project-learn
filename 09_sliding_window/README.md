# Project 9: Long-Context & Sliding Windows

## 🎯 Goal
Handle infinite-length streams by restricting attention to a local window.

## 📚 Learning Objectives
- Implement sliding window attention
- Understand the trade-off between context and memory
- Visualize effective receptive fields
- Demonstrate failure on long-range dependencies

## 🔬 The "Hard Way" Lesson
**Memory grows quadratically unless you restrict attention scope.** Sliding windows enable streaming inference.

## 🛠️ Implementation Tasks

### Task 1: Implement Sliding Window Attention

```python
def sliding_window_attention(Q, K, V, window_size=128):
    """
    Attention with sliding window mask.

    Each position can only attend to the previous 'window_size' tokens.

    Args:
        Q, K, V: [batch, seq_len, d_k]
        window_size: Maximum attention distance

    Returns:
        output: [batch, seq_len, d_v]
    """
    # YOUR CODE HERE
    # Create window mask: 1s within window, 0s outside
    pass

def create_sliding_window_mask(seq_len: int, window_size: int):
    """
    Create band-diagonal mask.

    Example (window_size=3, seq_len=6):
        [[1, 0, 0, 0, 0, 0],
         [1, 1, 0, 0, 0, 0],
         [1, 1, 1, 0, 0, 0],
         [0, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1]]
    """
    # YOUR CODE HERE
    pass
```

### Task 2: Visualize Effective Receptive Field

```python
def plot_receptive_field(window_size=128, seq_len=512, num_layers=4):
    """
    Show how receptive field grows with layer depth.

    Layer 1: window_size
    Layer 2: window_size * 2
    Layer L: window_size * L
    """
    # YOUR CODE HERE
    # Plot heatmap showing which positions can influence output
    pass
```

### Task 3: Passkey Retrieval Task

**Test long-range recall ability**

```python
def create_passkey_dataset(seq_len=2000, key_position=0):
    """
    Create test sequence:
        "The passkey is 12345. <filler text> What is the passkey?"

    Args:
        seq_len: Total sequence length
        key_position: Where to insert the passkey (0 = beginning)

    Returns:
        sequence, target
    """
    # YOUR CODE HERE
    pass

def test_passkey_retrieval(model, window_sizes=[128, 256, 512, 1024]):
    """
    Test if model can recall passkey at different distances.

    Expected: Fails when key_position > window_size
    """
    # YOUR CODE HERE
    pass
```

### Task 4: Memory Comparison

```python
def compare_memory_usage(seq_lengths=[512, 1024, 2048, 4096]):
    """
    Compare VRAM usage:
        - Full attention: O(n²)
        - Sliding window: O(n × window_size)

    Plot memory vs sequence length for both.
    """
    # YOUR CODE HERE
    pass
```

## 📊 Expected Results

### Receptive Field Growth

| Layer | Receptive Field |
|-------|----------------|
| 1 | 128 tokens |
| 2 | 256 tokens |
| 4 | 512 tokens |
| 8 | 1024 tokens |

### Passkey Retrieval

| Window Size | Max Recall Distance | Success Rate |
|------------|--------------------:|--------------|
| 128 | ~512 tokens (4 layers) | 95% |
| 256 | ~1024 tokens | 90% |
| 512 | ~2048 tokens | 85% |
| Full | Unlimited | 100% |

## 🎯 Deliverables

- [ ] Implemented sliding window attention
- [ ] Created window mask visualization
- [ ] Plotted receptive field growth
- [ ] Implemented passkey retrieval test
- [ ] Demonstrated recall failure beyond window
- [ ] Compared memory usage vs full attention

## 🚀 Next Steps
Move to **Project 10: Grouped Query Attention (GQA)**!
