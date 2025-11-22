# Project 12: Quantization (Int8)

## 🎯 Goal
Shrink model weights from FP32 to Int8 while measuring quality degradation.

## 📚 Learning Objectives
- Implement AbsMax quantization scheme
- Understand dynamic vs static quantization
- Measure perplexity degradation
- Visualize weight distribution before/after quantization

## 🔬 The "Hard Way" Lesson
**Outliers in activation matrices destroy quantization precision.** Careful outlier handling is critical.

## 🛠️ Implementation Tasks

### Task 1: Implement AbsMax Quantization

```python
def absmax_quantize(tensor: torch.Tensor, n_bits: int = 8):
    """
    Simple symmetric quantization.

    Q = round(X / scale)
    where scale = max(|X|) / (2^(n_bits-1) - 1)

    Args:
        tensor: FP32 weights or activations
        n_bits: Target bit width (typically 8)

    Returns:
        quantized: Int8 tensor
        scale: Dequantization scale factor
    """
    # YOUR CODE HERE
    # 1. Find max absolute value
    # 2. Compute scale = max_abs / (2^(n_bits-1) - 1)
    # 3. Quantize: round(tensor / scale)
    # 4. Clamp to int8 range [-127, 127]

    pass

def dequantize(quantized: torch.Tensor, scale: float):
    """
    Dequantize back to FP32.

    X_approx = Q * scale
    """
    # YOUR CODE HERE
    pass
```

### Task 2: Quantize Linear Layer

```python
class QuantizedLinear(nn.Module):
    """
    Linear layer with int8 weights.

    Stores weights as int8, performs computation in FP32.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.weight_quant = None
        self.weight_scale = None

    def quantize_weights(self):
        """Convert FP32 weights to int8."""
        self.weight_quant, self.weight_scale = absmax_quantize(self.weight)
        # Delete FP32 weights to save memory
        # self.weight = None

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, in_features] in FP32

        Returns:
            output: [batch, seq_len, out_features] in FP32
        """
        # YOUR CODE HERE
        # 1. Dequantize weights
        # 2. Perform linear operation
        # 3. Add bias
        pass
```

### Task 3: Visualize Weight Distributions

```python
def plot_weight_distributions(fp32_weights, int8_weights, scale):
    """
    Compare weight distributions before and after quantization.

    Show:
        1. Histogram of FP32 weights
        2. Histogram of dequantized int8 weights
        3. Quantization error distribution
    """
    # YOUR CODE HERE
    # Plot 3 subplots:
    # - Original FP32
    # - Quantized (dequantized back to FP32)
    # - Error (FP32 - quantized)

    pass
```

### Task 4: Measure Perplexity Degradation

```python
def compare_model_quality(model_fp32, model_int8, test_data):
    """
    Evaluate model before and after quantization.

    Metrics:
        - Perplexity
        - Accuracy (if classification)
        - Generation quality (if language model)
    """
    # YOUR CODE HERE
    # 1. Evaluate FP32 model
    # 2. Quantize to int8
    # 3. Evaluate int8 model
    # 4. Compare results

    pass
```

### Task 5: Handle Outliers

```python
def detect_outliers(activations: torch.Tensor, threshold: float = 6.0):
    """
    Detect outlier values in activation matrices.

    Outliers are values > threshold * std deviation from mean.
    """
    # YOUR CODE HERE
    pass

def plot_outlier_impact():
    """
    Show how outliers affect quantization error.

    Compare:
        - Quantizing with outliers
        - Quantizing after outlier clipping
        - Per-channel quantization (different scale per output channel)
    """
    # YOUR CODE HERE
    pass
```

## 📊 Expected Results

### Weight Distribution Analysis

**Original FP32:**
- Range: [-0.5, 0.5]
- Mean: ~0.0
- Std: ~0.1

**Int8 Quantized:**
- Range: [-127, 127] → scaled back to [-0.5, 0.5]
- Discretization: 256 unique values
- Quantization noise: std ~0.002

### Perplexity Comparison

| Model | Perplexity | Size | Speed |
|-------|------------|------|-------|
| FP32 | 12.5 | 100% | 1.0x |
| Int8 | 12.8 | 25% | 2.0x |
| Int4 | 14.2 | 12.5% | 3.5x |

**Typical degradation:** <3% increase in perplexity for int8.

### Outlier Analysis

**Without outlier handling:**
- Max activation: 47.3 (outlier!)
- Quantization error: High for non-outlier values

**With outlier clipping:**
- Max activation: 6.0 (clipped)
- Quantization error: Reduced

## 🧪 Testing Your Understanding

**Check 1:** Why is symmetric quantization simpler than asymmetric?
**Check 2:** What is the difference between static and dynamic quantization?
**Check 3:** Why do outliers hurt quantization?
**Check 4:** Can you quantize embeddings? Why or why not?

## 🎯 Deliverables

- [ ] Implemented `absmax_quantize()` and `dequantize()`
- [ ] Implemented `QuantizedLinear` layer
- [ ] Visualized weight distributions
- [ ] Measured perplexity degradation
- [ ] Analyzed outlier impact
- [ ] Compared model sizes

## ⚡ Bonus Challenges

1. **GPTQ:** Implement layer-wise quantization
2. **QLoRA:** Quantize base model, finetune with LoRA
3. **Mixed Precision:** Keep some layers in FP16
4. **Per-Channel Quantization:** Different scale per output channel

## 🚀 Next Steps
Move to **Phase 5: Project 13 - Objective Arena** to explore training objectives!
