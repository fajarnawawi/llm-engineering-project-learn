# Project 13: Pretraining Objectives (Objective Arena)

## 🎯 Goal
Compare three pretraining objectives: Causal LM, Masked LM (BERT), and Prefix LM.

## 📚 Learning Objectives
- Implement three different training objectives
- Understand encoder-only vs decoder-only vs encoder-decoder
- Compare learning dynamics and task suitability
- Measure which objective is "harder" to learn

## 🔬 The "Hard Way" Lesson
**Causal masking is harder to learn than bi-directional masking.** But autoregressive models are more versatile.

## 🛠️ Implementation Tasks

### Task 1: Implement Causal Language Modeling (GPT-style)

```python
def causal_lm_loss(model, input_ids, labels):
    """
    Causal LM: Predict next token given all previous tokens.

    Masking: Lower triangular (can only see past)
    Loss: Cross-entropy on next-token prediction

    Args:
        input_ids: [batch, seq_len]
        labels: [batch, seq_len] (shifted by 1)
    """
    # YOUR CODE HERE
    # 1. Forward pass with causal mask
    # 2. Shift logits and labels
    # 3. Compute cross-entropy loss
    pass
```

### Task 2: Implement Masked Language Modeling (BERT-style)

```python
def masked_lm_loss(model, input_ids, mask_prob=0.15):
    """
    Masked LM: Predict masked tokens given bidirectional context.

    Masking: No causal mask (can see both directions)
    Objective: Predict randomly masked tokens

    Args:
        input_ids: [batch, seq_len]
        mask_prob: Probability of masking each token
    """
    # YOUR CODE HERE
    # 1. Randomly select mask_prob% of tokens
    # 2. Replace with [MASK] token
    # 3. Forward pass without causal mask
    # 4. Compute loss only on masked positions
    pass
```

### Task 3: Implement Prefix LM (T5-style)

```python
def prefix_lm_loss(model, input_ids, prefix_length):
    """
    Prefix LM: Hybrid between encoder and decoder.

    Prefix tokens: Bidirectional attention (encoder)
    Suffix tokens: Causal attention (decoder)

    Args:
        input_ids: [batch, seq_len]
        prefix_length: Length of encoder prefix
    """
    # YOUR CODE HERE
    # 1. Create custom mask:
    #    - Prefix tokens can attend to all prefix tokens
    #    - Suffix tokens can attend to prefix + causal past
    # 2. Forward pass
    # 3. Compute loss on suffix tokens only
    pass
```

### Task 4: Compare Training Dynamics

```python
def compare_objectives(train_data, epochs=10):
    """
    Train three models with different objectives on same data.

    Models:
        1. Causal LM (GPT)
        2. Masked LM (BERT)
        3. Prefix LM (T5)

    Compare:
        - Loss curves
        - Convergence speed
        - Final perplexity
        - Sample quality
    """
    # YOUR CODE HERE
    # Train all three models
    # Plot loss curves
    # Evaluate on held-out set
    pass
```

### Task 5: Task-Specific Evaluation

```python
def evaluate_on_tasks(models):
    """
    Test each model on different downstream tasks.

    Tasks:
        1. Text generation (Causal LM wins)
        2. Sentence classification (Masked LM wins)
        3. Question answering (Prefix LM balanced)
    """
    # YOUR CODE HERE
    pass
```

## 📊 Expected Results

### Loss Curves (Same Dataset, Same Model Size)

| Epoch | Causal LM | Masked LM | Prefix LM |
|-------|-----------|-----------|-----------|
| 1 | 4.2 | 3.8 | 4.0 |
| 5 | 2.8 | 2.1 | 2.5 |
| 10 | 2.3 | 1.7 | 2.0 |

**Observation:** Masked LM learns faster (can see full context).

### Task Performance

| Task | Causal LM | Masked LM | Prefix LM |
|------|-----------|-----------|-----------|
| Generation | ★★★ | ★ | ★★ |
| Classification | ★ | ★★★ | ★★ |
| QA | ★★ | ★★ | ★★★ |

## 🎯 Deliverables

- [ ] Implemented Causal LM objective
- [ ] Implemented Masked LM objective
- [ ] Implemented Prefix LM objective
- [ ] Trained all three models
- [ ] Compared loss curves
- [ ] Evaluated on downstream tasks

## 🚀 Next Steps
Move to **Project 14: Instruction Tuning**!
