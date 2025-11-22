# Project 1: The Tokenizer Smith

## 🎯 Goal
Build a Byte-Pair Encoding (BPE) tokenizer from scratch without using HuggingFace's `tokenizers` library.

## 📚 Learning Objectives
- Understand how text is converted into discrete tokens
- Grasp the statistical nature of BPE merging rules
- Visualize token boundaries and "token drift" across languages
- Measure tokenization efficiency

## 🔬 The "Hard Way" Lesson
**Merging rules are just statistics.** Tokenization isn't magic—it's frequency counting and greedy pair merging.

## 🛠️ Implementation Tasks

### Task 1: Build the BPE Algorithm
Implement a script that:
1. Takes raw text as input
2. Initializes vocabulary with all unique characters
3. Counts frequency of all adjacent character/token pairs
4. Merges the most frequent pair into a new token
5. Repeats until desired vocabulary size is reached

**Key Functions to Implement:**
```python
def get_pair_frequencies(tokens: List[str]) -> Dict[Tuple[str, str], int]:
    """Count frequency of all adjacent pairs."""
    pass

def merge_pair(tokens: List[str], pair: Tuple[str, str], new_token: str) -> List[str]:
    """Replace all occurrences of pair with new_token."""
    pass

def train_bpe(text: str, vocab_size: int) -> Dict[str, int]:
    """Main training loop."""
    pass

def encode(text: str, merges: List[Tuple[str, str]]) -> List[str]:
    """Tokenize text using learned merge rules."""
    pass
```

### Task 2: Color-Coded Visualizer
Create a visualization tool that:
- Takes a sentence as input
- Color-codes each token chunk differently
- Shows how words get split into subword units
- Displays token boundaries clearly

**Options:**
- HTML output with `<span>` tags and CSS colors
- Matplotlib bar chart with colored segments
- Interactive Plotly visualization

**Example Output:**
```
"unhappiness" → ["un", "happiness"] or ["un", "happy", "ness"]
```

### Task 3: Ablation Experiment
**Hypothesis:** Tokenizers trained on one language perform poorly on others.

**Experiment:**
1. Train a BPE tokenizer on English code/text
2. Attempt to tokenize Chinese text with the same tokenizer
3. Measure:
   - Unknown token (`<unk>`) rate
   - Tokens-per-character ratio
   - Tokens-per-word efficiency
4. Compare with a tokenizer trained on Chinese text

**Metrics to Calculate:**
```python
def measure_efficiency(text: str, tokens: List[str]) -> Dict[str, float]:
    """
    Returns:
        - unk_rate: Percentage of unknown tokens
        - compression_ratio: len(text) / len(tokens)
        - avg_token_length: Average characters per token
    """
    pass
```

## 📊 Expected Results

### What You Should Observe:
1. **Frequent pairs merge first:** Common words like "the", "ing", "ed" become single tokens
2. **Cross-language failure:** English tokenizer creates many small tokens for Chinese
3. **Vocabulary efficiency:** Higher vocab size = longer tokens but bigger model

### Deliverables:
- [ ] Working BPE implementation (`bpe_tokenizer.py`)
- [ ] Visualization of token splits (`visualize_tokens.html` or `.png`)
- [ ] Ablation study results (`ablation_results.md`)
- [ ] Comparison table showing efficiency metrics

## 🧪 Testing Your Understanding

**Check 1:** Can you explain why BPE creates subwords instead of word-level tokens?

**Check 2:** What happens if you set `vocab_size = 256` (only characters)?

**Check 3:** Why do multilingual models need larger vocabularies?

## 📖 Resources
- Original BPE Paper: [Sennrich et al. 2016](https://arxiv.org/abs/1508.07909)
- SentencePiece: [Kudo & Richardson 2018](https://arxiv.org/abs/1808.06226)
- HuggingFace Tokenizers Course: (for reference AFTER implementation)

## ⚡ Bonus Challenges
1. Implement **WordPiece** (used by BERT) as a comparison
2. Add special tokens: `<BOS>`, `<EOS>`, `<PAD>`, `<UNK>`
3. Save/load tokenizer as JSON
4. Benchmark encoding speed: tokens/second

## 🚀 Next Steps
Once complete, move to **Project 2: Meaning Space** to learn how these tokens become dense vectors.
