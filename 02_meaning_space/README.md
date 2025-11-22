# Project 2: The Geometry of Meaning

## 🎯 Goal
Understand embedding spaces by training a Word2Vec-style Skip-gram model from scratch.

## 📚 Learning Objectives
- Grasp how discrete tokens become continuous vectors
- Understand the concept of "semantic distance" via cosine similarity
- Visualize high-dimensional embeddings in 2D/3D
- Demonstrate the famous "King - Man + Woman ≈ Queen" analogy

## 🔬 The "Hard Way" Lesson
**One-Hot vectors are orthogonal and useless.** Dense embeddings learned from context create a geometric space where similar words cluster together.

## 🛠️ Implementation Tasks

### Task 1: Build Skip-Gram Word2Vec
Implement the Skip-gram architecture:
1. **Input:** Center word (one-hot encoded)
2. **Embedding Layer:** Maps word index to dense vector
3. **Context Prediction:** Predict surrounding words
4. **Loss:** Negative log-likelihood (or negative sampling)

**Key Components:**
```python
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Args:
            vocab_size: Number of unique words
            embedding_dim: Dimension of embedding vectors (typically 100-300)
        """
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, center_word: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE
        pass
```

**Training Loop:**
```python
def train_skipgram(model, corpus, window_size=2, epochs=10):
    """
    Train Skip-gram model.

    Args:
        corpus: List of word indices
        window_size: How many words to predict on each side
    """
    # YOUR CODE HERE
    pass
```

### Task 2: Visualize Embedding Space
Create multiple visualizations:

#### 2a. Cosine Distance Matrix
```python
def plot_cosine_matrix(embeddings: np.ndarray, words: List[str]):
    """
    Plot heatmap of cosine similarities between words.
    """
    # Calculate pairwise cosine similarities
    # Display as heatmap
    pass
```

#### 2b. 2D Projection (PCA/t-SNE)
```python
def plot_embeddings_2d(embeddings: np.ndarray, words: List[str], method='tsne'):
    """
    Project embeddings to 2D and plot.

    Args:
        method: 'pca' or 'tsne'
    """
    # YOUR CODE HERE
    pass
```

#### 2c. Semantic Analogies
Verify algebraic relationships:
```python
def test_analogy(embeddings, vocab, a, b, c):
    """
    Test: a is to b as c is to ?

    Example: king - man + woman ≈ queen

    Returns:
        Top 5 closest words to (b - a + c)
    """
    # YOUR CODE HERE
    pass
```

### Task 3: Compare One-Hot vs. Learned Embeddings
**Experiment:** Train two models side-by-side

#### Model A: One-Hot Encoding (Baseline)
- Each word is a sparse vector of size `vocab_size`
- Only one element is 1, rest are 0
- No compression, no semantic similarity

#### Model B: Learned Embeddings (Skip-gram)
- Each word is a dense vector of size `embedding_dim` (e.g., 128)
- Learned from context during training
- Captures semantic relationships

**Metrics to Compare:**
```python
def compare_models(one_hot_model, skipgram_model, test_data):
    """
    Compare:
        1. Loss curves during training
        2. Model size (parameters)
        3. Nearest neighbor quality
        4. Analogy accuracy
    """
    # YOUR CODE HERE
    pass
```

## 📊 Expected Results

### What You Should Observe:
1. **Loss curves:** Skip-gram converges to lower loss than one-hot
2. **Clusters:** Related words (e.g., colors, animals) group together in 2D plot
3. **Analogies:** Vector arithmetic produces semantically similar words
4. **Scalability:** One-hot encoding fails for large vocabularies (memory explosion)

### Sample Analogies to Test:
- `king - man + woman ≈ queen`
- `paris - france + italy ≈ rome`
- `walking - walk + swim ≈ swimming`
- `bigger - big + small ≈ smaller`

## 🧪 Testing Your Understanding

**Check 1:** Why are one-hot vectors orthogonal to each other? (Hint: dot product)

**Check 2:** What does a high cosine similarity (close to 1) between two word vectors mean?

**Check 3:** Why is `embedding_dim=50000` impractical even if vocab size is 50,000?

## 📖 Resources
- Word2Vec Paper: [Mikolov et al. 2013](https://arxiv.org/abs/1301.3781)
- Jay Alammar's Illustrated Word2Vec: [Blog Post](http://jalammar.github.io/illustrated-word2vec/)
- CS224N Lecture on Word Embeddings

## 📦 Data Sources
Use one of these small corpora for training:
- **TinyShakespeare:** ~1MB of Shakespeare text
- **WikiText-2:** Small Wikipedia subset
- **Your own:** Collect tweets, news articles, or code comments

```python
# Download TinyShakespeare
import urllib.request
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
urllib.request.urlretrieve(url, "tinyshakespeare.txt")
```

## ⚡ Bonus Challenges
1. **Negative Sampling:** Implement the optimization trick used in real Word2Vec
2. **Subword Embeddings:** Combine with BPE from Project 1
3. **CBOW:** Implement the alternative architecture (predict center from context)
4. **FastText:** Add character n-grams to handle OOV words
5. **GloVe:** Implement the matrix factorization alternative

## 🚀 Next Steps
Once complete, move to **Project 3: RoPE Animator** to learn how to encode position information into sequences.
