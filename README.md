
# LLM Engineering: The Hard Way

> "You don't know how it works until you can fix it when it breaks."

This repository contains 16 implementations of core LLM concepts built from scratch (mostly raw PyTorch). No `AutoModel`, no black boxes. Just tensors and algebra.

## 🗺️ The Journey


### Phase I: The Atoms

| Project | Concept | The "Hard Way" Lesson | Status
| -- | -- | -- | -- |
| 01. Byte-Pair Smith | Tokenization | Merging rules are just statistics. Visualized token drift. | ⬜
| 02. Meaning Space | Embeddings | One-Hot vectors are orthogonal and useless. Cosine similarity is king. | ⬜
| 03. RoPE Animator | Positional Embeddings | Absolute positions fail on long docs. Rotation generalizes better. | ⬜

  
### Phase II: The Architecture

| Project | Concept | The "Hard Way" Lesson | Status
| -- | -- | -- | -- |
| 04. Attention Lab | Self-Attention | Scaling factors prevent softmax saturation. | ⬜
| 05. Block Builder | Transformers | Residual streams are the "information highway." | ⬜
| 06. Norm & Activation | Normalization | LayerNorm centers gradients; Pre-LN converges faster. | ⬜

### Phase III: Inference

| Project | Concept | The "Hard Way" Lesson | Status
| -- | -- | -- | -- |
| 07. Entropy Plotter | Sampling | Temperature controls the "sharpness" of the softmax distribution. | ⬜
| 08. KV Speedrun | KV Cache | The bottleneck is memory bandwidth, not compute. | ⬜
| 09. Sliding Window | Context Window | Memory grows quadratically unless you restrict attention scope. | ⬜
| 10. Grouped Query | GQA | Sharing KV heads saves massive VRAM with minimal perplexity loss. | ⬜

  

### Phase IV: Advanced & Scaling

| Project | Concept | The "Hard Way" Lesson | Status
| -- | -- | -- | -- |
| 11. Expert Router | Mixture of Experts | Experts collapse if you don't penalize uneven routing. | ⬜
| 12. Int8 Quant | Quantization | Outliers in activation matrices destroy quantization precision. | ⬜
| 13. Objective Arena | Pretraining | Causal masking is harder to learn than bi-directional masking. | ⬜
| 14. Instruct Tuner | Fine-tuning | Base models mimic; Instruct models obey. | ⬜
| 15. PPO Loop | RLHF | The Reward Model is easily gamed without KL penalties. | ⬜
| 16. Scaling Laws | Capacity | Loss scales linearly with the log of compute/parameters. | ⬜

## 🛠️ Tech Stack

-   **Core:** PyTorch, NumPy
    
-   **Visualization:** Matplotlib, Plotly (for 3D embeddings/heatmaps)
    
-   **UI:** Streamlit (for the Sampling Dashboard)
    
-   **Tracking:** Weights & Biases (for loss curves)
    

## 📉 How to run

Each folder contains a standalone Jupyter Notebook.

```
# Example: Run the Attention visualizer
cd 04_attention_lab
jupyter notebook attention_vis.ipynb

```

## 🤝 Contributing

If you find a bug in my math (or my backprop implementation), open an Issue. I am purposely implementing things manually to learn, so bugs are expected!
