# LLM Engineering Project Index

Complete guide to all 16 "Hard Way" learning projects.

## 📋 Quick Navigation

### Phase I: The Atoms (Tokens & Embeddings)

| # | Project | Concept | Key Learning | Directory |
|---|---------|---------|--------------|-----------|
| 01 | Byte-Pair Smith | Tokenization | BPE algorithm, cross-language tokenization | `01_byte_pair_smith/` |
| 02 | Meaning Space | Embeddings | Skip-gram, Word2Vec, semantic geometry | `02_meaning_space/` |
| 03 | RoPE Animator | Positional Embeddings | Sinusoidal, Learned, RoPE rotation | `03_rope_animator/` |

### Phase II: The Architecture (Attention & Transformers)

| # | Project | Concept | Key Learning | Directory |
|---|---------|---------|--------------|-----------|
| 04 | Attention Lab | Self-Attention | Scaled dot-product, masking, scaling factor | `04_attention_lab/` |
| 05 | Block Builder | Transformers | Residual connections, Pre-LN vs Post-LN | `05_block_builder/` |
| 06 | Norm & Activation | Normalization | RMSNorm, SwiGLU, BatchNorm failure | `06_norm_activation/` |

### Phase III: Inference (Making it Fast)

| # | Project | Concept | Key Learning | Directory |
|---|---------|---------|--------------|-----------|
| 07 | Entropy Plotter | Sampling | Temperature, Top-K, Top-P, entropy | `07_entropy_plotter/` |
| 08 | KV Speedrun | KV Cache | O(n²) → O(n) optimization, memory profiling | `08_kv_speedrun/` |
| 09 | Sliding Window | Context Window | Window attention, receptive fields | `09_sliding_window/` |
| 10 | Grouped Query | GQA | Multi-query attention, memory savings | `10_grouped_query/` |

### Phase IV: Advanced & Scaling

| # | Project | Concept | Key Learning | Directory |
|---|---------|---------|--------------|-----------|
| 11 | Expert Router | Mixture of Experts | Sparse experts, load balancing, collapse | `11_expert_router/` |
| 12 | Int8 Quant | Quantization | AbsMax quantization, outlier handling | `12_int8_quant/` |

### Phase V: The Life Cycle (Training & Tuning)

| # | Project | Concept | Key Learning | Directory |
|---|---------|---------|--------------|-----------|
| 13 | Objective Arena | Pretraining | Causal LM, Masked LM, Prefix LM | `13_objective_arena/` |
| 14 | Instruct Tuner | Fine-tuning | Instruction tuning, base vs instruct | `14_instruct_tuner/` |
| 15 | PPO Loop | RLHF | Reward model, PPO, KL penalty | `15_ppo_loop/` |
| 16 | Scaling Laws | Capacity | Log-log scaling, Chinchilla optimal | `16_scaling_laws/` |

## 🎯 Learning Path

### Recommended Order
Follow the projects in numerical order (1-16). Each project builds on concepts from previous ones.

### Time Estimates
- **Beginner:** ~1 week per project (16 weeks total)
- **Intermediate:** ~3-4 days per project (8-10 weeks total)
- **Advanced:** ~1-2 days per project (4-5 weeks total)

## 📁 Project Structure

Each project directory contains:
```
XX_project_name/
├── README.md           # Learning objectives, tasks, expected results
├── starter.ipynb       # Jupyter notebook with code templates
└── (optional files)    # Additional resources
```

## ✅ Completion Criteria

For each project, you should:
1. ✅ Implement all core functions from scratch
2. ✅ Create required visualizations
3. ✅ Complete ablation/experiment tasks
4. ✅ Answer reflection questions
5. ✅ Achieve expected results

## 🛠️ Prerequisites

### Required Knowledge
- Python programming
- Basic linear algebra (vectors, matrices)
- Understanding of neural networks
- Familiarity with PyTorch

### Tools & Libraries
```bash
pip install torch numpy matplotlib seaborn plotly
pip install jupyter scikit-learn scipy
pip install streamlit gradio  # For Project 7
pip install wandb  # Optional: for experiment tracking
```

## 📊 Progress Tracking

Track your progress:

```markdown
- [ ] Phase I Complete (Projects 1-3)
- [ ] Phase II Complete (Projects 4-6)
- [ ] Phase III Complete (Projects 7-10)
- [ ] Phase IV Complete (Projects 11-12)
- [ ] Phase V Complete (Projects 13-16)
```

## 🎓 Certification

When you complete all 16 projects, you will have:
- Implemented every core LLM component from scratch
- Debugged and fixed broken architectures
- Visualized internal model behavior
- Trained models at multiple scales
- Understood the "why" behind every design choice

## 💡 Tips for Success

1. **Don't skip the "break it" experiments** - Failures teach more than successes
2. **Visualize everything** - If you can't plot it, you don't understand it
3. **Start simple** - Get the naive version working first
4. **Compare with references** - Check your outputs match expected behavior
5. **Document your findings** - Write down observations and insights

## 🤝 Community

- Share your visualizations and results
- Help others debug their implementations
- Propose improvements to the curriculum
- Contribute additional experiments

## 📚 Additional Resources

### Papers by Phase
**Phase I:**
- BPE: [Sennrich et al. 2016](https://arxiv.org/abs/1508.07909)
- Word2Vec: [Mikolov et al. 2013](https://arxiv.org/abs/1301.3781)
- RoPE: [Su et al. 2021](https://arxiv.org/abs/2104.09864)

**Phase II:**
- Attention: [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)
- RMSNorm: [Zhang & Sennrich 2019](https://arxiv.org/abs/1910.07467)

**Phase III:**
- Nucleus Sampling: [Holtzman et al. 2019](https://arxiv.org/abs/1904.09751)
- GQA: [Ainslie et al. 2023](https://arxiv.org/abs/2305.13245)

**Phase V:**
- InstructGPT: [Ouyang et al. 2022](https://arxiv.org/abs/2203.02155)
- Chinchilla: [Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556)

## 🚀 Next Steps After Completion

1. **Build a full model:** Combine all components into a complete LLM
2. **Train on real data:** Use datasets like WikiText, The Pile, etc.
3. **Optimize for production:** Implement Flash Attention, tensor parallelism
4. **Contribute to open source:** Apply knowledge to projects like nanoGPT, lit-llama
5. **Research:** Explore novel architectures and training methods

---

**Remember: The hard way is the only way to truly understand.**

Good luck on your journey! 🎉
