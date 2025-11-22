# Project 15: RLHF (The PPO Loop)

## 🎯 Goal
Implement Reinforcement Learning from Human Feedback using Proximal Policy Optimization.

## 📚 Learning Objectives
- Build a reward model from preference data
- Implement PPO for language model fine-tuning
- Understand KL-divergence penalty
- Observe reward hacking without constraints

## 🔬 The "Hard Way" Lesson
**The Reward Model is easily gamed without KL penalties.** RLHF needs careful constraint tuning.

## 🛠️ Implementation Tasks

### Task 1: Build Reward Model

```python
class RewardModel(nn.Module):
    """
    Scalar reward predictor from language model.

    Takes output from base LM and adds a scalar head.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # Freeze base model (optional)
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Add reward head
        self.reward_head = nn.Linear(base_model.config.d_model, 1)

    def forward(self, input_ids):
        """
        Returns scalar reward for the sequence.
        """
        # YOUR CODE HERE
        # 1. Get final hidden state from base model
        # 2. Pass through reward head
        # 3. Return scalar
        pass
```

### Task 2: Train Reward Model on Preferences

```python
def create_preference_dataset():
    """
    Create dataset of (prompt, response_A, response_B, preference).

    Format:
        {
            "prompt": "Explain gravity",
            "response_A": "Gravity is a force...",
            "response_B": "Idk, things fall lol",
            "preference": "A"  # A is preferred
        }
    """
    # YOUR CODE HERE
    pass

def reward_model_loss(reward_model, prompt, response_A, response_B, preference):
    """
    Bradley-Terry preference loss.

    P(A > B) = σ(R(A) - R(B))

    Loss = -log P(preferred > rejected)
    """
    # YOUR CODE HERE
    # 1. Get rewards for both responses
    # 2. Compute preference probability
    # 3. Compute cross-entropy loss
    pass
```

### Task 3: Implement PPO Training Loop

```python
def ppo_step(policy_model, reward_model, ref_model, prompts, kl_coef=0.1):
    """
    One PPO update step.

    Steps:
        1. Generate responses from current policy
        2. Compute rewards from reward model
        3. Compute KL penalty vs reference model
        4. Compute PPO loss
        5. Update policy

    Args:
        policy_model: Current model being trained
        reward_model: Frozen reward predictor
        ref_model: Original SFT model (frozen reference)
        prompts: Batch of prompts
        kl_coef: Weight for KL penalty
    """
    # YOUR CODE HERE
    # 1. Sample responses from policy
    # 2. Compute reward(prompt, response)
    # 3. Compute KL(policy || ref)
    # 4. Combined reward = reward - kl_coef * KL
    # 5. PPO update
    pass

def compute_kl_penalty(policy_logits, ref_logits):
    """
    KL divergence between policy and reference model.

    KL(P || Q) = Σ P(x) log(P(x) / Q(x))
    """
    # YOUR CODE HERE
    pass
```

### Task 4: Visualize Reward vs KL Trade-off

```python
def plot_reward_kl_tradeoff(training_history):
    """
    Plot reward and KL penalty over training steps.

    Show:
        1. Reward increasing
        2. KL divergence increasing
        3. Combined objective
    """
    # YOUR CODE HERE
    # Create line plot with:
    # - Raw reward
    # - KL penalty
    # - Combined (reward - kl_coef * KL)
    pass
```

### Task 5: Demonstrate Reward Hacking

```python
def demonstrate_reward_hacking():
    """
    Train with kl_coef=0 (no KL penalty).

    Show that model learns to:
        - Generate nonsense that fools reward model
        - Repeat high-reward tokens
        - Produce incoherent but high-scoring text
    """
    # YOUR CODE HERE
    # Train two models:
    # 1. With KL penalty (kl_coef=0.1)
    # 2. Without KL penalty (kl_coef=0)

    # Compare outputs
    pass
```

## 📊 Expected Results

### Training Curves

| Step | Reward | KL | Combined |
|------|--------|-----|----------|
| 0 | 0.0 | 0.0 | 0.0 |
| 100 | 0.5 | 0.1 | 0.49 |
| 500 | 1.2 | 0.5 | 1.15 |
| 1000 | 1.8 | 1.0 | 1.70 |

**With KL penalty:** Reward increases but KL stays bounded.

### Without KL Penalty (Reward Hacking)

| Step | Reward | KL | Output Quality |
|------|--------|-----|----------------|
| 0 | 0.0 | 0.0 | Good |
| 100 | 2.0 | 5.0 | Degrading |
| 500 | 5.0 | 20.0 | Nonsense |

**Observation:** Model drifts far from reference, produces garbage.

## 🧪 Testing Your Understanding

**Check 1:** Why do we need a reference model in RLHF?
**Check 2:** What happens if kl_coef is too large?
**Check 3:** Why can't we just maximize reward directly?
**Check 4:** How does PPO differ from vanilla policy gradient?

## 🎯 Deliverables

- [ ] Implemented `RewardModel`
- [ ] Trained reward model on preference data
- [ ] Implemented PPO training loop
- [ ] Computed KL penalty
- [ ] Visualized reward vs KL trade-off
- [ ] Demonstrated reward hacking without KL penalty

## 🚀 Next Steps
Move to **Project 16: Scaling Laws** - the final project!
