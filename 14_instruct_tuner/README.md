# Project 14: Finetuning & Instruction Tuning

## 🎯 Goal
Transform a base language model into an instruction-following assistant.

## 📚 Learning Objectives
- Create an instruction-tuning dataset
- Implement supervised fine-tuning (SFT)
- Understand the difference between base and instruct models
- Measure behavioral changes after instruction tuning

## 🔬 The "Hard Way" Lesson
**Base models mimic; Instruct models obey.** Instruction tuning changes the model's completion behavior.

## 🛠️ Implementation Tasks

### Task 1: Create Instruction Dataset

```python
def create_instruction_dataset():
    """
    Format examples as instruction-response pairs.

    Format:
        {
            "instruction": "Summarize the following text: <text>",
            "response": "<summary>"
        }

    Categories:
        - Question answering
        - Summarization
        - Code generation
        - Math problems
        - Creative writing
    """
    # YOUR CODE HERE
    examples = [
        {
            "instruction": "What is the capital of France?",
            "response": "The capital of France is Paris."
        },
        {
            "instruction": "Write a Python function to reverse a string.",
            "response": "def reverse_string(s):\n    return s[::-1]"
        },
        # Add more examples
    ]

    return examples
```

### Task 2: Format for Training

```python
def format_instruction_example(instruction, response, tokenizer):
    """
    Format instruction-response pair with special tokens.

    Template:
        [INST] {instruction} [/INST] {response}

    Or:
        Below is an instruction...
        ### Instruction:
        {instruction}
        ### Response:
        {response}
    """
    # YOUR CODE HERE
    pass

class InstructionDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        # YOUR CODE HERE
        # 1. Format example
        # 2. Tokenize
        # 3. Create labels (mask instruction part, only train on response)
        pass
```

### Task 3: Supervised Fine-Tuning

```python
def finetune_instruct_model(base_model, instruct_dataset, epochs=3):
    """
    Fine-tune base model on instruction data.

    Training details:
        - Lower learning rate than pretraining (1e-5)
        - Fewer epochs (3-5)
        - Loss only on response tokens
    """
    # YOUR CODE HERE
    # 1. Freeze some layers (optional)
    # 2. Train with masked loss
    # 3. Evaluate on held-out instructions
    pass
```

### Task 4: Compare Base vs Instruct Behavior

```python
def compare_base_vs_instruct():
    """
    Test the same prompt on base and instruct models.

    Example:
        Prompt: "What is the capital of France?"

        Base model: "What is the capital of Germany? What is..."
        (continues the pattern, doesn't answer)

        Instruct model: "The capital of France is Paris."
        (answers the question)
    """
    # YOUR CODE HERE
    prompts = [
        "What is the capital of France?",
        "Write a haiku about AI",
        "Explain quantum computing in simple terms",
    ]

    for prompt in prompts:
        base_output = generate(base_model, prompt)
        instruct_output = generate(instruct_model, prompt)

        print(f"Prompt: {prompt}")
        print(f"Base: {base_output}")
        print(f"Instruct: {instruct_output}")
        print()
```

### Task 5: LoRA Fine-Tuning (Bonus)

```python
class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation: Add trainable low-rank matrices.

    W_adapted = W_frozen + A @ B

    where A: [d, r], B: [r, d], r << d
    """
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.frozen_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.frozen_weight.requires_grad = False

        # Low-rank adapters
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.randn(rank, out_features))

    def forward(self, x):
        # YOUR CODE HERE
        # output = x @ (W_frozen + A @ B)^T
        pass
```

## 📊 Expected Results

### Behavioral Comparison

**Prompt:** "Translate 'hello' to Spanish"

| Model | Output |
|-------|--------|
| Base | "Translate 'goodbye' to French. Translate 'thank you'..." |
| Instruct | "'Hello' in Spanish is 'Hola'." |

**Prompt:** "What is 2+2?"

| Model | Output |
|-------|--------|
| Base | "What is 3+3? What is 4+4? The answer to..." |
| Instruct | "2+2 equals 4." |

### Training Metrics

| Metric | Base Model | After Instruction Tuning |
|--------|------------|-------------------------|
| Instruction Following Rate | ~20% | ~95% |
| Completion Behavior | Mimics patterns | Answers questions |
| Refusal Rate | 0% | 5-10% (learned safety) |

## 🎯 Deliverables

- [ ] Created instruction dataset (100+ examples)
- [ ] Implemented instruction formatting
- [ ] Fine-tuned base model on instructions
- [ ] Compared base vs instruct behavior
- [ ] Demonstrated behavioral change
- [ ] (Bonus) Implemented LoRA fine-tuning

## 🚀 Next Steps
Move to **Project 15: RLHF (PPO Loop)**!
