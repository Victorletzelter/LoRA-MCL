# Installation Guide for peft-mcl

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/Victorletzelter/LoRA-MCL.git
cd LoRA-MCL
pip install -e .
```

## Requirements

- Python >= 3.10
- PyTorch == 2.6.0
- Transformers == 4.49.0
- PEFT == 0.14.0
- rootutils == 1.0.7

## Verifying Installation

After installation, verify it works:

```python
from peft_mcl import get_peft_mcl, MCLTrainer, patch_peft_for_mcl
print("✓ peft-mcl successfully installed!")
```

## Troubleshooting

### PyTorch Installation

If you need GPU support, install PyTorch first:

```bash
# For CUDA 12.4 (recommended)
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### HuggingFace Token

For most models, you'll need a HuggingFace access token:

```bash
export HF_TOKEN=your_token_here
```

Or set it in your Python script:

```python
from huggingface_hub import login
login(token="your_token_here")
```

