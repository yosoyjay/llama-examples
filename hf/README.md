# Fine-tuning Llama2 7B/70B using Hugging Face with FSDP

Notes here follow the blog post [Fine-tuning Llama2 70B using PyTorch FSDP](https://huggingface.co/blog/ram-efficient-pytorch-fsdp) with alterations following discussion on GitHub issue [MFU calculation in ram-efficient-pytroch-fsdp](https://github.com/huggingface/blog/issues/1649).

## Setup

```bash
# Create a virtual environment and activate
# Install PyTorch nightly and Flash Attention
python3 -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
python3 -m pip install flash-attention

# Install requirements from the tutorial repo
git clone https://github.com/pacman100/DHS-LLM-Workshop.git
cd DHS-LLM-Workshop/chat_assistant/training
python3 -m pip install -r requirements.txt

# Changes in code (see tutorial.patch)
# 1. Sepecify bfloat16 as default type
# 2. Add a different cache dir to point to NVMe and set token to get weights
# review patch file
```

## Run tests

```bash
# 7B
sbatch -N <number-of-hosts> run-7b.sh
# 70 B
sbatch -N <number-of-hosts> run-70b.sh
```

## Preliminary results on H100

Time per iteration [per GPU] is the time taken to complete one iteration of the training loop taken from logs.  Stable after a couple of iterations.
MFU [per GPU] is estimated using the formula `MFU = (sequence_length/time_per_iterations * 6 * {7,70}e9) / 9.89e14` assuming H100 (3.12e14 for A100).


### 7B

Number of VMs | Sequence length | Time per iteration [per GPU] (s) | MFU [per GPU] |
--- | --- | --- | --- |
1  | 4096 | 0.53 | 0.36 |
2  | 4096 | 0.54 | 0.35 |
4  | 4096 | 0.56 | 0.34 |
8  | 4096 | 0.56 | 0.34 |
16 | 4096 | 0.59 | 0.34 |
32 | 4096 | 0.59 | 0.32 |
40 | 4096 | 0.59 | 0.32 |

### 70B

Number of VMs | Sequence length | Time per iteration [per GPU] (s) | MFU [per GPU] |
--- | --- | --- | --- |
1  | 4096 | 10.9 | 0.17 |
2  | 4096 | 5.9  | 0.32 |
4  | 4096 | 5.9  | 0.32 |
8  | 4096 | 5.9  | 0.32 |
16 | 4096 | 5.9  | 0.32 |
32 | 4096 | 6.4  | 0.29 |
40 | 4096 | 6.7  | 0.28 |
