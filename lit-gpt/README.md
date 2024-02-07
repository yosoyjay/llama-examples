# Pretraining Llama2 using lit-gpt

Notes here follow example [openwebtext_trainer.py](https://github.com/Lightning-AI/lit-gpt/blob/main/pretrain/openwebtext_trainer.py) which pretrains a model on OpenWebText using lit-gpt.  Here we use Llama2 as an example.

## Setup

```bash
# Create a virtual environment and activate
# Install lit-gpt
git clone https://github.com/Lightning-AI/lit-gpt
cd lit-gpt
python3 -m pip install -r requirements-all.txt

# Changes in code from upstream repo (see code.path)
- Change `openwebtext_trainer.py` to use Llama2
- Change workers for dataloader
- Edits to hyperparameters

```

## Run tests

```bash
# 70B
sbatch -N <number-of-hosts> run-70b.slurm
```

## Preliminary results on H100

### 700 B

Number of VMs | Sequence length |  Measured MFU |
--- | --- |  --- |
8  | 4096 | 0.22 |
16 | 4096 | 0.18 |
32 | 4096 | 0.18 |
40 | 4096 | 0.11 |