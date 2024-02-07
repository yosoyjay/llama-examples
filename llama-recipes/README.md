# Fine-tuning Llama2 using llama-recipes

## Setup

```bash
# Create a virtual environment and activate
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes
```

## Run test on 7B

```bash
torchrun --nnodes 1 --nproc_per_node 8 ~/llama-recipes/examples/finetuning.py --enable_fsdp --model_name /mnt/resource_nvme/checkpoints/meta-llama/Llama-2-7b-hf --dist_checkpoint_root_folder ~/checkpoints --use_fast_kernels
```

## Estimated MFU on H100

### 7B

| Number of VMs | Sequence length |  Time per iteration | Measured MFU |
| --- | --- |  --- | --- |
| 1  | 4096 | 0.4  | 0.47 |