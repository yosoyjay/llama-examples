#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=hpc
#SBATCH --gres=gpu:8
set -x -e

module load mpi/hpcx

# Activate virutal environment conda environment
source /shared/home/cycleadmin/miniconda3/etc/profile.d/conda.sh
conda activate hf

# HF_HOME and HF_TOKEN
source .hf_settings
echo "HF_HOME: $HF_HOME"

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
 export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

# AWS specific
#export NCCL_PROTO=simple
#export RDMAV_FORK_SAFE=1
#export FI_EFA_FORK_SAFE=1
#export FI_EFA_USE_DEVICE_RDMA=1
#export FI_PROVIDER=efa
#export FI_LOG_LEVEL=1
#export NCCL_IB_DISABLE=1
#export NCCL_SOCKET_IFNAME=ens

# Azure specific
export OMPI_MCA_plm_rsh_no_tree_spawn=1
export OMPI_MCA_plm_rsh_num_concurrent=800
export OMPI_MCA_coll_hcoll_enable=0
export UCX_TLS=rc
export UCX_NET_DEVICES=mlx5_ib0:1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=warn
export NCCL_NET_GDR_LEVEL=5
export NCCL_MIN_NCHANNELS=32
export NCCL_TOPO_FILE=/opt/microsoft/ndv5-topo.xml
export NCCL_ALGO=CollnetChain,NVLS
export NCCL_COLLNET_ENABLE=1
export SHARP_COLL_ENABLE_SAT=1
export SHARP_COLL_LOG_LEVEL=1
export SHARP_COLL_ENABLE_PCI_RELAXED_ORDER1NG=1
export SHARP_SMX_UCX_INTERFACE=mlx5_ib0:1


echo "START TIME: $(date)"

# CHANGE TO CUMMULATIVELY LOG OUTPUTS
LOG_PATH="main_log-${SLURM_NNODES}-${SLURM_JOBID}.txt"

GPUS_PER_NODE=8
NNODES=$SLURM_NNODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

# so processes know who to talk to
# Can query for IB on Azure.
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

# OTHER LAUNCHERS CAN BE USED HERE
export LAUNCHER="accelerate launch \
    --config_file configs/fsdp_config.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    "
# Note: it is important to escape `$SLURM_PROCID` since we want the srun on each node to evaluate this variable

export PROGRAM="\
train.py \
    --model_name "meta-llama/Llama-2-70b-hf" \
    --dataset_name "HuggingFaceH4/ultrachat_200k" \
    --chat_template_format "chatml" \
    --add_special_tokens False \
    --append_concat_token False \
    --splits "train_sft,test_sft" \
    --max_seq_len 4096 \
    --num_train_epochs 1 \
    --logging_steps 5 \
    --log_level "info" \
    --logging_strategy "steps" \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --bf16 True \
    --packing True \
    --learning_rate 2e-5 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --output_dir "llama2-output" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --use_reentrant True \
    --dataset_text_field "content" \
    --use_flash_attn True \
    --ddp_timeout 5400 \
    --optim paged_adamw_32bit
"

export CMD="$LAUNCHER $PROGRAM"

srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_PATH

echo "END TIME: $(date)"
