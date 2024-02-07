#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --mem=1536G

module load mpi/hpcx

source /shared/home/cycleadmin/miniconda3/etc/profile.d/conda.sh
conda activate lit-gpt

# From NCCL test from IB engineering team
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

echo "Python: $(python --version), $(which python)"
# Workaround because model code uses globals for batch_size...
BATCH_SIZE=${1:-6}
export BATCH_SIZE
DEVICES=8
NUM_NODES=${SLURM_NNODES:-1}
echo "BATCH_SIZE: $BATCH_SIZE, NUM_NODES: $NUM_NODES"
srun --mpi=pmix python pretrain/openwebtext_trainer.py --devices $DEVICES --num_nodes $NUM_NODES --batch_size $BATCH_SIZE --precision 'bf16-true'