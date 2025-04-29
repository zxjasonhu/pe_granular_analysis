export OMP_NUM_THREADS=1
export NCCL_P2P_LEVEL=NVL

torchrun \
    --standalone \
    --nnodes 1 \
    --nproc_per_node 2 \ # 2 GPUs per node
    run.py