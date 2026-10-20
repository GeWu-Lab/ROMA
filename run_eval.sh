#!/bin/bash
# Run the ROMA-Bench evaluation of ROMA-7B on multiple GPUs in parallel.
# Each process is pinned to one GPU and handles its own shard of the QA set
# (all_qa[shard_id::num_shards]), batching batch_size QA at a time on that GPU.
# Results are written to <output_dir>/eval_roma_bench_<shard_id>.json and
# <output_dir>/error_roma_bench_<shard_id>.json (output_dir defaults to
# ROMA_BENCH_RESULTS_DIR in config/global_name.py), and saved every
# save_every finished QA on each GPU. Run check_eval_results.py afterwards
# for the final accuracy.
#
# Usage: bash run_eval.sh [gpus] [batch_size] [save_every]
#   e.g. bash run_eval.sh "0,1,2,3" 8 50

GPUS=${1:-"0,1,2,3,4,5,6,7"}
BATCH_SIZE=${2:-8}
SAVE_EVERY=${3:-50}

IFS=',' read -ra GPU_LIST <<< "$GPUS"
NUM_SHARDS=${#GPU_LIST[@]}

mkdir -p logs

echo "Running eval_roma_7b.py (batch_size=$BATCH_SIZE, save_every=$SAVE_EVERY) across $NUM_SHARDS shard(s): $GPUS"

for shard_id in "${!GPU_LIST[@]}"; do
    gpu=${GPU_LIST[$shard_id]}
    echo "Launching shard $shard_id on GPU $gpu"
    python -u eval_roma_7b.py \
        --gpu "$gpu" \
        --shard_id "$shard_id" \
        --num_shards "$NUM_SHARDS" \
        --batch_size "$BATCH_SIZE" \
        --save_every "$SAVE_EVERY" \
        > "logs/eval_roma_bench_shard${shard_id}_gpu${gpu}.log" 2>&1 &
done

wait
echo "All $NUM_SHARDS shards finished. Run: python check_eval_results.py"
