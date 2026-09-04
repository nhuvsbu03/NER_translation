#!/bin/bash
set -e
export VLLM_USE_V2_MODEL_RUNNER=1
export HF_TOKEN=$(cat ~/.hf_token)

nohup vllm serve nvidia/diffusiongemma-26B-A4B-it-NVFP4 \
  --trust-remote-code \
  --max-num-seqs 4 \
  --attention-backend TRITON_ATTN \
  > /root/vllm_server.log 2>&1 &

echo "vLLM server starting in background (PID $!), logging to /root/vllm_server.log"
