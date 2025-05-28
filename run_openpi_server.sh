#!/bin/bash

# Move to the OpenPI directory
cd ~/VLABench/third_party/openpi

# Activate the environment
source examples/vlabench/.venv/bin/activate

export PYTHONPATH=$PYTHONPATH:~/VLABench

# Run the OpenPI server
uv run scripts/serve_policy.py --env VLABENCH policy:checkpoint --policy.config=pi0_fast_vlabench_lora --policy.dir=checkpoints/pi0_fast_vlabench_lora/pi0_fast_lora_vlabench_primitive/29999