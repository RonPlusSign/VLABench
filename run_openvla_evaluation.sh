conda activate vlabench

python scripts/evaluate_policy.py --eval-track=track_1_in_distribution --n-episode=10 --policy=openvla \
    --model_ckpt="~/VLABench/checkpoints/openvla-7b" \
    --lora_ckpt="" \
    --save-dir="~/VLABench/logs/track_1_in_distribution" \
    --visualization --metrics success_rate intention_score progress_score \
   2>&1 | tee logs/openvla_execution.log