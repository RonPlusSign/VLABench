#bin/bash

conda activate vlabench

eval_tracks=("track_1_in_distribution" "track_2_cross_category" "track_3_common_sense" "track_4_semantic_instruction" "track_6_unseen_texture")

EPISODE=50
POLICY="openvla"
MODEL_CKPT="checkpoints/openvla-7b"
# LORA_CKPT=""
SAVE_DIR="logs"
METRICS="success_rate intention_score progress_score"

for eval_track in "${eval_tracks[@]}"; do
    echo -e "\n \033[0;32m-------------------- Evaluating on eval track: $eval_track ------------------- \033[0m\n"
    mkdir -p $SAVE_DIR/$eval_track
    touch $SAVE_DIR/$eval_track/execution_log.txt

    python scripts/evaluate_policy.py \
        --eval-track $eval_track \
        --n-episode $EPISODE \
        --policy $POLICY \
        --model_ckpt $MODEL_CKPT \
        --lora_ckpt "" \
        --visualization \
        --save-dir $SAVE_DIR \
        --metrics $METRICS \
     2>&1 | tee $SAVE_DIR/$eval_track/execution_log.txt
done
