from VLABench.evaluation.evaluator import Evaluator
from VLABench.evaluation.model.policy.openvla import OpenVLA
from VLABench.evaluation.model.policy.base import RandomPolicy
from VLABench.tasks import *
from VLABench.robots import *


"""
Alternatively:

for OpenVLA:
python scripts/evaluate_policy.py --eval-track=track_1_in_distribution --n-episode=10 --policy=openvla --model_ckpt="~/VLABench/checkpoints/openvla-7b" --lora_ckpt="" --save-dir="~/VLABench/logs/visual_generalization" --visualization --metrics success_rate intention_score progress_score

for Pi0:
python scripts/evaluate_policy.py --eval-track=track_1_in_distribution --n-episode=10 --policy=openpi --host="127.0.1.1" --save-dir="~/VLABench/logs/visual_generalization" --port="8000" --visualization --metrics success_rate intention_score progress_score
"""

# (Table 2a) Evaluation of visual generalization and knowledge transfer
visual_generalization_tasks = [
    "add_condiment", "add_condiment_common_sense",
    "insert_flower", "insert_flower_common_sense",
    "select_book",
    "select_drink", "select_drink_common_sense",
    "select_toy", "select_toy_common_sense",
    "select_chemistry_tube", "select_chemistry_tube_common_sense",
    "select_fruit", "select_fruit_common_sense",
    "select_painting", 
]

# (Table 2b) Evaluation of language instruction generalization.
language_generalization_tasks = ["add_condiment_semantic", "insert_flower_semantic", "select_drink_semantic", "select_toy_semantic", "select_chemistry_tube_semantic"]

# (Table 2c) Evaluation of unseen but similar task generalization.
unseen_tasks = [ "select_poker", "select_mahjong", "select_billards", "select_ingredient", "friction_qa"]

# (Table 2d) Evaluation of composite tasks.
composite_tasks = ["find_unseen_object", "texas_holdem", "cluster_toy", "hammer_nail_and_hang_picture", "get_coffee_with_milk"]

# demo_tasks = ["select_painting"]
unseen = True
save_dir = "~/VLABench/logs/visual_generalization"

model_ckpt = "~/VLABench/checkpoints/openvla-7b"
# lora_ckpt = "/home/adelli/VLABench/checkpoints/openvla-7b"


import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

evaluator = Evaluator(
    tasks=visual_generalization_tasks,
    n_episodes=10,
    max_substeps=10,
    save_dir=save_dir,
    visualization=True, # Save videos
    metrics=["success_rate", "intention_score", "progress_score"],
)

# OpenVLA
policy = OpenVLA(
    model_ckpt=model_ckpt,
    # lora_ckpt=lora_ckpt,
    norm_config_file=os.path.join(os.getenv("VLABENCH_ROOT"), "configs/model/openvla_config.json"),
    device="cuda:1",
)
result = evaluator.evaluate(policy)