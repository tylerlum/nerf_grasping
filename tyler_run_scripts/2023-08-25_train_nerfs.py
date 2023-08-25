import os
import subprocess
from tqdm import tqdm
import random

NERF_DATA_DIR = "2023-08-25_nerfdata"
OUTPUT_NERF_CHECKPOINTS_DIR = "2023-08-25_nerfcheckpoints"
MAX_NUM_ITERATIONS = 200
assert os.path.exists(NERF_DATA_DIR), f"{NERF_DATA_DIR} does not exist"

os.makedirs(OUTPUT_NERF_CHECKPOINTS_DIR, exist_ok=True)

object_code_and_scale_str_list = os.listdir(NERF_DATA_DIR)
RANDOMIZE_ORDER_SEED = 3
print(f"Randomizing order with seed {RANDOMIZE_ORDER_SEED}")
random.Random(RANDOMIZE_ORDER_SEED).shuffle(object_code_and_scale_str_list)

for object_code_and_scale_str in tqdm(object_code_and_scale_str_list):
    output_folder_to_be_created = os.path.join(
        OUTPUT_NERF_CHECKPOINTS_DIR, object_code_and_scale_str
    )
    if os.path.exists(output_folder_to_be_created):
        print(f"Found {output_folder_to_be_created}, skipping")
        continue

    command = f"ns-train nerfacto --data {NERF_DATA_DIR}/{object_code_and_scale_str} --max-num-iterations {MAX_NUM_ITERATIONS} --output-dir {OUTPUT_NERF_CHECKPOINTS_DIR} --vis wandb --pipeline.model.disable-scene-contraction True --pipeline.model.background-color black nerfstudio-data --auto-scale-poses False --scale-factor 1. --scene-scale 0.2 --center-method none --orientation-method none"
    print(f"Running command = {command}")
    subprocess.run(command, check=True, shell=True)
