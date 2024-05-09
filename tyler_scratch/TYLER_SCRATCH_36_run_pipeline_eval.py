# %%
import subprocess
import pathlib
import random
from tqdm import tqdm

# %%
EXPERIMENT_NAME = "2024-04-18_eval"
experiment_folder = pathlib.Path(f"/juno/u/tylerlum/github_repos/nerf_grasping/experiments/{EXPERIMENT_NAME}")
nerfdata_folder = experiment_folder / "nerfdata"
assert experiment_folder.exists(), experiment_folder

# %%
object_nerfdata_folders = sorted(list(nerfdata_folder.iterdir()))
random.Random(0).shuffle(object_nerfdata_folders)

# %%
for object_nerfdata_folder in tqdm(object_nerfdata_folders):
    command = f"python rough_hardware_deployment_code.py --experiment-name {EXPERIMENT_NAME} --init-grasp-config-dict-path /juno/u/tylerlum/github_repos/DexGraspNet/data/2024-04-09_rotated_grasps_aggregated_augmented_pose_HALTON_50/aggregated_evaled_grasp_config_dicts_train/aggregated_evaled_grasp_config_dict_train.npy --classifier-config-path /juno/u/tylerlum/github_repos/nerf_grasping/Train_DexGraspNet_NeRF_Grasp_Metric_workspaces/3000rotated_augmented_pose_HALTON_50_cnn-3d-xyz_l2_all_2024-04-17_00-53-42-594438/config.yaml --object_name {object_nerfdata_folder.stem}"
    print(f"Running command: {command}")
    subprocess.run(command, shell=True, check=True)
# %%
