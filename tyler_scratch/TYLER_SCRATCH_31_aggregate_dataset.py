# %%
import subprocess
import pathlib
from tqdm import tqdm
import random
import math

# %%
nerfcheckpoints_path_1 = pathlib.Path("/juno/u/tylerlum/github_repos/nerf_grasping/data/2024-04-13_rotated_grasps_bigger_aggregated_augmented_pose_HALTON_50/nerfcheckpoints/")
nerfcheckpoints_path_2 = pathlib.Path("/juno/u/tylerlum/github_repos/nerf_grasping/data/2024-04-13_rotated_grasps_big_aggregated_augmented_pose_HALTON_50/nerfcheckpoints/")
nerfcheckpoints_path_3 = pathlib.Path("/home/tylerlum/2024-04-09_rotated_grasps_aggregated_augmented_pose_HALTON_50/nerfcheckpoints/")

evaled_grasp_config_dicts_path_1 = pathlib.Path("/juno/u/tylerlum/github_repos/nerf_grasping/data/2024-04-13_rotated_grasps_bigger_aggregated_augmented_pose_HALTON_50/evaled_grasp_config_dicts/")
evaled_grasp_config_dicts_path_2 = pathlib.Path("/juno/u/tylerlum/github_repos/nerf_grasping/data/2024-04-13_rotated_grasps_big_aggregated_augmented_pose_HALTON_50/evaled_grasp_config_dicts/")
evaled_grasp_config_dicts_path_3 = pathlib.Path("/juno/u/tylerlum/github_repos/nerf_grasping/data/2024-04-09_rotated_grasps_aggregated_augmented_pose_HALTON_50/evaled_grasp_config_dicts/")

assert nerfcheckpoints_path_1.exists()
assert nerfcheckpoints_path_2.exists()
assert nerfcheckpoints_path_3.exists()

assert evaled_grasp_config_dicts_path_1.exists()
assert evaled_grasp_config_dicts_path_2.exists()
assert evaled_grasp_config_dicts_path_3.exists()

# %%
new_experiment_name = "2024-04-16_rotated_grasps_aggregated_augmented_pose_HALTON_50"
new_experiment_folder = pathlib.Path("/juno/u/tylerlum/github_repos/nerf_grasping/data/") / new_experiment_name
new_evaled_grasp_config_dicts_path = new_experiment_folder / "evaled_grasp_config_dicts"
new_nerfcheckpoints_path = new_experiment_folder / "nerfcheckpoints"

new_evaled_grasp_config_dicts_path.mkdir(parents=True, exist_ok=False)
new_nerfcheckpoints_path.mkdir(parents=True, exist_ok=False)

# %%
nerfcheckpoint_paths = sorted(list(nerfcheckpoints_path_1.iterdir()) + list(nerfcheckpoints_path_2.iterdir()) + list(nerfcheckpoints_path_3.iterdir()))
evaled_grasp_config_dict_paths = sorted(list(evaled_grasp_config_dicts_path_1.iterdir()) + list(evaled_grasp_config_dicts_path_2.iterdir()) + list(evaled_grasp_config_dicts_path_3.iterdir()))
assert len(nerfcheckpoint_paths) > 0
assert len(evaled_grasp_config_dict_paths) > 0
print(f"len(nerfcheckpoint_paths): {len(nerfcheckpoint_paths)}")
print(f"len(evaled_grasp_config_dict_paths): {len(evaled_grasp_config_dict_paths)}")

# %%
for path in tqdm(nerfcheckpoint_paths, desc="Linking nerfcheckpoints"):
    ln_command = f"ln -sr {path} {new_nerfcheckpoints_path}"
    subprocess.run(ln_command, shell=True, check=True)

# %%
for path in tqdm(evaled_grasp_config_dict_paths, desc="Linking evaled_grasp_config_dicts"):
    ln_command = f"ln -sr {path} {new_evaled_grasp_config_dicts_path}"
    subprocess.run(ln_command, shell=True, check=True)

# %%
object_codes = [
    path.stem.split("_0_")[0]
    for path in evaled_grasp_config_dict_paths
]
print(f"object_codes[:10]: {object_codes[:10]}")
print(f"len(object_codes): {len(object_codes)}")

# %%
unique_object_codes = list(set(object_codes))
print(f"len(unique_object_codes): {len(unique_object_codes)}")
print(f"unique_object_codes[:10]: {unique_object_codes[:10]}")


# %%
frac_train, frac_val, frac_test = 0.8, 0.1, 0.1

# %%
random.Random(1231).shuffle(unique_object_codes)

n_train, n_val = (
    math.ceil(frac_train * len(unique_object_codes)),
    int(frac_val * len(unique_object_codes)),
)
n_test = len(unique_object_codes) - n_train - n_val
print(f"n_train: {n_train}, n_val: {n_val}, n_test: {n_test}")
print()

train_object_codes = unique_object_codes[:n_train]
val_object_codes = unique_object_codes[n_train : n_train + n_val]
test_object_codes = unique_object_codes[n_train + n_val :]

assert len(set(train_object_codes).intersection(val_object_codes)) == 0
assert len(set(train_object_codes).intersection(test_object_codes)) == 0
assert len(set(val_object_codes).intersection(test_object_codes)) == 0

# %%
train_object_codes_set, val_object_codes_set, test_object_codes_set = set(train_object_codes), set(val_object_codes), set(test_object_codes)

# %%
train_evaled_paths, val_evaled_paths, test_evaled_paths = [], [], []
for evaled_path in evaled_grasp_config_dict_paths:
    object_code = evaled_path.stem.split("_0_")[0]
    if object_code in train_object_codes_set:
        train_evaled_paths.append(evaled_path)
    if object_code in val_object_codes_set:
        val_evaled_paths.append(evaled_path)
    if object_code in test_object_codes_set:
        test_evaled_paths.append(evaled_path)

# %%
total = len(train_evaled_paths) + len(val_evaled_paths) + len(test_evaled_paths)
assert total == len(evaled_grasp_config_dict_paths)
print(f"total: {total}")

# %%
train_folder = new_experiment_folder / "evaled_grasp_config_dicts_train"
val_folder = new_experiment_folder / "evaled_grasp_config_dicts_val"
test_folder = new_experiment_folder / "evaled_grasp_config_dicts_test"

train_folder.mkdir(parents=True, exist_ok=False)
val_folder.mkdir(parents=True, exist_ok=False)
test_folder.mkdir(parents=True, exist_ok=False)

# %%
for path in tqdm(train_evaled_paths, desc="Linking train evaled_grasp_config_dicts"):
    ln_command = f"ln -sr {path} {train_folder}"
    subprocess.run(ln_command, shell=True, check=True)

# %%
for path in tqdm(val_evaled_paths, desc="Linking val evaled_grasp_config_dicts"):
    ln_command = f"ln -sr {path} {val_folder}"
    subprocess.run(ln_command, shell=True, check=True)

# %%
for path in tqdm(test_evaled_paths, desc="Linking test evaled_grasp_config_dicts"):
    ln_command = f"ln -sr {path} {test_folder}"
    subprocess.run(ln_command, shell=True, check=True)

# %%
