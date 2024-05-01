# %%
import numpy as np
import pathlib

# %%
config_dicts_folder = pathlib.Path(
    "data/2024-04-09_rotated_grasps_aggregated_augmented_pose_HALTON_50/evaled_grasp_config_dicts"
)
assert config_dicts_folder.exists()

# %%
config_dict_paths = sorted(list(config_dicts_folder.glob("*.npy")))
assert len(config_dict_paths) > 0
print(f"len(config_dict_paths) = {len(config_dict_paths)}")

# %%
from tqdm import tqdm
all_xyz = []
for config_dict_path in tqdm(config_dict_paths):
    config_dict = np.load(config_dict_path, allow_pickle=True).item()
    object_state = config_dict["object_states_before_grasp"]
    assert object_state.shape[-1] == 13
    object_state = object_state.reshape(-1, 13)
    xyz = object_state[:, :3]
    quat_xyzw = object_state[:, 3:7]
    v = object_state[:, 7:10]
    w = object_state[:, 10:13]

    all_xyz.append(xyz)

# %%
all_xyz = np.concatenate(all_xyz, axis=0)
assert all_xyz.shape[1] == 3

# %%
import matplotlib.pyplot as plt
plt.hist(all_xyz[:, 0], bins=100)

# %%%
plt.hist(all_xyz[:, 1], bins=100)

# %%
plt.hist(all_xyz[:, 2], bins=100)

# %%
config_dict.keys()

# %%
object_state.shape

# %%
diff = xyz[:, np.newaxis, :] - xyz[np.newaxis, :, :]
# Compute squared differences, sum over the last dimension, and take the square root
dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

# %%
np.max(dist_matrix)
# %%
diff = quat_xyzw[:, np.newaxis, :] - quat_xyzw[np.newaxis, :, :]
# Compute squared differences, sum over the last dimension, and take the square root
dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

# %%
np.max(dist_matrix)

# %%
np.max(np.abs(v)), np.mean(np.abs(v))

# %%
np.max(np.abs(w)), np.mean(np.abs(w))

# %%
