# %%
import pathlib
import trimesh
import nerf_grasping
import numpy as np
from typing import List, Dict, Any
from nerf_grasping.optimizer_utils import AllegroHandConfig
import plotly.graph_objects as go
from nerf_grasping.dataset.DexGraspNet_NeRF_Grasps_utils import (
    plot_mesh,
    parse_object_code_and_scale,
)


# %%
dexgraspnet_data_root: pathlib.Path = (
    pathlib.Path(nerf_grasping.get_repo_root()) / "data"
)
dexgraspnet_meshdata_root: pathlib.Path = dexgraspnet_data_root / "meshdata_trial"
evaled_grasp_config_dicts_path: pathlib.Path = (
    dexgraspnet_data_root / "2023-09-06_evaled_grasp_config_dicts"
)
object_code_and_scale_str = "mug_0_1000"

object_code, object_scale = parse_object_code_and_scale(object_code_and_scale_str)

evaled_grasp_config_dict_filepath = (
    evaled_grasp_config_dicts_path / f"{object_code_and_scale_str}.npy"
)

evaled_grasp_config_dict: Dict[str, Any] = np.load(
    evaled_grasp_config_dict_filepath, allow_pickle=True
).item()
batch_size = evaled_grasp_config_dict["trans"].shape[0]
print(f"batch_size = {batch_size}")

# %%
mesh_path = dexgraspnet_meshdata_root / object_code / "coacd" / "decomposed.obj"
mesh = trimesh.load(mesh_path, force="mesh")
mesh.apply_transform(trimesh.transformations.scale_matrix(object_scale))

# %%
hand_config = AllegroHandConfig.from_hand_config_dict(
    evaled_grasp_config_dict,
)
fingertip_positions = (
    hand_config.get_fingertip_transforms().translation().detach().cpu().numpy()
)

finger_names = ["index", "middle", "ring", "thumb"]
n_fingers = len(finger_names)
assert fingertip_positions.shape == (
    batch_size,
    n_fingers,
    3,
)
successes = evaled_grasp_config_dict["passed_eval"]
assert successes.shape == (batch_size,)

# %%
successful_fingertip_positions = fingertip_positions[successes]
failed_fingertip_positions = fingertip_positions[~successes]


# %%
def filter(fingertip_positions: np.ndarray, max_dist: float) -> np.ndarray:
    max_fingertip_distance = np.linalg.norm(fingertip_positions, axis=-1).max(axis=-1)
    in_range_indices = max_fingertip_distance < max_dist
    filtered_fingertip_positions = fingertip_positions[in_range_indices]
    return filtered_fingertip_positions


min_bounds, max_bounds = mesh.bounds
x_min, y_min, z_min = min_bounds
x_max, y_max, z_max = max_bounds
max_dist = np.linalg.norm(max_bounds - min_bounds)
filtered_successful_fingertip_positions = filter(
    successful_fingertip_positions, max_dist
)
filtered_failed_fingertip_positions = filter(failed_fingertip_positions, max_dist)

# %%
successful_fig = plot_mesh(mesh=mesh)
for finger_idx, finger_name in enumerate(finger_names):
    successful_fig.add_trace(
        go.Scatter3d(
            x=filtered_successful_fingertip_positions[:, finger_idx, 0],
            y=filtered_successful_fingertip_positions[:, finger_idx, 1],
            z=filtered_successful_fingertip_positions[:, finger_idx, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=finger_idx,
                colorscale="Viridis",
                opacity=0.8,
            ),
            name=finger_name,
        )
    )


successful_fig.show()

# %%
failed_fig = plot_mesh(mesh=mesh)
for finger_idx, finger_name in enumerate(finger_names):
    failed_fig.add_trace(
        go.Scatter3d(
            x=filtered_failed_fingertip_positions[:, finger_idx, 0],
            y=filtered_failed_fingertip_positions[:, finger_idx, 1],
            z=filtered_failed_fingertip_positions[:, finger_idx, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=finger_idx,
                colorscale="Viridis",
                opacity=0.8,
            ),
            name=finger_name,
        )
    )


failed_fig.show()
