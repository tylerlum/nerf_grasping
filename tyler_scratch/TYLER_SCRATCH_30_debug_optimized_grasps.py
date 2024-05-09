# %%
from nerf_grasping.optimizer_utils import (
    get_sorted_grasps_from_dict,
)
import numpy as np
import pathlib

import plotly.graph_objects as go

from nerf_grasping.dexgraspnet_utils.hand_model import HandModel
from nerf_grasping.dexgraspnet_utils.hand_model_type import (
    HandModelType,
)
from nerf_grasping.dexgraspnet_utils.pose_conversion import (
    hand_config_to_pose,
)
import trimesh

# %%
optimized_grasp_config_dict_path = pathlib.Path("/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-04-16_DEBUG_12/optimized_grasp_config_dicts/sem-Vase-3a275e00d69810c62600e861c93ad9cc_0_0846.npy")
mesh_Oy_path = pathlib.Path("/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-04-16_DEBUG_12/nerf_to_mesh_Oy/sem-Vase-3a275e00d69810c62600e861c93ad9cc_0_0846.obj")
assert optimized_grasp_config_dict_path.exists()
assert mesh_Oy_path.exists()
GRASP_IDX = 0

# %%
optimized_grasp_config_dict = np.load(optimized_grasp_config_dict_path, allow_pickle=True).item()
X_Oy_H_array, joint_angles_array, target_joint_angles_array = (
    get_sorted_grasps_from_dict(
        optimized_grasp_config_dict=optimized_grasp_config_dict,
        error_if_no_loss=True,
        check=False,
        print_best=False,
    )
)

# %%
mesh_Oy = trimesh.load_mesh(mesh_Oy_path)

# %%
device = "cuda"
hand_model_type = HandModelType.ALLEGRO_HAND
hand_model = HandModel(hand_model_type=hand_model_type, device=device)

# %%
# Compute pregrasp and target hand poses
trans_array = X_Oy_H_array[:, :3, 3]
rot_array = X_Oy_H_array[:, :3, :3]

pregrasp_hand_pose = hand_config_to_pose(trans_array, rot_array, joint_angles_array).to(device)
target_hand_pose = hand_config_to_pose(trans_array, rot_array, target_joint_angles_array).to(device)

# %%
# Get plotly data
hand_model.set_parameters(pregrasp_hand_pose)
pregrasp_plot_data = hand_model.get_plotly_data(i=GRASP_IDX, opacity=1.0)

hand_model.set_parameters(target_hand_pose)
target_plot_data = hand_model.get_plotly_data(i=GRASP_IDX, opacity=0.5)

# %%
fig = go.Figure()
fig.add_trace(
    go.Mesh3d(
        x=mesh_Oy.vertices[:, 0],
        y=mesh_Oy.vertices[:, 1],
        z=mesh_Oy.vertices[:, 2],
        i=mesh_Oy.faces[:, 0],
        j=mesh_Oy.faces[:, 1],
        k=mesh_Oy.faces[:, 2],
        color="lightpink",
        opacity=0.50,
    )
)
for x in pregrasp_plot_data:
    fig.add_trace(x)
for x in target_plot_data:
    fig.add_trace(x)
fig.show()


# %%
