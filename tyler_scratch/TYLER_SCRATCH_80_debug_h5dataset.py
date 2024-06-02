# %%
import h5py
import pathlib
import trimesh
import numpy as np
import plotly.graph_objects as go
import pypose as pp
import torch
from nerf_grasping import grasp_utils
from nerf_grasping.config.fingertip_config import EvenlySpacedFingertipConfig

# %%
path = pathlib.Path(
    "/juno/u/tylerlum/github_repos/nerf_grasping/2024-06-02_TYLER_TO_ALBERT_noisy_organized/grid_dataset/val_dataset.h5"
)
file = h5py.File(
    path,
    "r",
)

# %%
file.attrs["num_data_points"]

# %%
(file["/passed_eval"][()] > 0.9).nonzero()


# %%
# GRASP_IDX = 61055
GRASP_IDX = 31000
assert GRASP_IDX < file.attrs["num_data_points"]

# %%
file["/object_scale"][()][GRASP_IDX]

# %%
file["/passed_simulation"][()].shape

# %%
file["/passed_penetration_threshold"][()].shape

# %%
file["/passed_eval"][()].shape

# %%
file["/nerf_densities"].shape

# %%
file["/nerf_densities"][GRASP_IDX].shape

# %%
file["/nerf_densities_global_idx"].shape

# %%
object_idx = file["/nerf_densities_global_idx"][GRASP_IDX]
print(f"Object idx: {object_idx}")

# %%
file["/nerf_densities_global"].shape

# %%
file["/nerf_densities_global"][object_idx].shape


# %%
file.keys()

# %%
file["/object_code"][GRASP_IDX].decode("utf-8")

# %%
# Extract mesh info
object_code = file["/object_code"][GRASP_IDX].decode("utf-8")
object_scale = file["/object_scale"][GRASP_IDX]

object_path = (
    pathlib.Path("/juno/u/tylerlum/github_repos/DexGraspNet/data/rotated_meshdata_v2")
    / object_code
    / "coacd"
    / "decomposed.obj"
)
assert object_path.exists()
mesh_Oy = trimesh.load_mesh(object_path)
mesh_Oy.apply_scale(object_scale)

# %%
# Compute object transform
X_N_Oy = np.eye(4)
bounds = mesh_Oy.bounds
assert bounds.shape == (2, 3)
min_y = bounds[0, 1]
X_N_Oy[1, 3] = -min_y

X_Oy_N = np.linalg.inv(X_N_Oy)

# Transform mesh to nerf frame
mesh_N = mesh_Oy.copy()
mesh_N.apply_transform(X_N_Oy)

# %%
# Extract density info
N_FINGERS = 4
grasp_frame_transforms = file["/grasp_transforms"][GRASP_IDX]
assert grasp_frame_transforms.shape == (N_FINGERS, 4, 4)
nerf_densities = file["/nerf_densities"][GRASP_IDX]
assert len(nerf_densities.shape) == 4

T_Oy_Fi = pp.from_matrix(torch.from_numpy(grasp_frame_transforms), pp.SE3_type)
assert T_Oy_Fi.lshape == (N_FINGERS,)

T_N_Oy = pp.from_matrix(
    torch.from_numpy(X_N_Oy)
    .float()
    .unsqueeze(dim=0)
    .repeat_interleave(N_FINGERS, dim=0)
    .reshape(N_FINGERS, 4, 4),
    pp.SE3_type,
).to(T_Oy_Fi.device)

# Transform grasp_frame_transforms to nerf frame
T_N_Fi = T_N_Oy @ T_Oy_Fi

# Generate RaySamples.
fingertip_config = EvenlySpacedFingertipConfig()
delta = fingertip_config.grasp_depth_mm / 1000 / (fingertip_config.num_pts_z - 1)

nerf_alphas = 1 - np.exp(-delta * nerf_densities)

ray_origins_finger_frame = grasp_utils.get_ray_origins_finger_frame(fingertip_config)
ray_samples = grasp_utils.get_ray_samples(
    ray_origins_finger_frame,
    T_N_Fi,
    fingertip_config,
)

query_points_N = ray_samples.frustums.get_positions().reshape(
    N_FINGERS,
    fingertip_config.num_pts_x,
    fingertip_config.num_pts_y,
    fingertip_config.num_pts_z,
    3,
)
query_points_Oy = query_points_N + X_Oy_N[:3, 3].reshape(1, 1, 1, 1, 3)

flatten_query_points_Oy = query_points_Oy.reshape(N_FINGERS, -1, 3)
flatten_nerf_alphas = nerf_alphas.reshape(N_FINGERS, -1,)

# %%
# Extract density global info
nerf_densities_global_idx = file["/nerf_densities_global_idx"][GRASP_IDX]
nerf_densities_global = file["/nerf_densities_global"][nerf_densities_global_idx]
assert len(nerf_densities_global.shape) == 3

nerf_alphas_global = 1 - np.exp(-delta * nerf_densities_global)

xx, yy, zz = np.meshgrid(
    np.linspace(-0.2, 0.2, nerf_densities_global.shape[0]),
    np.linspace(-0.2, 0.2, nerf_densities_global.shape[1]),
    np.linspace(-0.2, 0.2, nerf_densities_global.shape[2]),
    indexing="ij",
)
query_points_global_Oy = np.stack([xx, yy, zz], axis=-1)
assert query_points_global_Oy.shape == nerf_densities_global.shape + (3,)

flatten_query_points_global_Oy = query_points_global_Oy.reshape(-1, 3)
flatten_nerf_alphas_global = nerf_alphas_global.flatten()

# %%
# Extract labels
passed_eval = file["/passed_eval"][GRASP_IDX]
passed_simulation = file["/passed_simulation"][GRASP_IDX]
passed_penetration_threshold = file["/passed_penetration_threshold"][GRASP_IDX]

# %%
assert flatten_query_points_Oy.shape == (N_FINGERS, fingertip_config.num_pts_x * fingertip_config.num_pts_y * fingertip_config.num_pts_z, 3), f"{flatten_query_points_Oy.shape}"
assert flatten_nerf_alphas.shape == (N_FINGERS, fingertip_config.num_pts_x * fingertip_config.num_pts_y * fingertip_config.num_pts_z,), f"{flatten_nerf_alphas.shape}"
# Plot
fig = go.Figure()
fig.add_trace(
    go.Mesh3d(
        x=mesh_Oy.vertices[:, 0],
        y=mesh_Oy.vertices[:, 1],
        z=mesh_Oy.vertices[:, 2],
        i=mesh_Oy.faces[:, 0],
        j=mesh_Oy.faces[:, 1],
        k=mesh_Oy.faces[:, 2],
        opacity=0.5,
    )
)
for i in range(N_FINGERS):
    fig.add_trace(
        go.Scatter3d(
            x=flatten_query_points_Oy[i, :, 0],
            y=flatten_query_points_Oy[i, :, 1],
            z=flatten_query_points_Oy[i, :, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=flatten_nerf_alphas[i],
                # colorbar=dict(title="Density"),
            ),
            name=f"Query Points Finger {i}",
        )
    )
density_threshold = 0.01
fig.add_trace(
    go.Scatter3d(
        x=flatten_query_points_global_Oy[:, 0][
            flatten_nerf_alphas_global > density_threshold
        ],
        y=flatten_query_points_global_Oy[:, 1][
            flatten_nerf_alphas_global > density_threshold
        ],
        z=flatten_query_points_global_Oy[:, 2][
            flatten_nerf_alphas_global > density_threshold
        ],
        mode="markers",
        marker=dict(
            size=2,
            color=flatten_nerf_alphas_global[
                flatten_nerf_alphas_global > density_threshold
            ],
            # colorbar=dict(title="Density"),
        ),
        name="Query Points Global",
    )
)
fig.update_layout(
    scene_aspectmode="data",
    # title_text=f"{object_code}_{object_scale}_{GRASP_IDX}_\n{passed_eval}_{passed_simulation}_{passed_penetration_threshold}",
    # 3 decimal places
    title_text=f"Eval-{passed_eval:.3f}_{passed_simulation:.3f}_{passed_penetration_threshold:.3f}",
)
print(f"{object_scale}_{GRASP_IDX}_\n{passed_eval}_{passed_simulation}_{passed_penetration_threshold}")

# Hand
grasp_config = file['grasp_configs'][GRASP_IDX]
assert grasp_config.shape == (4, 27)

xyz = grasp_config[0, :3]
quat_xyzw = grasp_config[0, 3:7]
joint_angles = grasp_config[0, 7:23]
grasp_quat_orientations = grasp_config[:, 23:]

import pypose as pp

rot = pp.SO3(quat_xyzw)

grasp_orientations = pp.SO3(grasp_quat_orientations)

wrist_trans_array = xyz[None]
wrist_rot_array = rot.matrix().detach().cpu().numpy()[None]
joint_angles_array = joint_angles[None]

# Put into transforms X_Oy_H_array
B = 1
X_Oy_H_array = np.repeat(np.eye(4)[None, ...], B, axis=0)
assert X_Oy_H_array.shape == (B, 4, 4)
X_Oy_H_array[:, :3, :3] = wrist_rot_array
X_Oy_H_array[:, :3, 3] = wrist_trans_array

X_N_H_array = np.repeat(np.eye(4)[None, ...], B, axis=0)
for i in range(B):
    X_N_H_array[i] = X_N_Oy @ X_Oy_H_array[i]

from nerf_grasping.dexgraspnet_utils.hand_model import HandModel, HandModelType
from nerf_grasping.dexgraspnet_utils.pose_conversion import hand_config_to_pose
device = "cuda"
hand_model_type = HandModelType.ALLEGRO_HAND
hand_model = HandModel(hand_model_type=hand_model_type, device=device)

# Compute pregrasp and target hand poses
# trans_array = X_N_H_array[:, :3, 3]
# rot_array = X_N_H_array[:, :3, :3]
trans_array = X_Oy_H_array[:, :3, 3]
rot_array = X_Oy_H_array[:, :3, :3]

pregrasp_hand_pose = hand_config_to_pose(trans_array, rot_array, joint_angles_array).to(device)

# Get plotly data
hand_model.set_parameters(pregrasp_hand_pose)
pregrasp_plot_data = hand_model.get_plotly_data(i=0, opacity=1.0)

for x in pregrasp_plot_data:
    fig.add_trace(x)

fig.show()

# %%
from nerf_grasping.learned_metric.DexGraspNet_batch_data import BatchDataInput
batch_data_input = BatchDataInput(
    nerf_densities=torch.from_numpy(nerf_densities[None]),
    grasp_transforms=T_Oy_Fi[None],
    fingertip_config=fingertip_config,
    grasp_configs=torch.from_numpy(grasp_config[None]),
    nerf_densities_global=torch.from_numpy(nerf_densities_global[None]),
    object_y_wrt_table=None,
)

# %%

nerf_densities.shape, T_Oy_Fi.shape, grasp_config.shape, nerf_densities_global.shape
# %%
nerf_alphas_with_coords = batch_data_input.nerf_alphas_with_coords.squeeze(dim=0).detach().cpu().numpy()

# %%
nerf_alphas_with_coords.shape

# %%
N_FINGERS = 4
assert nerf_alphas_with_coords.shape == (N_FINGERS, 4, fingertip_config.num_pts_x, fingertip_config.num_pts_y, fingertip_config.num_pts_z), f"{nerf_alphas_with_coords.shape}"
nerf_alphas = nerf_alphas_with_coords[:, 0]
nerf_positions = nerf_alphas_with_coords[:, 1:]
assert nerf_alphas.shape == (N_FINGERS, fingertip_config.num_pts_x, fingertip_config.num_pts_y, fingertip_config.num_pts_z), f"{nerf_alphas.shape}"
assert nerf_positions.shape == (N_FINGERS, 3, fingertip_config.num_pts_x, fingertip_config.num_pts_y, fingertip_config.num_pts_z), f"{nerf_positions.shape}"

nerf_alphas_flatten = nerf_alphas.reshape(N_FINGERS, -1,)
nerf_positions_flatten = nerf_positions.reshape(N_FINGERS, 3, -1)

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
        opacity=0.5,
    )
)
for i in range(N_FINGERS):
    fig.add_trace(
        go.Scatter3d(
            x=nerf_positions_flatten[i, 0, :],
            y=nerf_positions_flatten[i, 1, :],
            z=nerf_positions_flatten[i, 2, :],
            mode="markers",
            marker=dict(
                size=2,
                color=nerf_alphas_flatten[i],
                # colorbar=dict(title="Density"),
            ),
            name=f"Query Points Finger {i}",
        )
    )
# density_threshold = 0.01
# fig.add_trace(
#     go.Scatter3d(
#         x=flatten_query_points_global_Oy[:, 0][
#             flatten_nerf_alphas_global > density_threshold
#         ],
#         y=flatten_query_points_global_Oy[:, 1][
#             flatten_nerf_alphas_global > density_threshold
#         ],
#         z=flatten_query_points_global_Oy[:, 2][
#             flatten_nerf_alphas_global > density_threshold
#         ],
#         mode="markers",
#         marker=dict(
#             size=2,
#             color=flatten_nerf_alphas_global[
#                 flatten_nerf_alphas_global > density_threshold
#             ],
#             # colorbar=dict(title="Density"),
#         ),
#         name="Query Points Global",
#     )
# )
fig.update_layout(
    scene_aspectmode="data",
    # title_text=f"{object_code}_{object_scale}_{GRASP_IDX}_\n{passed_eval}_{passed_simulation}_{passed_penetration_threshold}",
    # 3 decimal places
    title_text=f"Eval-{passed_eval:.3f}_{passed_simulation:.3f}_{passed_penetration_threshold:.3f}",
)
print(f"{object_scale}_{GRASP_IDX}_\n{passed_eval}_{passed_simulation}_{passed_penetration_threshold}")
fig.show()

# %%
