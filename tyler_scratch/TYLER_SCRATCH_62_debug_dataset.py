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
file = h5py.File(
    "/home/tylerlum/2024-05-09_rotated_stable_grasps_noisy_TUNED/grid_dataset/train_dataset.h5",
    "r",
)

# %%
file.attrs["num_data_points"]

# %%
GRASP_IDX = 15346
assert GRASP_IDX < file.attrs["num_data_points"]

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

flatten_query_points_Oy = query_points_Oy.reshape(-1, 3)
flatten_nerf_alphas = nerf_alphas.flatten()

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
            x=flatten_query_points_Oy[:, 0],
            y=flatten_query_points_Oy[:, 1],
            z=flatten_query_points_Oy[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=flatten_nerf_alphas,
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
fig.update_layout(scene_aspectmode="data", title_text=f"{object_code}_{object_scale}_{GRASP_IDX}_{passed_eval}_{passed_simulation}_{passed_penetration_threshold}")

fig.show()


# %%
file.keys()

# %%
