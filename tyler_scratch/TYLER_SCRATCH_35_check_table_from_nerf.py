# %%
import torch
from nerf_grasping.optimizer_utils import (
    get_sorted_grasps_from_dict,
    AllegroGraspConfig,
    predict_in_collision_with_object,
)
import numpy as np
import pathlib

import plotly.graph_objects as go

from nerf_grasping.grasp_utils import (
    load_nerf_field,
    get_nerf_configs,
    get_ray_samples,
    get_ray_origins_finger_frame,
)
from nerf_grasping.dexgraspnet_utils.hand_model import HandModel
from nerf_grasping.dexgraspnet_utils.hand_model_type import (
    HandModelType,
)
from nerf_grasping.dexgraspnet_utils.pose_conversion import (
    hand_config_to_pose,
)
import trimesh

from nerf_grasping.nerf_utils import (
    get_ray_samples_in_region,
    get_density,
)

# %%
optimized_grasp_config_dict_path = pathlib.Path(
    "/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-04-16_DEBUG_17/optimized_grasp_config_dicts/sem-Vase-3a275e00d69810c62600e861c93ad9cc_0_0846.npy"
)
mesh_Oy_path = pathlib.Path(
    "/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-04-16_DEBUG_17/nerf_to_mesh_Oy/sem-Vase-3a275e00d69810c62600e861c93ad9cc_0_0846.obj"
)
nerfcheckpoint_path = pathlib.Path(
    "/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-04-16_DEBUG_17/nerfcheckpoints/sem-Vase-3a275e00d69810c62600e861c93ad9cc_0_0846"
)
true_mesh_Oy_path = pathlib.Path(
    "/juno/u/tylerlum/github_repos/DexGraspNet/data/rotated_meshdata/sem-Vase-3a275e00d69810c62600e861c93ad9cc/coacd/decomposed.obj"
)
scale = 0.0846

experiment_folder = pathlib.Path(
    "/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-04-16_DEBUG"
)
object_name = "sem-Vase-3a275e00d69810c62600e861c93ad9cc_0_0846"
assert optimized_grasp_config_dict_path.exists()
assert mesh_Oy_path.exists()
assert nerfcheckpoint_path.exists()
assert true_mesh_Oy_path.exists()
GRASP_IDX = 3

# %%
optimized_grasp_config_dict = np.load(
    optimized_grasp_config_dict_path, allow_pickle=True
).item()
X_Oy_H_array, joint_angles_array, target_joint_angles_array = (
    get_sorted_grasps_from_dict(
        optimized_grasp_config_dict=optimized_grasp_config_dict,
        error_if_no_loss=True,
        check=False,
        print_best=False,
    )
)
optimized_grasp_config = AllegroGraspConfig.from_grasp_config_dict(
    optimized_grasp_config_dict
)


# %%
mesh_Oy = trimesh.load_mesh(mesh_Oy_path)
true_mesh_Oy = trimesh.load_mesh(true_mesh_Oy_path)
true_mesh_Oy.apply_scale(scale)

# %%
mesh_Oy.bounds

# %%
true_mesh_Oy.bounds

# %%
nerf_config = get_nerf_configs(str(nerfcheckpoint_path))[-1]
nerf_field = load_nerf_field(nerf_config)

# %%
x_min, y_min, z_min = -0.25, -0.25, -0.25
x_max, y_max, z_max = 0.25, 0.25, 0.25
num_pts_x, num_pts_y, num_pts_z = 100, 100, 100
ray_samples = get_ray_samples_in_region(
    x_min=x_min,
    x_max=x_max,
    y_min=y_min,
    y_max=y_max,
    z_min=z_min,
    z_max=z_max,
    num_pts_x=num_pts_x,
    num_pts_y=num_pts_y,
    num_pts_z=num_pts_z,
)
query_points = np.copy(
    ray_samples.frustums.get_positions()
    .cpu()
    .numpy()
    .reshape(
        num_pts_x,
        num_pts_y,
        num_pts_z,
        3,
    )
)
nerf_densities = (
    nerf_field.get_density(ray_samples.to("cuda"))[0]
    .detach()
    .cpu()
    .numpy()
    .reshape(
        num_pts_x,
        num_pts_y,
        num_pts_z,
    )
)

# %%


# %%
query_points = query_points.reshape(-1, 3)
nerf_densities = nerf_densities.reshape(-1)

query_points_colors = np.copy(nerf_densities)

# %%
fig = go.Figure()
num_pts = query_points.shape[0]
assert query_points.shape == (num_pts, 3), f"{query_points.shape}"
assert query_points_colors.shape == (num_pts,), f"{query_points_colors.shape}"

# Filter
query_point_plot = go.Scatter3d(
    x=query_points[query_points_colors > 15, 0],
    y=query_points[query_points_colors > 15, 1],
    z=query_points[query_points_colors > 15, 2],
    mode="markers",
    marker=dict(
        size=5,
        color=query_points_colors[query_points_colors > 15],
        colorscale="viridis",
        colorbar=dict(title="Density Scale"),
    ),
    name="Query Point Densities",
)
fig.add_trace(query_point_plot)

fig.add_trace(go.Mesh3d(x=mesh_Oy.vertices[:, 0], y=mesh_Oy.vertices[:, 1], z=mesh_Oy.vertices[:, 2], i=mesh_Oy.faces[:, 0], j=mesh_Oy.faces[:, 1], k=mesh_Oy.faces[:, 2], opacity=0.5, color="blue", name="Mesh Oy"))
fig.add_trace(go.Mesh3d(x=true_mesh_Oy.vertices[:, 0], y=true_mesh_Oy.vertices[:, 1], z=true_mesh_Oy.vertices[:, 2], i=true_mesh_Oy.faces[:, 0], j=true_mesh_Oy.faces[:, 1], k=true_mesh_Oy.faces[:, 2], opacity=0.5, color="red", name="True Mesh Oy"))

fig.update_layout(
    legend_orientation="h",
)  # Avoid overlapping legend
fig.update_layout(scene_aspectmode="data")
fig.update_layout(title="N frame")
fig.show()

# %%
import trimesh as tm
def get_hacky_table_mesh(obj_mesh: tm.Trimesh) -> tm.Trimesh:
    bounds = obj_mesh.bounds
    assert bounds.shape == (2, 3)
    min_bounds, max_bounds = bounds[0], bounds[1]
    min_x, min_y, min_z = min_bounds
    max_x, max_y, max_z = max_bounds

    table_y_Oy = min_y
    table_pos_Oy = np.array([0, table_y_Oy, 0])
    table_normal_Oy = np.array([0, 1, 0])
    table_parallel_Oy = np.array([1, 0, 0])
    assert table_pos_Oy.shape == table_normal_Oy.shape == (3,)

    SCALE_FACTOR = 2
    W, H = max_x - min_x, max_z - min_z
    W, H = SCALE_FACTOR * W, SCALE_FACTOR * H

    table_parallel_2_Oy = np.cross(table_normal_Oy, table_parallel_Oy)
    corner1 = table_pos_Oy + W / 2 * table_parallel_Oy + H / 2 * table_parallel_2_Oy
    corner2 = table_pos_Oy + W / 2 * table_parallel_Oy - H / 2 * table_parallel_2_Oy
    corner3 = table_pos_Oy - W / 2 * table_parallel_Oy + H / 2 * table_parallel_2_Oy
    corner4 = table_pos_Oy - W / 2 * table_parallel_Oy - H / 2 * table_parallel_2_Oy

    x = np.array([corner1[0], corner2[0], corner3[0], corner4[0]])
    y = np.array([corner1[1], corner2[1], corner3[1], corner4[1]])
    z = np.array([corner1[2], corner2[2], corner3[2], corner4[2]])

    i = [0, 0, 1]
    j = [1, 2, 2]
    k = [2, 3, 3]

    table_mesh = tm.Trimesh(vertices=np.stack([x, y, z], axis=1), faces=np.stack([i, j, k], axis=1))
    return table_mesh



table_mesh_Oy = get_hacky_table_mesh(true_mesh_Oy)
table_vertices = table_mesh_Oy.vertices
fig.add_trace(
    go.Mesh3d(
        x=table_vertices[:, 0],
        y=table_vertices[:, 1],
        z=table_vertices[:, 2],
        i=table_mesh_Oy.faces[:, 0],
        j=table_mesh_Oy.faces[:, 1],
        k=table_mesh_Oy.faces[:, 2],
        color="green",
        opacity=0.5,
        name="table",
    )
)
fig.show()


# %%
device = "cuda"
hand_model_type = HandModelType.ALLEGRO_HAND
hand_model = HandModel(hand_model_type=hand_model_type, device=device, n_surface_points=1000)

# %%
translation = optimized_grasp_config.wrist_pose.translation().detach().cpu().numpy()
rotation = optimized_grasp_config.wrist_pose.rotation().matrix().detach().cpu().numpy()
joint_angles = optimized_grasp_config.joint_angles.detach().cpu().numpy()
print(f"translation.shape: {translation.shape}")
print(f"rotation.shape: {rotation.shape}")
print(f"joint_angles.shape: {joint_angles.shape}")

# %%
hand_pose = hand_config_to_pose(translation, rotation, joint_angles).to(device)
hand_model.set_parameters(hand_pose)

# %%
surface_points = hand_model.get_surface_points()
print(f"surface_points.shape: {surface_points.shape}")

# %%
densities = get_density(
    field=nerf_field,
    positions=surface_points,
)[0].squeeze(dim=-1).detach().cpu().numpy()
print(f"densities.shape: {densities.shape}")

# %%
max_densities = densities.max(axis=-1)
print(f"max_densities.shape {max_densities.shape}")
print(f"max_densities: {max_densities}")

# %%
predict_penetrations = max_densities > 8.5

# %%
predict_no_penetration_idxs = [i for i in range(predict_penetrations.shape[0]) if not predict_penetrations[i]]
print(f"predict_no_penetration_idxs: {predict_no_penetration_idxs}")

# %%
actual_no_penetration_idxs = [2, 4, 5, 6, 7, 10, 12, 13, 16, 18, 20, 21, 22, 23, 24, 25, 27, 28, 29]
print(f"actual_no_penetration_idxs: {actual_no_penetration_idxs}")

# %%
num_wrong = len(set(predict_no_penetration_idxs).symmetric_difference(set(actual_no_penetration_idxs)))
num_total = predict_penetrations.shape[0]
print(f"num_wrong / num_total: {num_wrong} / {num_total} = {num_wrong / num_total * 100:.2f}%")

# %%
predicted_in_collision = predict_in_collision_with_object(nerf_field=nerf_field, grasp_config=optimized_grasp_config.to("cuda"))

# %%
predicted_no_collision_idxs = [i for i in range(predicted_in_collision.shape[0]) if not predicted_in_collision[i]]

# %%
predicted_no_collision_idxs

# %%
num_diff = len(set(predicted_no_collision_idxs).symmetric_difference(set(predict_no_penetration_idxs)))
num_wrong = len(set(predicted_no_collision_idxs).symmetric_difference(set(actual_no_penetration_idxs)))
print(f"num_diff: {num_diff}")
print(f"num_wrong: {num_wrong}")

# %%
type(predicted_in_collision)

# %%
