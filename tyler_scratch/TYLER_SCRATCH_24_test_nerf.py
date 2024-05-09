# %%
import trimesh
from nerf_grasping.grasp_utils import (
    get_ray_samples,
    get_ray_origins_finger_frame,
    get_nerf_configs,
    load_nerf_pipeline,
)
from nerf_grasping.dataset.DexGraspNet_NeRF_Grasps_utils import (
    plot_mesh_and_query_points,
    plot_mesh_and_transforms,
    plot_mesh_and_high_density_points,
    get_ray_samples_in_mesh_region,
    get_ray_samples_in_region,
    parse_object_code_and_scale,
)
from nerf_grasping.baselines.nerf_to_mesh import nerf_to_mesh
import pathlib
import numpy as np
from localscope import localscope
from typing import Tuple

# # %%
# nerf_checkpoints_path = pathlib.Path("/home/tylerlum/2024-04-09_rotated_grasps_aggregated_augmented_pose_HALTON_50/nerfcheckpoints")
# assert nerf_checkpoints_path.exists()
# 
# # %%
# nerf_configs = get_nerf_configs(
#     nerf_checkpoints_path=str(nerf_checkpoints_path)
# )
# 
# # %%
# nerf_config = nerf_configs[0]
# print(f"nerf_config: {nerf_config}")

# %%
nerf_config = pathlib.Path("/juno/u/tylerlum/github_repos/nerf_grasping/data/albert_real_world_data/nerfcheckpoints/milk-carton_0_9999/nerfacto/2024-03-13_224126/config.yml")
assert nerf_config.exists()
print(f"nerf_config: {nerf_config}")

#%%
object_code_and_scale_str = nerf_config.parent.parent.parent.name
print(f"object_code_and_scale_str: {object_code_and_scale_str}")

# %%
object_code, object_scale = parse_object_code_and_scale(object_code_and_scale_str)

# %%
nerf_pipeline = load_nerf_pipeline(nerf_config)

# %%
lb = -0.25 * np.ones(3)
ub = 0.25 * np.ones(3)
mesh = nerf_to_mesh(
    nerf_pipeline.model.field,
    level=15,
    npts=31,
    lb=lb,
    ub=ub,
    scale=1.0,
    min_len=100,
    # save_path=obj_path,
)

# %%
mesh_centroid = mesh.centroid
print(f"mesh_centroid: {mesh_centroid}")

# %%
@localscope.mfc
def get_nerf_densities_and_query_points_in_mesh_region(
    mesh: trimesh.Trimesh,
    nerf_pipeline,
    num_pts_x: int,
    num_pts_y: int,
    num_pts_z: int,
):
    ray_samples_in_region = get_ray_samples_in_mesh_region(
        mesh=mesh,
        num_pts_x=num_pts_x,
        num_pts_y=num_pts_y,
        num_pts_z=num_pts_z,
    )
    query_points_in_region = np.copy(
        ray_samples_in_region.frustums.get_positions()
        .cpu()
        .numpy()
        .reshape(
            num_pts_x,
            num_pts_y,
            num_pts_z,
            3,
        )
    )
    nerf_densities_in_region = (
        nerf_pipeline.model.field.get_density(ray_samples_in_region.to("cuda"))[0]
        .detach()
        .cpu()
        .numpy()
        .reshape(
            num_pts_x,
            num_pts_y,
            num_pts_z,
        )
    )
    return query_points_in_region, nerf_densities_in_region


@localscope.mfc
def get_nerf_densities_and_query_points_in_region(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    nerf_pipeline,
    num_pts_x: int,
    num_pts_y: int,
    num_pts_z: int,
) -> Tuple[np.ndarray, np.ndarray]:
    ray_samples_in_region = get_ray_samples_in_region(
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
    query_points_in_region = np.copy(
        ray_samples_in_region.frustums.get_positions()
        .cpu()
        .numpy()
        .reshape(
            num_pts_x,
            num_pts_y,
            num_pts_z,
            3,
        )
    )
    nerf_densities_in_region = (
        nerf_pipeline.model.field.get_density(ray_samples_in_region.to("cuda"))[0]
        .detach()
        .cpu()
        .numpy()
        .reshape(
            num_pts_x,
            num_pts_y,
            num_pts_z,
        )
    )
    return query_points_in_region, nerf_densities_in_region


# query_points_in_region, nerf_densities_in_region = get_nerf_densities_and_query_points_in_mesh_region(
#     mesh=mesh,
#     nerf_pipeline=nerf_pipeline,
#     num_pts_x=100,
#     num_pts_y=100,
#     num_pts_z=100,
# )

query_points_in_region, nerf_densities_in_region = get_nerf_densities_and_query_points_in_region(
    x_min=-0.25,
    x_max=0.25,
    # y_min=-0.25,
    y_min=0,
    y_max=0.25,
    z_min=-0.25,
    z_max=0.25,
    nerf_pipeline=nerf_pipeline,
    num_pts_x=100,
    num_pts_y=100,
    num_pts_z=100,
)

print(f"query_points_in_region.shape: {query_points_in_region.shape}")
print(f"nerf_densities_in_region.shape: {nerf_densities_in_region.shape}")

# %%
nerf_densities_in_mesh_region_repeated = np.repeat(
    nerf_densities_in_region[..., np.newaxis],
    3,
    axis=-1,
)
points_keep = query_points_in_region[nerf_densities_in_region > 15]
print(f"points_keep: {points_keep.shape}")



# %%
query_points_in_region[nerf_densities_in_region > 15].shape

# %%
(nerf_densities_in_region > 15).shape, (nerf_densities_in_region > 15).sum()

# %%

# %%
points_to_keep = np.where(
    (nerf_densities_in_region > 15)[..., None].repeat(3, axis=-1),
    query_points_in_region,
    np.nan,
)

# %%
points_to_keep.shape

# %%

points_avg = np.nanmean(points_to_keep.reshape(-1, 3), axis=0)

# %%
points_avg

# %%
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(
    go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        colorscale="Viridis",
        flatshading=True,
        opacity=0.5,
    )
)
fig.add_trace(
    go.Scatter3d(
        x=[points_avg[0]],
        y=[points_avg[1]],
        z=[points_avg[2]],
        mode="markers",
        marker=dict(size=2, color="red"),
    )
)
fig.add_trace(
    go.Scatter3d(
        x=[mesh_centroid[0]],
        y=[mesh_centroid[1]],
        z=[mesh_centroid[2]],
        mode="markers",
        marker=dict(size=2, color="blue"),
    )
)
fig.show()


# %%
points_to_keep_plot = points_to_keep[~np.isnan(points_to_keep)].reshape(-1, 3)
fig = go.Figure()
fig.add_trace(
    go.Scatter3d(
        x=points_to_keep_plot[:, 0],
        y=points_to_keep_plot[:, 1],
        z=points_to_keep_plot[:, 2],
        mode="markers",
        marker=dict(size=2, color="red"),
    )
)
fig.show()

# %%
points_to_keep.shape

# %%
