# %%
from nerf_grasping.nerf_utils import (
    compute_centroid_from_nerf,
    get_ray_samples_in_region,
)
import plotly.graph_objects as go

from nerf_grasping import grasp_utils
import numpy as np
import pathlib

# %%
nerf_config = pathlib.Path("/juno/u/tylerlum/github_repos/nerf_grasping/2024-04-25_ALBERT_data/nerfcheckpoints/goblet_0_9999/nerfacto/2024-04-29_192400/config.yml")
assert nerf_config.exists(), f"{nerf_config} does not exist"

nerf_field = grasp_utils.load_nerf_field(nerf_config)
# %%
lb_N = -0.25 * np.ones(3)
lb_N[2] = 0
ub_N = 0.25 * np.ones(3)
nerf_centroid_N = compute_centroid_from_nerf(
    nerf_field,
    lb=lb_N,
    ub=ub_N,
    level=10,
    num_pts_x=100,
    num_pts_y=100,
    num_pts_z=100,
)

# %%
print(nerf_centroid_N)

# %%
x_min, y_min, z_min = -0.25, -0.25, 0
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
query_points = query_points.reshape(-1, 3)
nerf_densities = nerf_densities.reshape(-1)

query_points_colors = np.copy(nerf_densities)

# %%
fig = go.Figure()
num_pts = query_points.shape[0]
assert query_points.shape == (num_pts, 3), f"{query_points.shape}"
assert query_points_colors.shape == (num_pts,), f"{query_points_colors.shape}"

# Filter
query_points = query_points[query_points_colors > 10]
query_points_colors = query_points_colors[query_points_colors > 10]

query_point_plot = go.Scatter3d(
    x=query_points[:, 0],
    y=query_points[:, 1],
    z=query_points[:, 2],
    mode="markers",
    marker=dict(
        size=1,
        color=query_points_colors,
        colorscale="viridis",
        colorbar=dict(title="Density Scale"),
    ),
    name="Query Point Densities",
)
fig.add_trace(query_point_plot)
fig.add_trace(
    go.Scatter3d(
        x=[nerf_centroid_N[0]],
        y=[nerf_centroid_N[1]],
        z=[nerf_centroid_N[2]],
        mode="markers",
        marker=dict(size=10, color="blue"),
        name="nerf_centroid",
    )
)

fig.update_layout(
    legend_orientation="h",
)  # Avoid overlapping legend
fig.update_layout(scene_aspectmode="data")
fig.update_layout(title="N frame")
fig.show()



# %%
