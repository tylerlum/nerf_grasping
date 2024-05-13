# %%
from tqdm import tqdm
import open3d as o3d
import numpy as np
import plotly.graph_objects as go
from bps import bps
import pathlib

# %%
path_str = "/juno/u/tylerlum/github_repos/nerf_grasping/data/2024-05-06_rotated_stable_grasps_0/pointclouds_250imgs_400iters_5k/"
path_bigger_str = "/juno/u/tylerlum/github_repos/nerf_grasping/data/2024-05-06_rotated_stable_grasps_bigger_0/pointclouds_250imgs_400iters_5k/"
path_smaller_str = "/juno/u/tylerlum/github_repos/nerf_grasping/data/2024-05-06_rotated_stable_grasps_smaller_0/pointclouds_250imgs_400iters_5k/"

paths = [pathlib.Path(path_str.replace("_0", f"_{i}")) for i in range(7)]
path_biggers = [pathlib.Path(path_bigger_str.replace("_0", f"_{i}")) for i in range(7)]
path_smallers = [
    pathlib.Path(path_smaller_str.replace("_0", f"_{i}")) for i in range(7)
]

all_paths = paths + path_biggers + path_smallers

# %%
all_data_paths = []
for path in tqdm(all_paths, desc="Finding data paths"):
    if not path.exists():
        print(f"Path {path} does not exist")
        continue

    data_paths = sorted(list(path.rglob("*.ply")))
    all_data_paths.extend(data_paths)

all_data_paths = sorted(all_data_paths)
print(f"Found {len(all_data_paths)} data paths")

# %%
point_clouds = []
for data_path in tqdm(all_data_paths, desc="Loading point clouds"):
    point_cloud = o3d.io.read_point_cloud(str(data_path))
    point_clouds.append(point_cloud)

print(f"Found {len(point_clouds)} point clouds")

# %%
all_points = []
for point_cloud in tqdm(point_clouds, desc="Extracting points"):
    point_cloud, _ = point_cloud.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )
    point_cloud, _ = point_cloud.remove_radius_outlier(nb_points=16, radius=0.05)
    points = np.asarray(point_cloud.points)
    all_points.append(points)

# %%
min_n_pts = min([x.shape[0] for x in all_points])
print(f"Minimum number of points: {min_n_pts}")

# %%
all_points = np.stack([x[:min_n_pts] for x in all_points])

# %%
n_point_clouds, n_point_cloud_pts = all_points.shape[:2]
print(f"Shape of all_points: {all_points.shape}")

# %%
N_BASIS_PTS = 4096
BASIS_RADIUS = 0.3

# %%
basis_points = bps.generate_random_basis(
    n_points=N_BASIS_PTS, radius=BASIS_RADIUS, random_seed=13
) + np.array(
    [0.0, BASIS_RADIUS / 2, 0.0]
)  # Shift up to get less under the table
assert basis_points.shape == (
    N_BASIS_PTS,
    3,
), f"Expected shape ({N_BASIS_PTS}, 3), got {basis_points.shape}"

x_bps = bps.encode(
    all_points,
    bps_arrangement="custom",
    bps_cell_type="dists",
    custom_basis=basis_points,
    verbose=0,
)
assert x_bps.shape == (
    n_point_clouds,
    N_BASIS_PTS,
), f"Expected shape ({n_point_clouds}, {N_BASIS_PTS}), got {x_bps.shape}"

# %%
POINT_CLOUD_IDX = -1

title = all_data_paths[POINT_CLOUD_IDX].parents[0].name
fig = go.Figure()
fig.add_trace(
    go.Scatter3d(
        x=basis_points[:, 0],
        y=basis_points[:, 1],
        z=basis_points[:, 2],
        mode="markers",
        marker=dict(
            size=1,
            color=x_bps[POINT_CLOUD_IDX, :],
            colorscale="rainbow",
            colorbar=dict(title="Basis points", orientation="h"),
        ),
        name="Basis points",
    )
)
# points_to_plot = all_points[POINT_CLOUD_IDX, :]
points_to_plot = np.asarray(point_clouds[0].points)
fig.add_trace(
    go.Scatter3d(
        x=points_to_plot[:, 0],
        y=points_to_plot[:, 1],
        z=points_to_plot[:, 2],
        mode="markers",
        marker=dict(size=5, color=points_to_plot[:, 1], colorscale="Viridis"),
        name="Point cloud",
    )
)
fig.update_layout(title=str(title))
fig.show()
