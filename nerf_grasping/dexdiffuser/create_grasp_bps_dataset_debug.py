# %%
from tqdm import tqdm
import open3d as o3d
import numpy as np
import plotly.graph_objects as go
from bps import bps
import pathlib
import networkx as nx
from scipy.spatial import KDTree

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
def construct_graph(points, distance_threshold=0.01):
    graph = nx.Graph()
    kdtree = KDTree(points)

    for i, point in enumerate(points):
        neighbors = kdtree.query_ball_point(point, distance_threshold)
        for neighbor in neighbors:
            if neighbor != i:
                graph.add_edge(i, neighbor)

    return graph


def get_largest_connected_component(graph):
    largest_cc = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest_cc).copy()


all_inlier_points = []
for points in tqdm(all_points, desc="Constructing graphs"):
    graph = construct_graph(points)
    largest_cc_graph = get_largest_connected_component(graph)
    largest_cc_indices = list(largest_cc_graph.nodes)
    inlier_points = points[largest_cc_indices]
    all_inlier_points.append(inlier_points)

# %%
n_point_clouds = len(all_inlier_points)
print(f"n_point_clouds: {n_point_clouds}")

# %%
num_points_list = [len(inlier_points) for inlier_points in all_inlier_points]
import matplotlib.pyplot as plt

plt.hist(num_points_list, bins=50)
plt.xlabel("Number of points")
plt.ylabel("Frequency")
plt.title("Histogram of number of points in point clouds")
plt.show()

# %%
N_BASIS_PTS = 4096
BASIS_RADIUS = 0.3

# %%
N_PTS_PER_PC = 1000

# %%
all_inlier_points = np.stack(
    [
        inlier_points[np.random.choice(len(inlier_points), N_PTS_PER_PC, replace=False)]
        for inlier_points in all_inlier_points
    ],
    axis=0,
)

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
    all_inlier_points,
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
from typing import Optional


def plot_bps_and_pc(
    basis_points: Optional[np.ndarray] = None,
    x_bps: Optional[np.ndarray] = None,
    point_cloud_points: Optional[np.ndarray] = None,
    title: Optional[str] = None,
) -> None:
    if title is None:
        title = "Basis points and point cloud"

    fig = go.Figure()
    if basis_points is not None and x_bps is not None:
        B = n_bps = basis_points.shape[0]
        assert basis_points.shape == (
            B,
            3,
        ), f"Expected shape ({B}, 3), got {basis_points.shape}"
        assert x_bps.shape == (n_bps,), f"Expected shape ({n_bps},), got {x_bps.shape}"
        fig.add_trace(
            go.Scatter3d(
                x=basis_points[:, 0],
                y=basis_points[:, 1],
                z=basis_points[:, 2],
                mode="markers",
                marker=dict(
                    size=1,
                    color=x_bps,
                    colorscale="rainbow",
                    colorbar=dict(title="Basis points", orientation="h"),
                ),
                name="Basis points",
            )
        )
    if point_cloud_points is not None:
        N = n_points = point_cloud_points.shape[0]
        assert point_cloud_points.shape == (
            N,
            3,
        ), f"Expected shape ({N}, 3), got {point_cloud_points.shape}"
        fig.add_trace(
            go.Scatter3d(
                x=point_cloud_points[:, 0],
                y=point_cloud_points[:, 1],
                z=point_cloud_points[:, 2],
                mode="markers",
                marker=dict(
                    size=5, color=point_cloud_points[:, 1], colorscale="Viridis"
                ),
                name="Point cloud",
            )
        )
    fig.update_layout(title=str(title))
    fig.show()


POINT_CLOUD_IDX = 0
while True:
    plot_bps_and_pc(
        basis_points=basis_points,
        x_bps=x_bps[POINT_CLOUD_IDX, :],
        point_cloud_points=all_inlier_points[POINT_CLOUD_IDX],
        title=all_data_paths[POINT_CLOUD_IDX].parents[0].name,
    )
    user_input = input("Enter point cloud index (q to quit): ")
    if user_input == "q":
        break
    if user_input.isdigit():
        POINT_CLOUD_IDX = int(user_input)
    else:
        print(f"Invalid input {user_input}")
