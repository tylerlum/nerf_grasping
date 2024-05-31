# %%
from typing import Optional
from tqdm import tqdm
import open3d as o3d
import numpy as np
import plotly.graph_objects as go
from bps import bps
import pathlib
import networkx as nx
from scipy.spatial import KDTree

# %%
point_cloud_folder = pathlib.Path("/juno/u/tylerlum/github_repos/nerf_grasping/data/2024-05-06_rotated_stable_grasps_0/pointclouds_250imgs_400iters_5k/")
assert point_cloud_folder.exists(), f"Path {point_cloud_folder} does not exist"

# %%
all_point_cloud_paths = sorted(list(point_cloud_folder.rglob("*.ply")))
print(f"Found {len(all_point_cloud_paths)} data paths")

# %%
point_clouds = []
for point_cloud_path in tqdm(all_point_cloud_paths, desc="Loading point clouds"):
    point_cloud = o3d.io.read_point_cloud(str(point_cloud_path))
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
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
from scipy.sparse.csgraph import connected_components


def construct_graph(points, distance_threshold=0.01):
    kdtree = KDTree(points)
    rows, cols = [], []

    for i, point in enumerate(points):
        neighbors = kdtree.query_ball_point(point, distance_threshold)
        for neighbor in neighbors:
            if neighbor != i:
                rows.append(i)
                cols.append(neighbor)

    data = np.ones(len(rows), dtype=np.int8)
    adjacency_matrix = csr_matrix(
        (data, (rows, cols)), shape=(len(points), len(points))
    )
    return adjacency_matrix


def get_largest_connected_component(adjacency_matrix):
    n_components, labels = connected_components(
        csgraph=adjacency_matrix, directed=False, return_labels=True
    )
    largest_cc_label = np.bincount(labels).argmax()
    largest_cc_indices = np.where(labels == largest_cc_label)[0]
    return largest_cc_indices


def process_point_cloud(points, distance_threshold=0.01):
    adjacency_matrix = construct_graph(points, distance_threshold)
    largest_cc_indices = get_largest_connected_component(adjacency_matrix)
    return points[largest_cc_indices]


all_inlier_points = []
for points in tqdm(all_points, desc="Constructing graphs"):
    inlier_points = process_point_cloud(points)
    all_inlier_points.append(inlier_points)

# %%
n_point_clouds = len(all_inlier_points)
print(f"n_point_clouds: {n_point_clouds}")

MIN_N_PTS = 3000
good_idxs = []
for i, inlier_points in enumerate(all_inlier_points):
    if inlier_points.shape[0] >= MIN_N_PTS:
        good_idxs.append(i)
print(f"For MIN_N_PTS {MIN_N_PTS}, good_idxs: {good_idxs} (len {len(good_idxs)}")

# %%
filtered_inlier_points = np.stack([all_inlier_points[i][:MIN_N_PTS] for i in good_idxs], axis=0)
assert filtered_inlier_points.shape == (len(good_idxs), MIN_N_PTS, 3), f"Expected shape ({len(good_idxs)}, {MIN_N_PTS}, 3), got {filtered_inlier_points.shape}"
n_point_clouds = filtered_inlier_points.shape[0]

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
    filtered_inlier_points,
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
def plot_bps_and_pc(
    basis_points: Optional[np.ndarray] = None,
    x_bps: Optional[np.ndarray] = None,
    point_cloud_points: Optional[np.ndarray] = None,
    point_cloud_points2: Optional[np.ndarray] = None,
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
    if point_cloud_points2 is not None:
        N2 = n_points2 = point_cloud_points2.shape[0]
        assert point_cloud_points2.shape == (
            N2,
            3,
        ), f"Expected shape ({N2}, 3), got {point_cloud_points2.shape}"
        fig.add_trace(
            go.Scatter3d(
                x=point_cloud_points2[:, 0],
                y=point_cloud_points2[:, 1],
                z=point_cloud_points2[:, 2],
                mode="markers",
                marker=dict(
                    size=5, color=point_cloud_points2[:, 1], colorscale="Viridis"
                ),
                name="Point cloud 2",
            )
        )
    fig.update_layout(title=str(title))
    fig.show()



# %%
POINT_CLOUD_IDX = 0
while True:
    print(f"POINT_CLOUD_IDX: {POINT_CLOUD_IDX}, has {filtered_inlier_points[POINT_CLOUD_IDX].shape[0]} points, had {all_points[POINT_CLOUD_IDX].shape[0]} points, path: {all_point_cloud_paths[POINT_CLOUD_IDX].parents[0].name}")
    plot_bps_and_pc(
        basis_points=basis_points,
        x_bps=x_bps[POINT_CLOUD_IDX, :],
        point_cloud_points=filtered_inlier_points[POINT_CLOUD_IDX],
        title=all_point_cloud_paths[POINT_CLOUD_IDX].parents[0].name,
    )
    user_input = input("Enter point cloud index (q to quit, b to breakpoint): ")
    if user_input == "q":
        break
    elif user_input == "b":
        breakpoint()
    elif user_input[0] == "i":
        if user_input[1:].isdigit():
            POINT_CLOUD_IDX = int(user_input[1:])
        else:
            print(f"Invalid input {user_input}")
    else:
        print(f"Invalid input {user_input}")


# %%
