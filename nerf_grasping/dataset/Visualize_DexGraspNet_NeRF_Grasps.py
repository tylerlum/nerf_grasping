# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Visualize DexGraspNet NeRF Grasps
#
# ## Summary (Jul 23, 2023)
#
# The purpose of this script is to load a NeRF object model and labeled grasps on this object, and then visualize it

# %%
import math
import nerf_grasping
from localscope import localscope
from nerf import utils
import torch
import os
import trimesh
import numpy as np
from plotly import graph_objects as go
from dataclasses import dataclass
import matplotlib.pyplot as plt

# %%
# PARAMS
dexgraspnet_data_root = "/juno/u/tylerlum/github_repos/DexGraspNet/data"
dexgraspnet_meshdata_root = os.path.join(dexgraspnet_data_root, "meshdata")
dexgraspnet_dataset_root = os.path.join(dexgraspnet_data_root, "2023-07-01_dataset_DESIRED_DIST_TOWARDS_OBJECT_SURFACE_MULTIPLE_STEPS_v2")
mesh_path = os.path.join(
    dexgraspnet_meshdata_root, "ddg-gd_banana_poisson_001", "coacd", "decomposed.obj"
)
dataset_path = os.path.join(dexgraspnet_dataset_root, "ddg-gd_banana_poisson_001.npy")
nerf_checkpoint_folder = "2023-07-21_nerf_checkpoints"
nerf_model_workspace = "ddg-gd_banana_poisson_001_0_06"
nerf_size_scale = 0.06
nerf_bound = 2.0
nerf_scale = 1.0

# %%
grasp_dataset = np.load(dataset_path, allow_pickle=True)
grasp_dataset.shape

# %%
for data_dict in grasp_dataset:
    link_name_to_contact_candidates = data_dict["link_name_to_contact_candidates"]
    link_name_to_target_contact_candidates = data_dict["link_name_to_target_contact_candidates"]
    contact_candidates = np.concatenate([contact_candidate for _, contact_candidate in link_name_to_contact_candidates.items()], axis=0)
    target_contact_candidates = np.concatenate([target_contact_candidate for _, target_contact_candidate in link_name_to_target_contact_candidates.items()], axis=0)
    scale = data_dict["scale"]
    if not math.isclose(scale, nerf_size_scale, rel_tol=1e-3):
        continue
    print(f"contact_candidates.shape: {contact_candidates.shape}")
    print(f"target_contact_candidates.shape: {target_contact_candidates.shape}")
    break

# %%
mesh = trimesh.load(mesh_path, force="mesh")

# %%
mesh.centroid

# %%
mesh.apply_transform(trimesh.transformations.scale_matrix(nerf_size_scale))

# %%
mesh.centroid


# %%
@dataclass
class Bounds3D:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def max_bounds(self, other):
        assert isinstance(other, Bounds3D)
        return Bounds3D(
            x_min=min(self.x_min, other.x_min),
            x_max=max(self.x_max, other.x_max),
            y_min=min(self.y_min, other.y_min),
            y_max=max(self.y_max, other.y_max),
            z_min=min(self.z_min, other.z_min),
            z_max=max(self.z_max, other.z_max),
        )

    def extend(self, scale: float):
        return Bounds3D(
            x_min=self.x_min * scale,
            x_max=self.x_max * scale,
            y_min=self.y_min * scale,
            y_max=self.y_max * scale,
            z_min=self.z_min * scale,
            z_max=self.z_max * scale,
        )

    @property
    def x_range(self):
        return self.x_max - self.x_min

    @property
    def y_range(self):
        return self.y_max - self.y_min

    @property
    def z_range(self):
        return self.z_max - self.z_min


# %%
@localscope.mfc
def get_bounds(mesh: trimesh.Trimesh) -> Bounds3D:
    min_bounds, max_bounds = mesh.bounds
    return Bounds3D(
        x_min=min_bounds[0],
        x_max=max_bounds[0],
        y_min=min_bounds[1],
        y_max=max_bounds[1],
        z_min=min_bounds[2],
        z_max=max_bounds[2],
    )


@localscope.mfc
def get_scene_dict(mesh: trimesh.Trimesh):
    # bounds = get_bounds(mesh)
    # return dict(
    #     xaxis=dict(title="X", range=[bounds.x_min, bounds.x_max]),
    #     yaxis=dict(title="Y", range=[bounds.y_min, bounds.y_max]),
    #     zaxis=dict(title="Z", range=[bounds.z_min, bounds.z_max]),
    #     aspectratio=dict(x=bounds.x_range, y=bounds.y_range, z=bounds.z_range),
    #     aspectmode="manual",
    # )
    return dict(
        xaxis=dict(title="X"),
        yaxis=dict(title="Y"),
        zaxis=dict(title="Z"),
        aspectmode="data",
    )


# %%
@localscope.mfc
def plot_mesh(mesh, color="lightpink"):
    vertices = mesh.vertices
    faces = mesh.faces

    # Create the mesh3d trace
    mesh_plot = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=0.5,
        name="Mesh",
    )

    # Create the layout
    layout = go.Layout(
        scene=get_scene_dict(mesh),
        showlegend=True,
        title="Mesh",
    )

    # Create the figure
    fig = go.Figure(data=[mesh_plot], layout=layout)

    # Return the figure
    return fig


# %%
fig = plot_mesh(mesh)
fig.show()

# %%


@localscope.mfc(allowed=["nerf_checkpoint_folder"])
def load_nerf(workspace: str, bound: float, scale: float):
    root_dir = nerf_grasping.get_repo_root()

    parser = utils.get_config_parser()
    opt = parser.parse_args(
        [
            "--workspace",
            f"{root_dir}/{nerf_checkpoint_folder}/{workspace}",
            "--fp16",
            "--test",
            "--bound",
            f"{bound}",
            "--scale",
            f"{scale}",
            "--mode",
            "blender",
            f"{root_dir}/torch-ngp",
        ]
    )
    # Use options to determine proper network structure.
    if opt.ff:
        assert opt.fp16, "fully-fused mode must be used with fp16 mode"
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        assert opt.fp16, "tcnn mode must be used with fp16 mode"
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork

    # Create uninitialized network.
    model = NeRFNetwork(
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
    )

    # Create trainer with NeRF; use its constructor to load network weights from file.
    trainer = utils.Trainer(
        "ngp",
        vars(opt),
        model,
        workspace=opt.workspace,
        criterion=None,
        fp16=opt.fp16,
        metrics=[None],
        use_checkpoint="latest",
    )
    assert len(trainer.stats["checkpoints"]) != 0, "failed to load checkpoint"
    return trainer.model


# %%
nerf_model = load_nerf(
    workspace=nerf_model_workspace, bound=nerf_bound, scale=nerf_scale
)

# %%
nerf_model


# %%
@localscope.mfc
def get_nerf_densities(nerf_model, query_points: torch.Tensor):
    """
    Evaluates density of a batch of grasp points, shape [B, n_f, 3].
    query_points is torch.Tensor in nerf frame
    """
    B, n_f, _ = query_points.shape
    query_points = query_points.reshape(1, -1, 3)

    return nerf_model.density(query_points).reshape(B, n_f)

# %%
@localscope.mfc
def get_query_points_in_bounds(bounds: Bounds3D, n_pts_per_dim: int) -> np.ndarray:
    """
    Returns a batch of query points in the mesh region.
    """
    x = np.linspace(bounds.x_min, bounds.x_max, n_pts_per_dim)
    y = np.linspace(bounds.y_min, bounds.y_max, n_pts_per_dim)
    z = np.linspace(bounds.z_min, bounds.z_max, n_pts_per_dim)
    xv, yv, zv = np.meshgrid(x, y, z)
    return np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)


query_points_mesh_region_obj_frame = get_query_points_in_bounds(
    get_bounds(mesh).extend(1.5), n_pts_per_dim=100
)

# %%

query_points_mesh_region_obj_frame.shape

# %%
from nerf_grasping.grasp_utils import nerf_to_ig, ig_to_nerf
query_points_mesh_region_isaac_frame = np.copy(
    query_points_mesh_region_obj_frame
)
query_points_mesh_region_nerf_frame = ig_to_nerf(
    query_points_mesh_region_isaac_frame, return_tensor=True
)

# %%
nerf_densities_torch = get_nerf_densities(
    nerf_model, query_points_mesh_region_nerf_frame.reshape(1, -1, 3).float().cuda()
).reshape(query_points_mesh_region_nerf_frame.shape[:-1])
nerf_densities = nerf_densities_torch.detach().cpu().numpy()

# %%
points = query_points_mesh_region_obj_frame.reshape(-1, 3)
densities = nerf_densities.reshape(-1)

# %%
USE_PLOTLY = False
if USE_PLOTLY:
    import plotly.express as px

    fig = px.histogram(
        x=densities,
        log_y=True,
        title="Densities",
        labels={"x": "Values", "y": "Frequency"},
    )

    fig.show()
else:
    plt.hist(densities, log=True)
    plt.title("Densities")
    plt.show()

# %%
@localscope.mfc
def get_colored_points_scatter(points, colors):
    assert len(points.shape) == 2 and points.shape[1] == 3

    points_to_plot = points

    # Use plotly to make scatter3d plot
    scatter = go.Scatter3d(
        x=points_to_plot[:, 0],
        y=points_to_plot[:, 1],
        z=points_to_plot[:, 2],
        mode="markers",
        marker=dict(
            size=5,
            color=colors,
            colorscale="viridis",
            colorbar=dict(title="Density Scale"),
        ),
        name="Query Points Densities",
    )

    return scatter


# %%
threshold = 200.0
filtered_points = points[densities > threshold]
filtered_densities = densities[densities > threshold]
colored_points_scatter = get_colored_points_scatter(
    points=filtered_points, colors=filtered_densities
)

# Add the scatter plot to a figure and display it
fig = plot_mesh(mesh)
fig.add_trace(colored_points_scatter)
# Plot contact_candidates and target_contact_candidates
starts_plot = go.Scatter3d(
    x=contact_candidates[:, 0],
    y=contact_candidates[:, 1],
    z=contact_candidates[:, 2],
    mode="markers",
    marker=dict(
        size=5,
        color="red",
        colorscale="viridis",
    ),
    name="Contact Candidates",
)
ends_plot = go.Scatter3d(
    x=target_contact_candidates[:, 0],
    y=target_contact_candidates[:, 1],
    z=target_contact_candidates[:, 2],
    mode="markers",
    marker=dict(
        size=5,
        color="blue",
        colorscale="viridis",
    ),
    name="Target Contact Candidates",
)
fig.add_trace(starts_plot)
fig.add_trace(ends_plot)


fig.update_layout(legend_orientation="h")

fig.show()


# %%
import numpy as np
from sklearn.cluster import KMeans

def compress_vectors(original_vectors, N):
    # Step 1: Perform k-means clustering
    kmeans = KMeans(n_clusters=N, random_state=42)
    cluster_ids = kmeans.fit_predict(original_vectors)

    # Step 2: Compute the mean of each cluster
    compressed_vectors = np.zeros((N, original_vectors.shape[1]))
    cluster_counts = np.zeros(N)

    for i, cluster_id in enumerate(cluster_ids):
        compressed_vectors[cluster_id] += original_vectors[i]
        cluster_counts[cluster_id] += 1

    for i in range(N):
        if cluster_counts[i] > 0:
            compressed_vectors[i] /= cluster_counts[i]

    # Step 3: Normalize the mean vectors
    # compressed_vectors /= np.linalg.norm(compressed_vectors, axis=1)[:, np.newaxis]

    return compressed_vectors, cluster_ids

# Example usage:
original_vectors = target_contact_candidates - contact_candidates
N = 4
compressed, cluster_ids = compress_vectors(original_vectors, N)
print("Original vectors:")
print(original_vectors)
print("Compressed vectors:")
print(compressed)
print("Cluster IDs for each vector:")
print(cluster_ids)
# %%
# Use plotly to scatter 3d
fig = go.Figure(
    data=[
        go.Scatter3d(
            x=compressed[:, 0],
            y=compressed[:, 1],
            z=compressed[:, 2],
            mode="markers",
            marker=dict(
                size=5,
                color="red",
                colorscale="viridis",
            ),
            name="Compressed Vectors",
        ),
        go.Scatter3d(
            x=original_vectors[:, 0],
            y=original_vectors[:, 1],
            z=original_vectors[:, 2],
            mode="markers",
            marker=dict(
                size=5,
                color="blue",
                colorscale="viridis",
            ),
            name="Original Vectors",
        ),
    ]
)
fig.show()

# %%
np.linalg.norm(compressed, axis=-1)

# %%

