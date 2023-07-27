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
from typing import List, Dict, Any, Tuple
import math
import nerf_grasping
from nerf_grasping.grasp_utils import nerf_to_ig, ig_to_nerf
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
@localscope.mfc
def get_nerf_size_scale(workspace: str) -> float:
    # BRITTLE
    # Assumes "_0_" only shows up once at the end
    # eg. sem-VideoGameConsole-49ba4f628a955bf03742135a31826a22_0_06
    assert "_0_" in workspace, f"_0_ not in {workspace}"
    idx = workspace.index("_0_")
    scale = float(workspace[idx + 1 :].replace("_", "."))
    return scale


# %%
# PARAMS
DEXGRASPNET_DATA_ROOT = "/juno/u/tylerlum/github_repos/DexGraspNet/data"
DEXGRASPNET_MESHDATA_ROOT = os.path.join(DEXGRASPNET_DATA_ROOT, "meshdata")
DEXGRASPNET_DATASET_ROOT = os.path.join(
    DEXGRASPNET_DATA_ROOT,
    "2023-07-01_dataset_DESIRED_DIST_TOWARDS_OBJECT_SURFACE_MULTIPLE_STEPS_v2",
)
OBJECT_CODE = "sem-VideoGameConsole-49ba4f628a955bf03742135a31826a22"
MESH_PATH = os.path.join(
    DEXGRASPNET_MESHDATA_ROOT,
    OBJECT_CODE,
    "coacd",
    "decomposed.obj",
)
GRASP_DATASET_PATH = os.path.join(
    DEXGRASPNET_DATASET_ROOT,
    f"{OBJECT_CODE}.npy",
)
NERF_CHECKPOINT_FOLDER = "2023-07-25_nerf_checkpoints"
NERF_MODEL_WORKSPACE = "sem-VideoGameConsole-49ba4f628a955bf03742135a31826a22_0_06"
assert OBJECT_CODE in NERF_MODEL_WORKSPACE
OBJECT_SCALE = get_nerf_size_scale(NERF_MODEL_WORKSPACE)
TORCH_NGP_BOUND = 2.0  # Copied from nerf collection script
TORCH_NGP_SCALE = 1.0  # Copied from nerf collection script


# %%
@localscope.mfc
def validate_nerf_checkpoints(root_dir: str, nerf_checkpoint_folder: str) -> None:
    path_to_nerf_checkpoints = f"{root_dir}/{nerf_checkpoint_folder}"
    workspaces = os.listdir(path_to_nerf_checkpoints)

    num_ok = 0
    for workspace in workspaces:
        path = f"{root_dir}/{nerf_checkpoint_folder}/{workspace}/checkpoints"
        if not os.path.exists(path):
            print(f"path {path} does not exist")
            continue

        num_checkpoints = len(os.listdir(path))
        if num_checkpoints > 0:
            print(workspace)
            num_ok += 1

    print(f"num_ok / len(workspaces): {num_ok} / {len(workspaces)}")


validate_nerf_checkpoints(
    root_dir=nerf_grasping.get_repo_root(),
    nerf_checkpoint_folder=NERF_CHECKPOINT_FOLDER,
)

# %%
FULL_GRASP_DATA_LIST = np.load(GRASP_DATASET_PATH, allow_pickle=True)
FULL_GRASP_DATA_LIST.shape

# %%
# Get correct scale grasps
CORRECT_SCALE_GRASP_DATA_LIST = [
    grasp_data
    for grasp_data in FULL_GRASP_DATA_LIST
    if math.isclose(grasp_data["scale"], OBJECT_SCALE, rel_tol=1e-3)
]
print(f"len(CORRECT_SCALE_GRASP_DATA_LIST): {len(CORRECT_SCALE_GRASP_DATA_LIST)}")

# %%
GRASP_IDX = 0
GRASP_DATA = CORRECT_SCALE_GRASP_DATA_LIST[GRASP_IDX]
print(f"GRASP_IDX: {GRASP_IDX}")
print(f"GRASP_DATA.keys(): {GRASP_DATA.keys()}")


# %%
# Get contact candidates and target contact candidates
@localscope.mfc
def get_contact_candidates_and_target_candidates(
    grasp_data: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    link_name_to_contact_candidates = grasp_data["link_name_to_contact_candidates"]
    link_name_to_target_contact_candidates = grasp_data[
        "link_name_to_target_contact_candidates"
    ]
    contact_candidates = np.concatenate(
        [
            contact_candidate
            for _, contact_candidate in link_name_to_contact_candidates.items()
        ],
        axis=0,
    )
    target_contact_candidates = np.concatenate(
        [
            target_contact_candidate
            for _, target_contact_candidate in link_name_to_target_contact_candidates.items()
        ],
        axis=0,
    )
    return contact_candidates, target_contact_candidates


(
    contact_candidates,
    target_contact_candidates,
) = get_contact_candidates_and_target_candidates(GRASP_DATA)
print(f"contact_candidates.shape: {contact_candidates.shape}")
print(f"target_contact_candidates.shape: {target_contact_candidates.shape}")


# %%
@localscope.mfc
def get_ordered_cluster_ids(cluster_ids: np.ndarray) -> np.ndarray:
    # Want order of clusters to be ascending
    idxs_of_change = []
    for i, cluster_id in enumerate(cluster_ids):
        if i == 0:
            continue
        if cluster_id == cluster_ids[i - 1]:
            continue
        idxs_of_change.append(i)

    idxs_of_change.append(len(cluster_ids))
    ordered_cluster_ids = []
    for i, idx in enumerate(idxs_of_change):
        if i == 0:
            num_to_add = idx
        else:
            num_to_add = idx - idxs_of_change[i - 1]
        new_cluster_id = i
        ordered_cluster_ids += [new_cluster_id] * num_to_add
    return np.array(ordered_cluster_ids)


N_CONTACTS_PER_FINGER = 6
N_FINGERS = 4


@localscope.mfc
def get_start_and_end_points(
    contact_candidates: np.ndarray,
    target_contact_candidates: np.ndarray,
    n_fingers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_contacts = contact_candidates.shape[0]
    assert (
        contact_candidates.shape == target_contact_candidates.shape == (n_contacts, 3)
    )

    from sklearn.cluster import KMeans

    # Cluster by directions
    directions = target_contact_candidates - contact_candidates
    kmeans = KMeans(n_clusters=n_fingers, random_state=42)
    cluster_ids = kmeans.fit_predict(directions)
    ordered_cluster_ids = get_ordered_cluster_ids(cluster_ids)

    start_points, end_points = [] , []
    for finger_idx in range(n_fingers):
        contact_candidates_this_finger = contact_candidates[
            ordered_cluster_ids == finger_idx
        ]
        target_contact_candidates_this_finger = target_contact_candidates[
            ordered_cluster_ids == finger_idx
        ]
        start_points.append(contact_candidates_this_finger.mean(axis=0))
        end_points.append(target_contact_candidates_this_finger.mean(axis=0))

    start_points, end_points = np.array(start_points), np.array(end_points)
    assert start_points.shape == end_points.shape == (n_fingers, 3)
    return start_points, end_points



# %%
@localscope.mfc
def get_start_and_end_points_faster(
    contact_candidates: np.ndarray,
    target_contact_candidates: np.ndarray,
    n_fingers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import KMeans
    contact_candidates_per_finger = contact_candidates.reshape(n_fingers, -1, 3)
    target_contact_candidates_per_finger = target_contact_candidates.reshape(
        n_fingers, -1, 3
    )
    start_points = contact_candidates_per_finger.mean(axis=1)
    end_points = target_contact_candidates_per_finger.mean(axis=1)
    assert start_points.shape == end_points.shape == (n_fingers, 3)
    return np.array(start_points), np.array(end_points)

# start_points, end_points = get_start_and_end_points(
#     contact_candidates=contact_candidates,
#     target_contact_candidates=target_contact_candidates,
#     n_fingers=N_FINGERS,
# )
start_points, end_points = get_start_and_end_points_faster(
    contact_candidates=contact_candidates,
    target_contact_candidates=target_contact_candidates,
    n_fingers=N_FINGERS,
)
start_points, end_points 

# %%
# Open mesh
mesh = trimesh.load(MESH_PATH, force="mesh")

# %%
print(f"Before scaling, mesh.centroid = {mesh.centroid}")

# %%
mesh.apply_transform(trimesh.transformations.scale_matrix(OBJECT_SCALE))

# %%
print(f"After scaling, mesh.centroid = {mesh.centroid}")


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
def plot_mesh(mesh: trimesh.Trimesh, color="lightpink") -> go.Figure:
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


@localscope.mfc
def load_nerf(path_to_workspace: str, bound: float, scale: float):
    root_dir = nerf_grasping.get_repo_root()

    parser = utils.get_config_parser()
    opt = parser.parse_args(
        [
            "--workspace",
            f"{path_to_workspace}",
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
    path_to_workspace=os.path.join(
        nerf_grasping.get_repo_root(), NERF_CHECKPOINT_FOLDER, NERF_MODEL_WORKSPACE
    ),
    bound=TORCH_NGP_BOUND,
    scale=TORCH_NGP_SCALE,
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
n_pts = query_points_mesh_region_obj_frame.shape[0]
assert query_points_mesh_region_obj_frame.shape == (n_pts, 3)
print(
    f"query_points_mesh_region_obj_frame.shape: {query_points_mesh_region_obj_frame.shape}"
)

# %%
query_points_mesh_region_isaac_frame = np.copy(query_points_mesh_region_obj_frame)
query_points_mesh_region_nerf_frame = ig_to_nerf(
    query_points_mesh_region_isaac_frame, return_tensor=True
)

# %%
nerf_densities_torch = get_nerf_densities(
    nerf_model, query_points_mesh_region_nerf_frame.reshape(1, n_pts, 3).float().cuda()
).reshape(n_pts)
nerf_densities = nerf_densities_torch.detach().cpu().numpy()

# %%
points = query_points_mesh_region_obj_frame.reshape(n_pts, 3)
densities = nerf_densities.reshape(n_pts)

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
THRESHOLD = 200.0
filtered_points = points[densities > THRESHOLD]
filtered_densities = densities[densities > THRESHOLD]
colored_points_scatter = get_colored_points_scatter(
    points=filtered_points, colors=filtered_densities
)

# Add the scatter plot to a figure and display it
fig = plot_mesh(mesh)
fig.add_trace(colored_points_scatter)
# Plot contact_candidates and target_contact_candidates
starts_plot = go.Scatter3d(
    x=start_points[:, 0],
    y=start_points[:, 1],
    z=start_points[:, 2],
    mode="markers",
    marker=dict(
        size=5,
        color="red",
        colorscale="viridis",
    ),
    name="Start Points",
)
ends_plot = go.Scatter3d(
    x=end_points[:, 0],
    y=end_points[:, 1],
    z=end_points[:, 2],
    mode="markers",
    marker=dict(
        size=5,
        color="blue",
        colorscale="viridis",
    ),
    name="End Points",
)
fig.add_trace(starts_plot)
fig.add_trace(ends_plot)


fig.update_layout(legend_orientation="h")

fig.show()
