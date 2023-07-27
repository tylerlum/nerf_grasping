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

    start_points, end_points = [], []
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
def get_start_and_end_and_up_points(
    contact_candidates: np.ndarray,
    target_contact_candidates: np.ndarray,
    n_fingers: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # BRITTLE: Assumes same number of contact points per finger
    # BRITTLE: Assumes UP_POINT_IDX is position of contact candidate up from center
    UP_POINT_IDX = 3
    contact_candidates_per_finger = contact_candidates.reshape(n_fingers, -1, 3)
    target_contact_candidates_per_finger = target_contact_candidates.reshape(
        n_fingers, -1, 3
    )
    start_points = contact_candidates_per_finger.mean(axis=1)
    end_points = target_contact_candidates_per_finger.mean(axis=1)
    up_points = contact_candidates_per_finger[:, UP_POINT_IDX, :]
    assert start_points.shape == end_points.shape == up_points.shape == (n_fingers, 3)
    return np.array(start_points), np.array(end_points), np.array(up_points)

# start_points, end_points = get_start_and_end_points(
#     contact_candidates=contact_candidates,
#     target_contact_candidates=target_contact_candidates,
#     n_fingers=N_FINGERS,
# )
start_points, end_points, up_points = get_start_and_end_and_up_points(
    contact_candidates=contact_candidates,
    target_contact_candidates=target_contact_candidates,
    n_fingers=N_FINGERS,
)
start_points, end_points, up_points

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
contact_candidates_plot = go.Scatter3d(
    x=contact_candidates[:, 0],
    y=contact_candidates[:, 1],
    z=contact_candidates[:, 2],
    mode="markers",
    marker=dict(
        size=4,
        color="red",
        colorscale="viridis",
    ),
    name="Contact Candidates",
)
target_contact_candidates_plot = go.Scatter3d(
    x=target_contact_candidates[:, 0],
    y=target_contact_candidates[:, 1],
    z=target_contact_candidates[:, 2],
    mode="markers",
    marker=dict(
        size=4,
        color="blue",
        colorscale="viridis",
    ),
    name="Target Contact Candidates",
)
starts_plot = go.Scatter3d(
    x=start_points[:, 0],
    y=start_points[:, 1],
    z=start_points[:, 2],
    mode="markers",
    marker=dict(
        size=8,
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
        size=8,
        color="blue",
        colorscale="viridis",
    ),
    name="End Points",
)
ups_plot = go.Scatter3d(
    x=up_points[:, 0],
    y=up_points[:, 1],
    z=up_points[:, 2],
    mode="markers",
    marker=dict(
        size=8,
        color="yellow",
        colorscale="viridis",
    ),
    name="Up Points",
)
fig.add_trace(contact_candidates_plot)
fig.add_trace(target_contact_candidates_plot)
fig.add_trace(starts_plot)
fig.add_trace(ends_plot)
fig.add_trace(ups_plot)


fig.update_layout(legend_orientation="h")

fig.show()

# %%
@localscope.mfc
def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)

@localscope.mfc
def get_transform(start: np.ndarray, end: np.ndarray, up: np.ndarray) -> np.ndarray:
    # BRITTLE: Assumes new_z and new_y are pretty much perpendicular
    # If not, tries to find closest possible
    new_z = normalize(end-start)
    # new_y should be perpendicular to new_z
    up_dir = normalize(up-start)
    new_y = normalize(up_dir - np.dot(up_dir, new_z) * new_z)
    new_x = np.cross(new_y, new_z)

    transform = np.eye(4)
    transform[:3, :3] = np.stack([new_x, new_y, new_z], axis=1)
    transform[:3, 3] = start
    return transform

FINGER_IDX = 0
get_transform(start_points[FINGER_IDX], end_points[FINGER_IDX], up_points[FINGER_IDX])


# %%
# Add the scatter plot to a figure and display it
fig = plot_mesh(mesh)
for finger_idx in range(N_FINGERS):
    transform = get_transform(start_points[finger_idx], end_points[finger_idx], up_points[finger_idx])
    length = 0.02
    origin = np.array([0, 0, 0])
    x_axis = np.array([length, 0, 0])
    y_axis = np.array([0, length, 0])
    z_axis = np.array([0, 0, length])

    new_origin = transform @ np.concatenate([origin, [1]])
    new_x_axis = transform @ np.concatenate([x_axis, [1]])
    new_y_axis = transform @ np.concatenate([y_axis, [1]])
    new_z_axis = transform @ np.concatenate([z_axis, [1]])
    x_plot = go.Scatter3d(
        x=[new_origin[0], new_x_axis[0]],
        y=[new_origin[1], new_x_axis[1]],
        z=[new_origin[2], new_x_axis[2]],
        mode="lines",
        marker=dict(
            size=8,
            color="red",
            colorscale="viridis",
        ),
        name=f"Finger {finger_idx} X Axis",
    )
    y_plot = go.Scatter3d(
        x=[new_origin[0], new_y_axis[0]],
        y=[new_origin[1], new_y_axis[1]],
        z=[new_origin[2], new_y_axis[2]],
        mode="lines",
        marker=dict(
            size=8,
            color="green",
            colorscale="viridis",
        ),
        name=f"Finger {finger_idx} Y Axis",
    )
    z_plot = go.Scatter3d(
        x=[new_origin[0], new_z_axis[0]],
        y=[new_origin[1], new_z_axis[1]],
        z=[new_origin[2], new_z_axis[2]],
        mode="lines",
        marker=dict(
            size=8,
            color="blue",
            colorscale="viridis",
        ),
        name=f"Finger {finger_idx} Z Axis",
    )

    fig.add_trace(x_plot)
    fig.add_trace(y_plot)
    fig.add_trace(z_plot)


fig.update_layout(legend_orientation="h")

fig.show()


# %%
# Grid of points in grasp frame (x, y, z)
GRASP_DEPTH_MM = 20
FINGER_WIDTH_MM = 10
FINGER_HEIGHT_MM = 15

# Want points equally spread out in space
DIST_BTWN_PTS_MM = 0.5

# +1 to include both end points
NUM_PTS_X = int(GRASP_DEPTH_MM / DIST_BTWN_PTS_MM) + 1
NUM_PTS_Y = int(FINGER_WIDTH_MM / DIST_BTWN_PTS_MM) + 1
NUM_PTS_Z = int(FINGER_HEIGHT_MM / DIST_BTWN_PTS_MM) + 1

@localscope.mfc
def get_query_points_finger_frame(
    num_pts_x: int,
    num_pts_y: int,
    num_pts_z: int,
    grasp_depth_mm: float,
    finger_width_mm: float,
    finger_height_mm: float,
):
    num_pts = num_pts_x * num_pts_y * num_pts_z
    print(f"num_pts: {num_pts}")
    grasp_depth_m = grasp_depth_mm / 1000.0
    gripper_finger_width_m = finger_width_mm / 1000.0
    gripper_finger_height_m = finger_height_mm / 1000.0

    # Create grid of points in grasp frame with shape (num_pts_x, num_pts_y, num_pts_z, 3)
    # So that grid_of_points[2, 3, 5] = [x, y, z], where x, y, z are the coordinates of the point
    # Origin of transform is at center of xy at one end of z
    # x is width, y is height, z is depth
    x_coords = np.linspace(-gripper_finger_width_m / 2, gripper_finger_width_m / 2, num_pts_x)
    y_coords = np.linspace(
        -gripper_finger_height_m / 2, gripper_finger_height_m / 2, num_pts_y
    )
    z_coords = np.linspace(
        0.0, grasp_depth_m, num_pts_z
    )

    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
    assert xx.shape == yy.shape == zz.shape == (num_pts_x, num_pts_y, num_pts_z)
    grid_of_points = np.stack([xx, yy, zz], axis=-1)
    assert grid_of_points.shape == (num_pts_x, num_pts_y, num_pts_z, 3)
    return grid_of_points

query_points_finger_frame = get_query_points_finger_frame(
    num_pts_x=NUM_PTS_X,
    num_pts_y=NUM_PTS_Y,
    num_pts_z=NUM_PTS_Z,
    grasp_depth_mm=GRASP_DEPTH_MM,
    finger_width_mm=FINGER_WIDTH_MM,
    finger_height_mm=FINGER_HEIGHT_MM,
)

# %%
fig = go.Figure(
    data=[
        go.Scatter3d(
            x=query_points_finger_frame[:, :, :, 0].reshape(-1),
            y=query_points_finger_frame[:, :, :, 1].reshape(-1),
            z=query_points_finger_frame[:, :, :, 2].reshape(-1),
            mode="markers",
            marker=dict(
                size=4,
                color="red",
                colorscale="viridis",
            ),
            name="Query Points",
        )
    ],
    layout=go.Layout(
        scene=get_scene_dict(mesh),
        showlegend=True,
        title="Query Points",
    ),
)
fig.show()

# %%
@localscope.mfc
def get_transformed_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    n_points = points.shape[0]
    assert points.shape == (n_points, 3)
    assert transform.shape == (4, 4)

    extra_ones = np.ones((n_points, 1))
    points_homogeneous = np.concatenate([points, extra_ones], axis=1)

    # First (4, 4) @ (4, N) = (4, N)
    # Then transpose to get (N, 4)
    transformed_points = np.matmul(transform, points_homogeneous.T).T

    transformed_points = transformed_points[:, :3]
    assert transformed_points.shape == (n_points, 3)
    return transformed_points

# %%
fig = plot_mesh(mesh)

for finger_idx in range(N_FINGERS):
    transform = get_transform(start_points[finger_idx], end_points[finger_idx], up_points[finger_idx])
    query_points_object_frame = get_transformed_points(
        query_points_finger_frame.reshape(-1, 3), transform
    ).reshape(query_points_finger_frame.shape)
    query_point_plot = go.Scatter3d(
        x=query_points_object_frame[:, :, :, 0].reshape(-1),
        y=query_points_object_frame[:, :, :, 1].reshape(-1),
        z=query_points_object_frame[:, :, :, 2].reshape(-1),
        mode="markers",
        marker=dict(
            size=4,
            color="red",
            colorscale="viridis",
        ),
        name=f"Query Points Finger {finger_idx}",
    )
    fig.add_trace(query_point_plot)

fig.show()
# %%
