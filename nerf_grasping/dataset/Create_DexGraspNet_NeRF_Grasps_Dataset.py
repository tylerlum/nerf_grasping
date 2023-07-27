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
# # Create DexGraspNet NeRF Grasps
#
# ## Summary (Jul 26, 2023)
#
# The purpose of this script is to iterate through each NeRF object and labeled grasp, sample densities in the grasp trajectory, and storing the data

# %%
from typing import Dict, Any, Tuple, List
import math
import nerf_grasping
from nerf_grasping.grasp_utils import ig_to_nerf
from localscope import localscope
from nerf import utils
import torch
import os
import trimesh
import numpy as np
from plotly import graph_objects as go
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm import tqdm


# %%
# PARAMS
DEXGRASPNET_DATA_ROOT = "/juno/u/tylerlum/github_repos/DexGraspNet/data"
GRASP_DATASET_FOLDER = (
    "2023-07-01_dataset_DESIRED_DIST_TOWARDS_OBJECT_SURFACE_MULTIPLE_STEPS_v2"
)
NERF_CHECKPOINTS_FOLDER = "2023-07-25_nerf_checkpoints"
PLOT_ONLY_ONE = True

# %%
DEXGRASPNET_MESHDATA_ROOT = os.path.join(DEXGRASPNET_DATA_ROOT, "meshdata")
DEXGRASPNET_DATASET_ROOT = os.path.join(
    DEXGRASPNET_DATA_ROOT,
    GRASP_DATASET_FOLDER,
)
NERF_CHECKPOINTS_PATH = os.path.join(
    nerf_grasping.get_repo_root(), NERF_CHECKPOINTS_FOLDER
)
TORCH_NGP_BOUND = 2.0  # Copied from nerf collection script
TORCH_NGP_SCALE = 1.0  # Copied from nerf collection script
N_FINGERS = 4


# %%
@localscope.mfc
def get_object_scale(workspace: str) -> float:
    # BRITTLE
    # Assumes "_0_" only shows up once at the end
    # eg. sem-VideoGameConsole-49ba4f628a955bf03742135a31826a22_0_06
    assert "_0_" in workspace, f"_0_ not in {workspace}"
    idx = workspace.index("_0_")
    scale = float(workspace[idx + 1 :].replace("_", "."))
    return scale


@localscope.mfc
def get_object_code(workspace: str) -> str:
    # BRITTLE
    # Assumes "_0_" only shows up once at the end
    # eg. sem-VideoGameConsole-49ba4f628a955bf03742135a31826a22_0_06
    assert "_0_" in workspace, f"_0_ not in {workspace}"
    idx = workspace.index("_0_")
    object_code = workspace[:idx]
    return object_code


# %%
@localscope.mfc
def validate_nerf_checkpoints_path(nerf_checkpoints_path: str) -> None:
    workspaces = os.listdir(nerf_checkpoints_path)

    num_ok = 0
    for workspace in workspaces:
        path = os.path.join(nerf_checkpoints_path, workspace, "checkpoints")
        if not os.path.exists(path):
            print(f"path {path} does not exist")
            continue

        num_checkpoints = len(os.listdir(path))
        if num_checkpoints > 0:
            print(workspace)
            num_ok += 1

    print(f"num_ok / len(workspaces): {num_ok} / {len(workspaces)}")


validate_nerf_checkpoints_path(
    nerf_checkpoints_path=NERF_CHECKPOINTS_PATH,
)


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



# %%
@localscope.mfc
def get_scene_dict() -> Dict[str, Any]:
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
        scene=get_scene_dict(),
        showlegend=True,
        title="Mesh",
    )

    # Create the figure
    fig = go.Figure(data=[mesh_plot], layout=layout)

    # Return the figure
    return fig


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
@localscope.mfc
def get_nerf_densities(nerf_model, query_points: torch.Tensor):
    """
    Evaluates density of a batch of grasp points, shape [N, 3].
    query_points is torch.Tensor in nerf frame
    """
    N, _ = query_points.shape
    query_points = query_points.reshape(1, N, 3)

    return nerf_model.density(query_points).reshape(N)


# %%
@localscope.mfc
def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


@localscope.mfc
def get_transform(start: np.ndarray, end: np.ndarray, up: np.ndarray) -> np.ndarray:
    # BRITTLE: Assumes new_z and new_y are pretty much perpendicular
    # If not, tries to find closest possible
    new_z = normalize(end - start)
    # new_y should be perpendicular to new_z
    up_dir = normalize(up - start)
    new_y = normalize(up_dir - np.dot(up_dir, new_z) * new_z)
    new_x = np.cross(new_y, new_z)

    transform = np.eye(4)
    transform[:3, :3] = np.stack([new_x, new_y, new_z], axis=1)
    transform[:3, 3] = start
    return transform


# %%
@localscope.mfc
def plot_mesh_and_transforms(
    mesh: trimesh.Trimesh, transforms: List[np.ndarray], n_fingers: int
) -> go.Figure:
    assert len(transforms) == n_fingers, f"{len(transforms)} != {n_fingers}"

    # Add the scatter plot to a figure and display it
    fig = plot_mesh(mesh)
    for finger_idx in range(n_fingers):
        transform = transforms[finger_idx]
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
    return fig


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
) -> np.ndarray:
    grasp_depth_m = grasp_depth_mm / 1000.0
    gripper_finger_width_m = finger_width_mm / 1000.0
    gripper_finger_height_m = finger_height_mm / 1000.0

    # Create grid of points in grasp frame with shape (num_pts_x, num_pts_y, num_pts_z, 3)
    # So that grid_of_points[2, 3, 5] = [x, y, z], where x, y, z are the coordinates of the point
    # Origin of transform is at center of xy at 1/4 of the way into the depth z
    # x is width, y is height, z is depth
    x_coords = np.linspace(
        -gripper_finger_width_m / 2, gripper_finger_width_m / 2, num_pts_x
    )
    y_coords = np.linspace(
        -gripper_finger_height_m / 2, gripper_finger_height_m / 2, num_pts_y
    )
    z_coords = np.linspace(-grasp_depth_m / 4, 3 * grasp_depth_m / 4, num_pts_z)

    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
    assert xx.shape == yy.shape == zz.shape == (num_pts_x, num_pts_y, num_pts_z)
    grid_of_points = np.stack([xx, yy, zz], axis=-1)
    assert grid_of_points.shape == (
        num_pts_x,
        num_pts_y,
        num_pts_z,
        3,
    ), f"{grid_of_points.shape}"
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


@localscope.mfc
def get_transformed_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    n_points = points.shape[0]
    assert points.shape == (n_points, 3), f"{points.shape}"
    assert transform.shape == (4, 4), f"{transform.shape}"

    extra_ones = np.ones((n_points, 1))
    points_homogeneous = np.concatenate([points, extra_ones], axis=1)

    # First (4, 4) @ (4, N) = (4, N)
    # Then transpose to get (N, 4)
    transformed_points = np.matmul(transform, points_homogeneous.T).T

    transformed_points = transformed_points[:, :3]
    assert transformed_points.shape == (n_points, 3), f"{transformed_points.shape}"
    return transformed_points


# %%
@localscope.mfc
def plot_mesh_and_query_points(
    mesh: trimesh.Trimesh,
    query_points_list: List[np.ndarray],
    query_points_colors_list: List[np.ndarray],
    n_fingers: int,
) -> go.Figure:
    assert (
        len(query_points_list) == len(query_points_colors_list) == n_fingers
    ), f"{len(query_points_list)} != {n_fingers}"
    fig = plot_mesh(mesh)

    for finger_idx in range(n_fingers):
        query_points = query_points_list[finger_idx]
        query_points_colors = query_points_colors_list[finger_idx]
        query_point_plot = go.Scatter3d(
            x=query_points[:, 0],
            y=query_points[:, 1],
            z=query_points[:, 2],
            mode="markers",
            marker=dict(
                size=4,
                color=query_points_colors,
                colorscale="viridis",
                colorbar=dict(title="Density Scale") if finger_idx == 0 else {},
            ),
            name=f"Query Point Densities Finger {finger_idx}",
        )
        fig.add_trace(query_point_plot)

    fig.update_layout(legend_orientation="h")  # Avoid overlapping legend
    return fig


# %%
# Iterate through all
workspaces = os.listdir(NERF_CHECKPOINTS_PATH)
for workspace in tqdm(workspaces, desc="nerf workspaces", dynamic_ncols=True):
    # Prepare to read in data
    workspace_path = os.path.join(NERF_CHECKPOINTS_PATH, workspace)
    object_code = get_object_code(workspace)
    object_scale = get_object_scale(workspace)
    mesh_path = os.path.join(
        DEXGRASPNET_MESHDATA_ROOT,
        object_code,
        "coacd",
        "decomposed.obj",
    )
    grasp_dataset_path = os.path.join(
        DEXGRASPNET_DATASET_ROOT,
        f"{object_code}.npy",
    )

    # Check that mesh and grasp dataset exist
    assert os.path.exists(mesh_path), f"mesh_path {mesh_path} does not exist"
    assert os.path.exists(
        grasp_dataset_path
    ), f"grasp_dataset_path {grasp_dataset_path} does not exist"

    # Read in data
    nerf_model = load_nerf(
        path_to_workspace=workspace_path,
        bound=TORCH_NGP_BOUND,
        scale=TORCH_NGP_SCALE,
    )
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.apply_transform(trimesh.transformations.scale_matrix(object_scale))
    full_grasp_data_list = np.load(grasp_dataset_path, allow_pickle=True)
    correct_scale_grasp_data_list = [
        grasp_data
        for grasp_data in full_grasp_data_list
        if math.isclose(grasp_data["scale"], object_scale, rel_tol=1e-3)
    ]

    # Store query points in finger frame (before transform)
    query_points_finger_frame = get_query_points_finger_frame(
        num_pts_x=NUM_PTS_X,
        num_pts_y=NUM_PTS_Y,
        num_pts_z=NUM_PTS_Z,
        grasp_depth_mm=GRASP_DEPTH_MM,
        finger_width_mm=FINGER_WIDTH_MM,
        finger_height_mm=FINGER_HEIGHT_MM,
    )

    for grasp_data in tqdm(
        correct_scale_grasp_data_list, desc="grasp data", dynamic_ncols=True
    ):
        # Go from contact candidates to transforms
        (
            contact_candidates,
            target_contact_candidates,
        ) = get_contact_candidates_and_target_candidates(grasp_data)
        start_points, end_points, up_points = get_start_and_end_and_up_points(
            contact_candidates=contact_candidates,
            target_contact_candidates=target_contact_candidates,
            n_fingers=N_FINGERS,
        )
        transforms = [
            get_transform(start_points[i], end_points[i], up_points[i])
            for i in range(N_FINGERS)
        ]

        # Transform query points
        query_points_object_frame_list = [
            get_transformed_points(
                query_points_finger_frame.reshape(-1, 3), transform
            )
            for transform in transforms
        ]
        query_points_isaac_frame_list = [
            np.copy(query_points_object_frame)
            for query_points_object_frame in query_points_object_frame_list
        ]
        query_points_nerf_frame_list = [
            ig_to_nerf(query_points_isaac_frame, return_tensor=True)
            for query_points_isaac_frame in query_points_isaac_frame_list
        ]

        # Get densities
        nerf_densities = [
            get_nerf_densities(nerf_model, query_points_nerf_frame.float().cuda())
            .reshape(-1)
            .detach()
            .cpu()
            .numpy()
            for query_points_nerf_frame in query_points_nerf_frame_list
        ]

        # Plot
        if PLOT_ONLY_ONE:
            fig = plot_mesh_and_query_points(
                mesh=mesh,
                query_points_list=query_points_object_frame_list,
                query_points_colors_list=nerf_densities,
                n_fingers=N_FINGERS,
            )
            fig.show()
            fig2 = plot_mesh_and_transforms(
                mesh=mesh,
                transforms=transforms,
                n_fingers=N_FINGERS,
            )
            fig2.show()
            assert False, "PLOT_ONLY_ONE is True"

        # Save values

# %%
