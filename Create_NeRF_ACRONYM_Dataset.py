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
# # Create NeRF ACRONYM Dataset
#
# ## Summary (April 21, 2023)
#
# The purpose of this script is to create a dataset for predicting the quality of grasps from NeRF representations of objects.
#
# ## Script Inputs
#
# ## Dataset
# ```
# {output_dataset_dir}
# ├── isaac_3Shelves_ea3807911b86d7e7b53aed25092632f_0.0046205018
# │   ├── 0.pkl
# │   ├── 1.pkl
# │   ├── 2.pkl
# │   ├── ...
# │   ├── 1999.pkl
# ├── isaac_5Shelves_fc45d911f8b87b1aca32354d178e0245_0.0023979975
# │   ├── 0.pkl
# │   ├── 1.pkl
# │   ├── 2.pkl
# │   ├── ...
# │   ├── 1999.pkl
# ```
#
# where `<x>.pkl` is a `dict` with the following keys:
#
# ```
# {
#   "nerf_grid_input": np.array w/ shape (4, 83, 21, 37),
#   "grasp_success": 0 or 1,
#   "mesh_centroid_isaac_frame": np.array w/ shape (3,),
#   "acronym_data_filename": hdf5 filename
#   "grasp_idx": idx of this grasp within acronym_data_filename (should be same as filename)
#   <may add other helpful meta-data>
# }
# ```
# where the first channel of `nerf_grid_input` is the NeRF density at this point
# and the remaining channels of `nerf_grid_input` are the x, y, and z coordinates of the point.
# and these coordinates are in isaac frame, so we would likely subtract the mesh centroid to get the relative position of the point.

# %%
from nerf_grasping.sim import acronym_objects

# %%
import wandb
import os
import h5py
import numpy as np
from localscope import localscope
import time
import math
from tqdm import tqdm

import random
import torch
import plotly.graph_objects as go
import trimesh

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mplot3d

from nerf_grasping.grasp_utils import nerf_to_ig, ig_to_nerf


# %% [markdown]
# # Read in Objects

# %%
obj_class_names = [k for k in tqdm(list(acronym_objects.__dict__.keys())) if k.startswith("Obj_")]

# %%
nerf_checkpoints_path = "nerf_checkpoints"

# %%
checkpoints = os.listdir(nerf_checkpoints_path)
checkpoints_set = set(checkpoints)

# %%
# Get objs with checkpoint, hacky
objs = []
for obj_class_name in tqdm(obj_class_names):
    workspace = "isaac_" + eval(f"acronym_objects.{obj_class_name}.workspace")
    if workspace in checkpoints_set:
        objs.append(eval(f"acronym_objects.{obj_class_name}()"))

# %%
print(f"Found {len(objs)} objs")
print(f"First 10 are {objs[:10]}")

# %%
found_invalid_workspace = False
for obj in objs:
    workspace_path = os.path.join(nerf_checkpoints_path, "isaac_" + obj.workspace)
    checkpoints_path = os.path.join(workspace_path, "checkpoints")
    if not os.path.exists(workspace_path) or not os.path.exists(checkpoints_path) or len(os.listdir(checkpoints_path)) == 0:
        print(f"workspace_path = {workspace_path} missing files")
        found_invalid_workspace = True
if not found_invalid_workspace:
    print("All workspaces are valid")

# %%
# useful constants
LEFT_TIP_POSITION_GRASP_FRAME = np.array(
    [4.10000000e-02, -7.27595772e-12, 1.12169998e-01]
)
RIGHT_TIP_POSITION_GRASP_FRAME = np.array(
    [-4.10000000e-02, -7.27595772e-12, 1.12169998e-01]
)


# %%
# Useful functions
@localscope.mfc
def position_to_transformed_positions(position, transforms):
    assert position.shape == (3,)
    assert len(transforms.shape) == 3 and transforms.shape[1:] == (4, 4)
    num_transforms = transforms.shape[0]

    # transformed_positions = (transforms @ np.array([*position, 1.0]).reshape(1, 4, 1))[
    #     :, :3, :
    # ].squeeze()
    transformed_positions = (transforms @ np.array([*position, 1.0]).reshape(1, 4, 1))[
        :, :3, :
    ].squeeze(-1)
    assert transformed_positions.shape == (num_transforms, 3)
    return transformed_positions


@localscope.mfc
def position_to_transformed_positions_unvectorized(position, transforms):
    assert position.shape == (3,)
    assert len(transforms.shape) == 3 and transforms.shape[1:] == (4, 4)
    num_transforms = transforms.shape[0]

    transformed_positions = []
    for i in range(num_transforms):
        transformed_positions.append((transforms[i] @ np.array([*position, 1.0]))[:3])
    transformed_positions = np.stack(transformed_positions)
    return transformed_positions


@localscope.mfc
def run_sanity_check(position, transforms):
    # Non-vectorized
    start = time.time()
    positions_object_frame = position_to_transformed_positions_unvectorized(
        position=position, transforms=transforms
    )
    print(f"Non-vectorized took {1000 * (time.time() - start):.2f} ms")

    # Vectorized version
    start = time.time()
    positions_object_frame_2 = position_to_transformed_positions(
        position=position, transforms=transforms
    )
    print(f"Vectorized took {1000 * (time.time() - start):.2f} ms")

    assert np.max(np.abs(positions_object_frame - positions_object_frame_2)) < 1e-5
    print("Passed the test, they match!")
    return


# %%
# run_sanity_check(position=LEFT_TIP_POSITION_GRASP_FRAME, transforms=grasp_transforms)

# %%
@localscope.mfc
def plot_obj(obj_filepath, scale=1.0, offset=None, color="lightpink"):
    if offset is None:
        offset = np.zeros(3)

    # Read in the OBJ file
    with open(obj_filepath, "r") as f:
        lines = f.readlines()

    # Extract the vertex coordinates and faces from the OBJ file
    vertices = []
    faces = []
    for line in lines:
        if line.startswith("v "):
            vertex = [float(i) * scale for i in line.split()[1:4]]
            vertices.append(vertex)
        elif line.startswith("f "):
            face = [int(i.split("/")[0]) - 1 for i in line.split()[1:4]]
            faces.append(face)

    # Convert the vertex coordinates and faces to numpy arrays
    vertices = np.array(vertices)
    faces = np.array(faces)

    assert len(vertices.shape) == 2 and vertices.shape[1] == 3
    assert len(faces.shape) == 2 and faces.shape[1] == 3

    vertices += offset.reshape(1, 3)

    # Create the mesh3d trace
    mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=0.5,
        name=f"Mesh: {os.path.basename(obj_filepath)}",
    )

    # Create the layout
    coordinates = "Object Coordinates" if np.all(offset == 0) else "Isaac Coordinates"
    layout = go.Layout(
        scene=dict(xaxis=dict(title="X"), yaxis=dict(title="Y"), zaxis=dict(title="Z")),
        showlegend=True,
        title=f"Mesh: {os.path.basename(obj_filepath)} ({coordinates})",
    )

    # Create the figure
    fig = go.Figure(data=[mesh], layout=layout)

    # Return the figure
    return fig


# %%
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation
@localscope.mfc
def plot_obj(obj_filepath, scale=1.0, offset=None, color="lightpink", rotation_offset=None):
    if offset is None:
        offset = np.zeros(3)

    # Read in the OBJ file
    with open(obj_filepath, "r") as f:
        lines = f.readlines()

    # Extract the vertex coordinates and faces from the OBJ file
    vertices = []
    faces = []
    for line in lines:
        if line.startswith("v "):
            vertex = [float(i) * scale for i in line.split()[1:4]]
            vertices.append(vertex)
        elif line.startswith("f "):
            face = [int(i.split("/")[0]) - 1 for i in line.split()[1:4]]
            faces.append(face)

    # Convert the vertex coordinates and faces to numpy arrays
    vertices = np.array(vertices)
    faces = np.array(faces)

    assert len(vertices.shape) == 2 and vertices.shape[1] == 3
    assert len(faces.shape) == 2 and faces.shape[1] == 3

    vertices += offset.reshape(1, 3)

    # Apply rotation offset
    if rotation_offset is not None:
        r = Rotation.from_quat(rotation_offset)
        vertices = r.apply(vertices)

    # Create the mesh3d trace
    mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=0.5,
        name=f"Mesh: {os.path.basename(obj_filepath)}",
    )

    # Create the layout
    coordinates = "Object Coordinates" if np.all(offset == 0) else "Isaac Coordinates"
    layout = go.Layout(
        scene=dict(xaxis=dict(title="X"), yaxis=dict(title="Y"), zaxis=dict(title="Z")),
        showlegend=True,
        title=f"Mesh: {os.path.basename(obj_filepath)} ({coordinates})",
    )

    # Create the figure
    fig = go.Figure(data=[mesh], layout=layout)

    # Return the figure
    return fig

# %%


@localscope.mfc
def get_mesh_centroid(mesh, scale=1):
    return np.array(mesh.centroid) * scale


# Get bounds of mesh
@localscope.mfc
def get_mesh_bounds(mesh, scale=1):
    min_points, max_points = mesh.bounds
    return np.array(min_points) * scale, np.array(max_points) * scale

@localscope.mfc
def get_mesh_centroid_scatter(mesh_centroid, offset=None):
    if offset is None:
        offset = np.zeros(3)

    translated_mesh_centroid = mesh_centroid + offset
    scatter = go.Scatter3d(
        x=[translated_mesh_centroid[0]],
        y=[translated_mesh_centroid[1]],
        z=[translated_mesh_centroid[2]],
        mode="markers",
        marker=dict(size=10, color="black"),
        name="Mesh Centroid",
    )
    return scatter


@localscope.mfc
def get_mesh_origin_lines(offset=None):
    if offset is None:
        offset = np.zeros(3)

    x_lines = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    y_lines = np.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]])
    z_lines = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]])
    x_lines += offset
    y_lines += offset
    z_lines += offset
    lines = [
        go.Scatter3d(
            x=x_lines[:, 0],
            y=x_lines[:, 1],
            z=x_lines[:, 2],
            mode="lines",
            line=dict(width=2, color="red"),
            name="Mesh Origin X Axis",
        ),
        go.Scatter3d(
            x=y_lines[:, 0],
            y=y_lines[:, 1],
            z=y_lines[:, 2],
            mode="lines",
            line=dict(width=2, color="green"),
            name="Mesh Origin Y Axis",
        ),
        go.Scatter3d(
            x=z_lines[:, 0],
            y=z_lines[:, 1],
            z=z_lines[:, 2],
            mode="lines",
            line=dict(width=2, color="blue"),
            name="Mesh Origin Z Axis",
        ),
    ]
    return lines

# %%
@localscope.mfc
def get_grasp_gripper_lines(grasp_transforms, grasp_successes, offset=None):
    if offset is None:
        offset = np.zeros(3)

    raw_left_tip = np.array([4.10000000e-02, -7.27595772e-12, 1.12169998e-01])
    raw_right_tip = np.array([-4.10000000e-02, -7.27595772e-12, 1.12169998e-01])
    raw_left_knuckle = np.array([4.10000000e-02, -7.27595772e-12, 6.59999996e-02])
    raw_right_knuckle = np.array([-4.10000000e-02, -7.27595772e-12, 6.59999996e-02])
    raw_hand_origin = np.array([0.0, 0.0, 0.0])
    left_tips = position_to_transformed_positions(
        position=raw_left_tip, transforms=grasp_transforms
    )
    right_tips = position_to_transformed_positions(
        position=raw_right_tip, transforms=grasp_transforms
    )
    left_knuckles = position_to_transformed_positions(
        position=raw_left_knuckle, transforms=grasp_transforms
    )
    right_knuckles = position_to_transformed_positions(
        position=raw_right_knuckle, transforms=grasp_transforms
    )
    hand_origins = position_to_transformed_positions(
        position=raw_hand_origin, transforms=grasp_transforms
    )

    assert (
        left_tips.shape
        == right_tips.shape
        == left_knuckles.shape
        == right_knuckles.shape
        == hand_origins.shape
        == (len(grasp_successes), 3)
    )
    left_tips += offset.reshape(1, 3)
    right_tips += offset.reshape(1, 3)
    left_knuckles += offset.reshape(1, 3)
    right_knuckles += offset.reshape(1, 3)
    hand_origins += offset.reshape(1, 3)

    grasp_lines = []
    for i, (
        left_tip,
        right_tip,
        left_knuckle,
        right_knuckle,
        hand_origin,
        grasp_success,
    ) in enumerate(
        zip(
            left_tips,
            right_tips,
            left_knuckles,
            right_knuckles,
            hand_origins,
            grasp_successes,
        )
    ):
        assert grasp_success in [0, 1]
        color = "green" if grasp_success == 1 else "red"

        # left tip => left knuckle => right knuckle => right tip => right_knuckle => btwn knuckles => hand_origin
        btwn_knuckles = (left_knuckle + right_knuckle) / 2

        points = np.stack(
            [
                left_tip,
                left_knuckle,
                right_knuckle,
                right_tip,
                btwn_knuckles,
                hand_origin,
            ],
            axis=0,
        )
        assert points.shape == (6, 3)

        # Create 1 continous line per grasp
        grasp_line = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="lines",
            line=dict(color=color, width=5),
            name=f"Grasp {i}",
        )
        grasp_lines.append(grasp_line)
    return grasp_lines

# %%
@localscope.mfc
def get_grasp_ray_lines(grasp_transforms, grasp_successes, offset=None):
    if offset is None:
        offset = np.zeros(3)

    raw_left_tip = np.array([4.10000000e-02, -7.27595772e-12, 1.12169998e-01])
    raw_right_tip = np.array([-4.10000000e-02, -7.27595772e-12, 1.12169998e-01])
    left_tips = position_to_transformed_positions(
        position=raw_left_tip, transforms=grasp_transforms
    )
    right_tips = position_to_transformed_positions(
        position=raw_right_tip, transforms=grasp_transforms
    )

    assert left_tips.shape == right_tips.shape == (len(grasp_successes), 3)
    left_tips += offset.reshape(1, 3)
    right_tips += offset.reshape(1, 3)

    grasp_lines = []
    for i, (
        left_tip,
        right_tip,
        grasp_success,
    ) in enumerate(
        zip(
            left_tips,
            right_tips,
            grasp_successes,
        )
    ):
        assert grasp_success in [0, 1]
        color = "green" if grasp_success == 1 else "red"

        # left tip => right_tip
        points = np.stack(
            [
                left_tip,
                right_tip,
            ],
            axis=0,
        )
        assert points.shape == (2, 3)

        # Create 1 continous line per grasp
        grasp_line = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="lines",
            line=dict(color=color, width=5),
            name=f"Grasp {i}",
        )
        grasp_lines.append(grasp_line)
    return grasp_lines
# %%
@localscope.mfc(
    allowed=["LEFT_TIP_POSITION_GRASP_FRAME", "RIGHT_TIP_POSITION_GRASP_FRAME"]
)
def get_grasp_query_points_grasp_frame():
    # Grid of points in grasp frame (x, y, z)
    GRIPPER_WIDTH_MM = 82
    GRIPPER_FINGER_WIDTH_MM = 20
    GRIPPER_FINGER_HEIGHT_MM = 36

    # Want points equally spread out in space
    DIST_BTWN_PTS_MM = 1

    NUM_PTS_X = int(GRIPPER_WIDTH_MM / DIST_BTWN_PTS_MM)
    NUM_PTS_Y = int(GRIPPER_FINGER_WIDTH_MM / DIST_BTWN_PTS_MM)
    NUM_PTS_Z = int(GRIPPER_FINGER_HEIGHT_MM / DIST_BTWN_PTS_MM)

    assert NUM_PTS_X * DIST_BTWN_PTS_MM == GRIPPER_WIDTH_MM
    assert NUM_PTS_Y * DIST_BTWN_PTS_MM == GRIPPER_FINGER_WIDTH_MM
    assert NUM_PTS_Z * DIST_BTWN_PTS_MM == GRIPPER_FINGER_HEIGHT_MM
    num_pts = (NUM_PTS_X + 1) * (NUM_PTS_Y + 1) * (NUM_PTS_Z + 1)
    print(f"num_pts: {num_pts}")

    GRIPPER_WIDTH_M = GRIPPER_WIDTH_MM / 1000.0
    GRIPPER_FINGER_WIDTH_M = GRIPPER_FINGER_WIDTH_MM / 1000.0
    GRIPPER_FINGER_HEIGHT_M = GRIPPER_FINGER_HEIGHT_MM / 1000.0

    # Create grid of points in grasp frame with shape (NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z, 3)
    # So that grid_of_points[2, 3, 5] = [x, y, z], where x, y, z are the coordinates of the point
    x_coords = np.linspace(-GRIPPER_WIDTH_M / 2, GRIPPER_WIDTH_M / 2, NUM_PTS_X + 1)
    y_coords = np.linspace(
        -GRIPPER_FINGER_WIDTH_M / 2, GRIPPER_FINGER_WIDTH_M / 2, NUM_PTS_Y + 1
    )
    z_coords = np.linspace(0, GRIPPER_FINGER_HEIGHT_M, NUM_PTS_Z + 1)

    # Offset so centered between LEFT_TIP_POSITION_GRASP_FRAME and RIGHT_TIP_POSITION_GRASP_FRAME
    center_point = (LEFT_TIP_POSITION_GRASP_FRAME + RIGHT_TIP_POSITION_GRASP_FRAME) / 2
    x_coords += center_point[0]
    y_coords += center_point[1]
    z_coords += center_point[2]

    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
    assert (
        xx.shape
        == yy.shape
        == zz.shape
        == (NUM_PTS_X + 1, NUM_PTS_Y + 1, NUM_PTS_Z + 1)
    )
    grid_of_points = np.stack([xx, yy, zz], axis=-1)
    assert grid_of_points.shape == (NUM_PTS_X + 1, NUM_PTS_Y + 1, NUM_PTS_Z + 1, 3)
    return grid_of_points


# %%
# Transform query points to object frame


@localscope.mfc
def get_transformed_points(points, transform):
    assert len(points.shape) == 2 and points.shape[1] == 3
    assert transform.shape == (4, 4)

    points_homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)

    # First (4, 4) @ (4, N) = (4, N)
    # Then transpose to get (N, 4)
    transformed_points = np.matmul(transform, points_homogeneous.T).T

    return transformed_points[:, :3]

# %%
# Visualize points
@localscope.mfc
def get_points_scatter(points, offset=None):
    if offset is None:
        offset = np.zeros(3)

    assert len(points.shape) == 2 and points.shape[1] == 3

    points_to_plot = points + offset.reshape(1, 3)

    # Use plotly to make scatter3d plot
    scatter = go.Scatter3d(
        x=points_to_plot[:, 0],
        y=points_to_plot[:, 1],
        z=points_to_plot[:, 2],
        mode="markers",
        marker=dict(size=5, color="blue"),
        name="Query Points",
    )

    return scatter

# %%
from pathlib import Path
from nerf_grasping import nerf_utils
from nerf import utils


def get_root_dir():
    root_dir = None
    try:
        root_dir = Path(os.path.abspath(__file__)).parents[0]
    except:
        pass
    try:
        root_dir = Path(os.path.abspath("."))
    except:
        pass
    if root_dir is None:
        raise ValueError("Can't get path to this file")
    return root_dir


root_dir = get_root_dir()
print(root_dir)


# %%
@localscope.mfc(allowed=["root_dir"])
def load_nerf(workspace, bound, scale):
    parser = utils.get_config_parser()
    opt = parser.parse_args(
        [
            "--workspace",
            f"{root_dir}/nerf_checkpoints/{workspace}",
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

@localscope.mfc
def get_nerf_densities(nerf_model, query_points):
    """
    Evaluates density of a batch of grasp points, shape [B, n_f, 3].
    query_points is torch.Tensor in nerf frame
    """
    B, n_f, _ = query_points.shape
    query_points = query_points.reshape(1, -1, 3)

    return nerf_model.density(query_points).reshape(B, n_f)

# %%
# Visualize points
@localscope.mfc
def get_colored_points_scatter(points, colors, offset=None):
    if offset is None:
        offset = np.zeros(3)

    assert len(points.shape) == 2 and points.shape[1] == 3

    points_to_plot = points + offset.reshape(1, 3)

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
@localscope.mfc
def get_query_points_mesh_region(min_points, max_points, n_pts_per_dim):
    """
    Returns a batch of query points in the mesh region.
    """
    x = np.linspace(min_points[0], max_points[0], n_pts_per_dim)
    y = np.linspace(min_points[1], max_points[1], n_pts_per_dim)
    z = np.linspace(min_points[2], max_points[2], n_pts_per_dim)
    xv, yv, zv = np.meshgrid(x, y, z)
    return np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)







# %% [markdown]
# # Load Data From Files

# %%
# Absolute paths
root_dir = "/juno/u/tylerlum/github_repos/nerf_grasping"
assets_dir_filepath = os.path.join(root_dir, "assets")
objects_dir_filepath = os.path.join(assets_dir_filepath, "objects")
acronym_dir_filepath = "/juno/u/tylerlum/github_repos/acronym/data/grasps"


for selected_obj in tqdm(objs):
    # TODO HACK
    if "Mug_10f" not in selected_obj.workspace:
        continue

    # Prepare filenames
    nerf_model_workspace = "isaac_" + selected_obj.workspace
    acronym_data_filepath = os.path.join(acronym_dir_filepath, selected_obj.acronym_file)
    urdf_filepath = os.path.join(
        assets_dir_filepath, selected_obj.asset_file
    )
    obj_filepath = os.path.join(
        objects_dir_filepath, selected_obj._get_mesh_path_from_urdf(urdf_filepath)
    )

    # Read acronym data
    acronym_data = h5py.File(acronym_data_filepath, "r")
    mesh_scale = float(acronym_data["object/scale"][()])

    grasp_transforms = np.array(acronym_data["grasps/transforms"])
    grasp_successes = np.array(acronym_data["grasps/qualities/flex/object_in_gripper"])
    assert(grasp_transforms.shape == (2000, 4, 4))
    assert(grasp_successes.shape == (2000,))
    
    # Get mesh info
    mesh = trimesh.load(obj_filepath, force="mesh")
    min_points_obj_frame, max_points_obj_frame = get_mesh_bounds(mesh, scale=mesh_scale)

    # Use this offset for all plots so that the plot is in isaac coordinates
    bound_min_z_obj_frame = min_points_obj_frame[2]
    mesh_centroid_obj_frame = get_mesh_centroid(mesh, scale=mesh_scale)

    # Compute offset
    obj_offset = np.array(
        [
            -mesh_centroid_obj_frame[0],
            -mesh_centroid_obj_frame[1],
            -bound_min_z_obj_frame,
        ]
    )
    
    # Load nerf
    print("Loading NeRF...")
    nerf_model = load_nerf(workspace=nerf_model_workspace, bound=2, scale=1)
    print("Done loading NeRF")

    # Make plot for each grasp
    num_grasps = grasp_transforms.shape[0]
    assert(num_grasps == grasp_successes.shape[0])

    for grasp_idx in range(grasp_transforms.shape[0]):
        # TODO REMOVE
        if grasp_idx < 3:
            continue

        # Create plot of mesh
        fig = plot_obj(obj_filepath, scale=mesh_scale, offset=obj_offset)
        mesh_centroid_scatter = get_mesh_centroid_scatter(
            mesh_centroid_obj_frame, offset=obj_offset
        )
        mesh_origin_lines = get_mesh_origin_lines(offset=obj_offset)
        fig.add_trace(mesh_centroid_scatter)
        for mesh_origin_line in mesh_origin_lines:
            fig.add_trace(mesh_origin_line)

        # Plot grasps
        USE_GRASP_RAY_LINES = True
        if USE_GRASP_RAY_LINES:
            grasp_lines = get_grasp_ray_lines(
                grasp_transforms[grasp_idx:grasp_idx+1], grasp_successes[grasp_idx:grasp_idx+1], offset=obj_offset
            )
        else:
            grasp_lines = get_grasp_gripper_lines(
                grasp_transforms[grasp_idx:grasp_idx+1], grasp_successes[grasp_idx:grasp_idx+1], offset=obj_offset
            )
        for grasp_line in grasp_lines:
            fig.add_trace(grasp_line)


        # Plot grasp query points
        grasp_query_points_grasp_frame = get_grasp_query_points_grasp_frame()

        grasp_query_points_object_frame = get_transformed_points(
            grasp_query_points_grasp_frame.reshape(-1, 3), grasp_transforms[grasp_idx]
        ).reshape(grasp_query_points_grasp_frame.shape)

        grasp_query_points_isaac_frame = np.copy(grasp_query_points_object_frame).reshape(
            -1, 3
        ) + obj_offset.reshape(1, 3)
        grasp_query_points_nerf_frame = ig_to_nerf(
            grasp_query_points_isaac_frame, return_tensor=True
        )

        nerf_densities_torch = get_nerf_densities(
            nerf_model=nerf_model,
            query_points=grasp_query_points_nerf_frame.reshape(1, -1, 3).float().cuda(),
        ).reshape(grasp_query_points_object_frame.shape[:-1])

        nerf_densities = nerf_densities_torch.detach().cpu().numpy()

        colored_points_scatter = get_colored_points_scatter(
            points=grasp_query_points_object_frame.reshape(-1, 3),
            colors=nerf_densities.reshape(-1),
            offset=obj_offset,
        )

        # Add the scatter plot to a figure and display it
        fig.add_trace(colored_points_scatter)

        # Avoid legend overlap
        fig.update_layout(legend_orientation="h")

        PLOT_ALL_HIGH_DENSITY_POINTS = True
        if PLOT_ALL_HIGH_DENSITY_POINTS:
            ## Only for fancy plot of whole nerf
            query_points_mesh_region_obj_frame = get_query_points_mesh_region(
                min_points_obj_frame, max_points_obj_frame, n_pts_per_dim=50
            )

            query_points_mesh_region_obj_frame.shape

            query_points_mesh_region_isaac_frame = np.copy(
                query_points_mesh_region_obj_frame
            ).reshape(-1, 3) + obj_offset.reshape(1, 3)
            query_points_mesh_region_nerf_frame = ig_to_nerf(
                query_points_mesh_region_isaac_frame.reshape(-1, 3), return_tensor=True
            )

            # Compute nerf densities
            nerf_densities_torch = get_nerf_densities(
                nerf_model, query_points_mesh_region_nerf_frame.reshape(1, -1, 3).float().cuda()
            ).reshape(query_points_mesh_region_nerf_frame.shape[:-1])
            nerf_densities = nerf_densities_torch.detach().cpu().numpy()

            points = query_points_mesh_region_obj_frame.reshape(-1, 3)
            densities = nerf_densities.reshape(-1)

            threshold = 100
            filtered_points = points[densities > threshold]
            filtered_densities = densities[densities > threshold]
            colored_points_scatter = get_colored_points_scatter(
                points=filtered_points, colors=filtered_densities, offset=obj_offset
            )

            # Add the scatter plot to a figure and display it
            fig.add_trace(colored_points_scatter)
            fig.update_layout(legend_orientation="h")

        fig.show()
        break
    break

# %%
position_to_transformed_positions(np.array([4.10000000e-02, -7.27595772e-12, 1.12169998e-01]),
                                  grasp_transforms[grasp_idx:grasp_idx+1])

# %%
obj_offset

# %% [markdown]
# # Create Dataset

# %% [markdown]
# # Visualize Data

# %% [markdown]
# # Create Model

# %% [markdown]
# # Run Training

# %% [markdown]
# # Run Evaluation

# %% [markdown]
# # Visualize Results

# %%
