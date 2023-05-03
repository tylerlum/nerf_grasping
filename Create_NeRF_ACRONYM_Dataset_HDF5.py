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
# ## Summary (April 23, 2023)
#
# The purpose of this script is to create a dataset for predicting the quality of grasps from NeRF representations of objects.
#
# ## Script Inputs
#
# ## Dataset
# ```
# {output_hdf5_filename} with hierarchical structure
# /
# ├── /nerf_grid_input [shape (N, 4, 80, 20, 30)]
# ├── /grasp_success [shape (N,)]
# ```
#
# where the first channel of `nerf_grid_input` is the NeRF density at this point
# and the remaining channels of `nerf_grid_input` are the x, y, and z coordinates of the point.
# with respect to the mesh centroid

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
import pickle


# %% [markdown]
# # Read in Objects

# %%
obj_class_names = [
    k
    for k in tqdm(
        list(acronym_objects.__dict__.keys()), desc="Getting Object Class Names"
    )
    if k.startswith("Obj_")
]

# %%
nerf_checkpoints_path = "nerf_checkpoints"

# %%
checkpoints = os.listdir(nerf_checkpoints_path)

# Should be like isaac_AccentTable_93f7c646dc5dcbebbf7421909bb3502_0.0061787559
print(f"Found {len(checkpoints)} checkpoints")
print(f"First 10 are {checkpoints[:10]}")

# %%


@localscope.mfc
def create_objs(obj_class_names, checkpoints):
    workspace_to_obj_class_name = {
        eval(f"acronym_objects.{obj_class_name}.workspace"): obj_class_name
        for obj_class_name in tqdm(obj_class_names, desc="Getting workspace to obj")
    }
    workspaces_set = set(workspace_to_obj_class_name.keys())

    objs = []
    for checkpoint in tqdm(
        checkpoints, desc="Getting objects associated with checkpoints"
    ):
        if not checkpoint.startswith("isaac_"):
            print(f"Skipping {checkpoint} because it doesn't start with isaac_")
            continue

        # isaac_AccentTable_93f7c646dc5dcbebbf7421909bb3502_0.0061787559 => AccentTable_93f7c646dc5dcbebbf7421909bb3502_0.0061787559
        workspace = checkpoint.split("isaac_", maxsplit=1)[1]
        if workspace in workspaces_set:
            obj_class_name = workspace_to_obj_class_name[workspace]
            objs.append(eval(f"acronym_objects.{obj_class_name}()"))
    return objs


# %%
@localscope.mfc
def create_objs_2(obj_class_names, checkpoints):
    checkpoints_set = set(checkpoints)

    # Get objs with checkpoint, hacky
    # Simpler than the other way, but tqdm bar doesn't make sense
    objs = []
    for obj_class_name in tqdm(obj_class_names):
        workspace = "isaac_" + eval(f"acronym_objects.{obj_class_name}.workspace")
        if workspace in checkpoints_set:
            objs.append(eval(f"acronym_objects.{obj_class_name}()"))
    return objs


# %%
# HACK
objs = [acronym_objects.Obj_Plant_ed25fff42d9880dcfe572c9a3e59d718_0_005260()]
# objs = create_objs(obj_class_names, checkpoints)

# %%
print(f"Found {len(objs)} objs")
print(f"First 10 are {objs[:10]}")

# %%
found_invalid_workspace = False
for obj in objs:
    workspace_path = os.path.join(nerf_checkpoints_path, "isaac_" + obj.workspace)
    checkpoints_path = os.path.join(workspace_path, "checkpoints")
    if (
        not os.path.exists(workspace_path)
        or not os.path.exists(checkpoints_path)
        or len(os.listdir(checkpoints_path)) == 0
    ):
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
        width=800,
        height=800,
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

    x_line_np = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    y_line_np = np.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]])
    z_line_np = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]])

    lines = []
    for line_np, name, color in [
        (x_line_np, "X", "red"),
        (y_line_np, "Y", "green"),
        (z_line_np, "Z", "blue"),
    ]:
        line_np += offset
        lines.append(
            go.Scatter3d(
                x=line_np[:, 0],
                y=line_np[:, 1],
                z=line_np[:, 2],
                mode="lines",
                line=dict(width=2, color=color),
                name=f"Mesh Origin {name} Axis",
            )
        )
    return lines


# %%
@localscope.mfc(
    allowed=[
        "LEFT_TIP_POSITION_GRASP_FRAME",
        "RIGHT_TIP_POSITION_GRASP_FRAME",
    ]
)
def get_grasp_gripper_lines(grasp_transforms, grasp_successes, offset=None):
    if offset is None:
        offset = np.zeros(3)

    raw_left_tip = LEFT_TIP_POSITION_GRASP_FRAME
    raw_right_tip = RIGHT_TIP_POSITION_GRASP_FRAME
    raw_left_knuckle = LEFT_TIP_POSITION_GRASP_FRAME - np.array([0.0, 0.0, 0.04617])
    raw_right_knuckle = RIGHT_TIP_POSITION_GRASP_FRAME - np.array([0.0, 0.0, 0.04617])
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

        points += offset.reshape(1, 3)

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
    allowed=[
        "LEFT_TIP_POSITION_GRASP_FRAME",
        "RIGHT_TIP_POSITION_GRASP_FRAME",
    ]
)
def get_grasp_ray_lines(grasp_transforms, grasp_successes, offset=None):
    if offset is None:
        offset = np.zeros(3)

    raw_left_tip = LEFT_TIP_POSITION_GRASP_FRAME
    raw_right_tip = RIGHT_TIP_POSITION_GRASP_FRAME
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
# Grid of points in grasp frame (x, y, z)
GRIPPER_WIDTH_MM = 82
GRIPPER_FINGER_WIDTH_MM = 20
GRIPPER_FINGER_HEIGHT_MM = 36

# Want points equally spread out in space
DIST_BTWN_PTS_MM = 1

# +1 to include both end points
NUM_PTS_X = int(GRIPPER_WIDTH_MM / DIST_BTWN_PTS_MM) + 1
NUM_PTS_Y = int(GRIPPER_FINGER_WIDTH_MM / DIST_BTWN_PTS_MM) + 1
NUM_PTS_Z = int(GRIPPER_FINGER_HEIGHT_MM / DIST_BTWN_PTS_MM) + 1

assert (NUM_PTS_X - 1) * DIST_BTWN_PTS_MM == GRIPPER_WIDTH_MM
assert (NUM_PTS_Y - 1) * DIST_BTWN_PTS_MM == GRIPPER_FINGER_WIDTH_MM
assert (NUM_PTS_Z - 1) * DIST_BTWN_PTS_MM == GRIPPER_FINGER_HEIGHT_MM


@localscope.mfc(
    allowed=[
        "LEFT_TIP_POSITION_GRASP_FRAME",
        "RIGHT_TIP_POSITION_GRASP_FRAME",
        "NUM_PTS_X",
        "NUM_PTS_Y",
        "NUM_PTS_Z",
        "GRIPPER_WIDTH_MM",
        "GRIPPER_FINGER_WIDTH_MM",
        "GRIPPER_FINGER_HEIGHT_MM",
    ]
)
def get_grasp_query_points_grasp_frame():
    num_pts = NUM_PTS_X * NUM_PTS_Y * NUM_PTS_Z
    print(f"num_pts: {num_pts}")

    GRIPPER_WIDTH_M = GRIPPER_WIDTH_MM / 1000.0
    GRIPPER_FINGER_WIDTH_M = GRIPPER_FINGER_WIDTH_MM / 1000.0
    GRIPPER_FINGER_HEIGHT_M = GRIPPER_FINGER_HEIGHT_MM / 1000.0

    # Create grid of points in grasp frame with shape (NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z, 3)
    # So that grid_of_points[2, 3, 5] = [x, y, z], where x, y, z are the coordinates of the point
    x_coords = np.linspace(-GRIPPER_WIDTH_M / 2, GRIPPER_WIDTH_M / 2, NUM_PTS_X)
    y_coords = np.linspace(
        -GRIPPER_FINGER_WIDTH_M / 2, GRIPPER_FINGER_WIDTH_M / 2, NUM_PTS_Y
    )
    z_coords = np.linspace(
        -GRIPPER_FINGER_HEIGHT_M / 2, GRIPPER_FINGER_HEIGHT_M / 2, NUM_PTS_Z
    )

    # Offset so centered between LEFT_TIP_POSITION_GRASP_FRAME and RIGHT_TIP_POSITION_GRASP_FRAME
    center_point = (LEFT_TIP_POSITION_GRASP_FRAME + RIGHT_TIP_POSITION_GRASP_FRAME) / 2
    x_coords += center_point[0]
    y_coords += center_point[1]
    z_coords += center_point[2]

    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
    assert xx.shape == yy.shape == zz.shape == (NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z)
    grid_of_points = np.stack([xx, yy, zz], axis=-1)
    assert grid_of_points.shape == (NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z, 3)
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


# %%
class Timer:
    def __init__(self, name_of_timed_commands, get_current_time_fn=time.perf_counter):
        self.name = name_of_timed_commands
        self.get_current_time_fn = get_current_time_fn

    def __enter__(self):
        # self.start = self.get_current_time_fn()
        return self

    def __exit__(self, type, value, traceback):
        return
        # print(
        #     f"Time elapsed for '{self.name}' is {self.get_current_time_fn() - self.start} seconds"
        # )


# %% [markdown]
# # Load Data From Files

# %%
# Same query points for each grasp in grasp frame
grasp_query_points_grasp_frame = get_grasp_query_points_grasp_frame()
assert grasp_query_points_grasp_frame.shape == (NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z, 3)

# %%
# Absolute paths
root_dir = "/juno/u/tylerlum/github_repos/nerf_grasping"
assets_dir_filepath = os.path.join(root_dir, "assets")
objects_dir_filepath = os.path.join(assets_dir_filepath, "objects")
acronym_dir_filepath = "/juno/u/tylerlum/github_repos/acronym/data/grasps"


CREATE_PLOTS = False
SAVE_DATASET = True

LIMIT_NUM_OBJECTS = False
if LIMIT_NUM_OBJECTS:
    num_objects = 3
    print(f"Limiting number of objects to {num_objects}")
    objs = objs[:num_objects]

output_hdf5_filename = os.path.join(
    root_dir, f"nerf_acronym_grasp_success_dataset_{len(objs)}_categories.h5"
)

ACRONYM_NUM_GRASPS_PER_OBJ = 2000
max_num_data_points = ACRONYM_NUM_GRASPS_PER_OBJ * len(objs)  # Simple heuristic

if os.path.exists(output_hdf5_filename):
    print(f"Found {output_hdf5_filename}, exiting...")
    exit()
    # print(f"Found {output_hdf5_filename}, removing...")
    # os.remove(output_hdf5_filename)
    # print("Done removing")

with h5py.File(output_hdf5_filename, "w") as hdf5_file:
    current_idx = 0

    # TODO: May not want to keep all coordinate information
    nerf_grid_input_dataset = hdf5_file.create_dataset(
        "/nerf_grid_input",
        shape=(max_num_data_points, 4, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z),
        dtype="f",
        chunks=(1, 4, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z),
    )
    grasp_success_dataset = hdf5_file.create_dataset(
        "/grasp_success", shape=(max_num_data_points,), dtype="i"
    )
    acronym_filename_dataset = hdf5_file.create_dataset(
        "/acronym_filename", shape=(max_num_data_points,), dtype=h5py.string_dtype()
    )
    grasp_idxs_dataset = hdf5_file.create_dataset(
        "/grasp_idx", shape=(max_num_data_points,), dtype="i"
    )

    for selected_obj in (pbar := tqdm(objs)):
        pbar.set_description(f"{selected_obj.workspace}")

        # Prepare filename
        nerf_model_workspace = "isaac_" + selected_obj.workspace
        acronym_data_filepath = os.path.join(
            acronym_dir_filepath, selected_obj.acronym_file
        )
        urdf_filepath = os.path.join(assets_dir_filepath, selected_obj.asset_file)
        obj_filepath = os.path.join(
            objects_dir_filepath, selected_obj._get_mesh_path_from_urdf(urdf_filepath)
        )

        # Read acronym data
        with Timer("Read acronym data file"):
            acronym_data = h5py.File(acronym_data_filepath, "r")

        with Timer("Access acronym data"):
            mesh_scale = float(acronym_data["object/scale"][()])
            grasp_transforms = np.array(acronym_data["grasps/transforms"])
            grasp_successes = np.array(
                acronym_data["grasps/qualities/flex/object_in_gripper"]
            )

        assert grasp_transforms.shape == (2000, 4, 4)
        assert grasp_successes.shape == (2000,)

        # Get mesh info
        with Timer("Load mesh"):
            mesh = trimesh.load(obj_filepath, force="mesh")

        with Timer("Get mesh info"):
            min_points_obj_frame, max_points_obj_frame = get_mesh_bounds(
                mesh, scale=mesh_scale
            )

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
        with Timer("Load NeRF"):
            nerf_model = load_nerf(workspace=nerf_model_workspace, bound=2, scale=1)
        print("Done loading NeRF")

        # Make plot for each grasp
        num_grasps = grasp_transforms.shape[0]
        assert num_grasps == grasp_successes.shape[0]

        for grasp_idx in tqdm(range(num_grasps)):
            # Get grasp query points
            with Timer("Get grasp query points obj frame"):
                grasp_query_points_object_frame = get_transformed_points(
                    grasp_query_points_grasp_frame.reshape(-1, 3),
                    grasp_transforms[grasp_idx],
                )

            with Timer("Get grasp query points nerf frame"):
                grasp_query_points_isaac_frame = (
                    grasp_query_points_object_frame + obj_offset.reshape(1, 3)
                )
                grasp_query_points_nerf_frame = ig_to_nerf(
                    grasp_query_points_isaac_frame, return_tensor=True
                )

            with Timer("Get grasp query densities"):
                import ipdb; ipdb.set_trace()
                grasp_query_nerf_densities_torch = get_nerf_densities(
                    nerf_model=nerf_model,
                    query_points=grasp_query_points_nerf_frame.reshape(1, -1, 3)
                    .float()
                    .cuda(),
                )

            with Timer("Convert grasp query densities to numpy"):
                grasp_query_nerf_densities = (
                    grasp_query_nerf_densities_torch.detach().cpu().numpy()
                )

            with Timer("Reshape nerf densities"):
                grasp_query_nerf_densities = grasp_query_nerf_densities.reshape(
                    NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z
                )

            # Save dataset
            if SAVE_DATASET:
                mesh_centroid_isaac_frame = mesh_centroid_obj_frame + obj_offset
                grasp_query_points_wrt_centroid = (
                    grasp_query_points_isaac_frame
                    - mesh_centroid_isaac_frame.reshape(1, 3)
                )

                # Merge together grasp_query_points_wrt_centroid and grasp_query_nerf_densities
                # So goes from (83, 21, 37, 3) and (83, 21, 37) to (83, 21, 37, 4)
                nerf_grid_input = np.concatenate(
                    [
                        grasp_query_points_wrt_centroid.reshape(
                            NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z, 3
                        ),
                        grasp_query_nerf_densities.reshape(
                            NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z, 1
                        ),
                    ],
                    axis=-1,
                )

                # Switch from (83, 21, 37, 4) to (4, 83, 21, 37)
                nerf_grid_input = np.transpose(nerf_grid_input, (3, 0, 1, 2))
                assert nerf_grid_input.shape == (4, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z)

                if np.isnan(nerf_grid_input).any():
                    # breakpoint()
                    import ipdb; ipdb.set_trace()
                    print(f"Skipping grasp {grasp_idx} because it has NaNs")

                nerf_grid_input_dataset[current_idx] = nerf_grid_input
                grasp_success_dataset[current_idx] = grasp_successes[grasp_idx]
                acronym_filename_dataset[current_idx] = selected_obj.acronym_file
                grasp_idxs_dataset[current_idx] = grasp_idx
                current_idx += 1

            # Create plot of mesh
            if CREATE_PLOTS:
                fig = plot_obj(obj_filepath, scale=mesh_scale, offset=obj_offset)
                mesh_centroid_scatter = get_mesh_centroid_scatter(
                    mesh_centroid_obj_frame, offset=obj_offset
                )
                fig.add_trace(mesh_centroid_scatter)
                mesh_origin_lines = get_mesh_origin_lines(offset=obj_offset)
                for mesh_origin_line in mesh_origin_lines:
                    fig.add_trace(mesh_origin_line)

                # Plot grasps
                USE_GRASP_RAY_LINES = True
                if USE_GRASP_RAY_LINES:
                    grasp_lines = get_grasp_ray_lines(
                        grasp_transforms[grasp_idx : grasp_idx + 1],
                        grasp_successes[grasp_idx : grasp_idx + 1],
                        offset=obj_offset,
                    )
                else:
                    grasp_lines = get_grasp_gripper_lines(
                        grasp_transforms[grasp_idx : grasp_idx + 1],
                        grasp_successes[grasp_idx : grasp_idx + 1],
                        offset=obj_offset,
                    )
                for grasp_line in grasp_lines:
                    fig.add_trace(grasp_line)

                colored_points_scatter = get_colored_points_scatter(
                    points=grasp_query_points_object_frame.reshape(-1, 3),
                    colors=grasp_query_nerf_densities.reshape(-1),
                    offset=obj_offset,
                )

                # Add the scatter plot to a figure and display it
                fig.add_trace(colored_points_scatter)

                # Avoid legend overlap
                fig.update_layout(legend_orientation="h")

                PLOT_ALL_HIGH_DENSITY_POINTS = False
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
                        query_points_mesh_region_isaac_frame.reshape(-1, 3),
                        return_tensor=True,
                    )

                    # Compute nerf densities
                    query_nerf_densities_torch = get_nerf_densities(
                        nerf_model,
                        query_points_mesh_region_nerf_frame.reshape(1, -1, 3)
                        .float()
                        .cuda(),
                    ).reshape(query_points_mesh_region_nerf_frame.shape[:-1])
                    query_nerf_densities = (
                        query_nerf_densities_torch.detach().cpu().numpy()
                    )

                    points = query_points_mesh_region_obj_frame.reshape(-1, 3)
                    densities = query_nerf_densities.reshape(-1)

                    threshold = 100
                    filtered_points = points[densities > threshold]
                    filtered_densities = densities[densities > threshold]
                    colored_points_scatter = get_colored_points_scatter(
                        points=filtered_points,
                        colors=filtered_densities,
                        offset=obj_offset,
                    )

                    # Add the scatter plot to a figure and display it
                    fig.add_trace(colored_points_scatter)
                    fig.update_layout(legend_orientation="h")

                fig.show()

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
