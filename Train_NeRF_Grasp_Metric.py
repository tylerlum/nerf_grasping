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
# # Train NeRF Grasp Metric
#
# ## Summary (April 18, 2023)
#
# The purpose of this script is to train a neural network model to take in:
#
# * a NeRF object model
#
# * $n$ ray origins and directions representating fingers approaching (for now, $n = 2$)
#
# and output:
#
# * a grasp metric $g$ representing the quality of grasp (for now, $g \in [0, 1]$, where 0 is failed grasp and 1 is successful grasp).
#
# To do this, we will be using the [ACRONYM dataset](https://sites.google.com/nvidia.com/graspdataset), which contains ~1.7M grasps on over 8k objects each labeled with the grasp success.

# %%
import wandb
import os
import h5py
import numpy as np
from localscope import localscope
import time

import random
import torch
import plotly.graph_objects as go
import trimesh


# %% [markdown]
# # Read In Config


# %%
def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


# %%
# Use with hydra later
# if is_notebook():
#     arguments = []
# else:
#     arguments = sys.argv[1:]
#     print(f"arguments = {arguments}")
#
# OmegaConf.register_new_resolver("eval", eval)
# with initialize(version_base=None, config_path="train_bc_config_files"):
#     cfg = compose(config_name="config", overrides=arguments)
#     print(OmegaConf.to_yaml(cfg))

# %%

# %% [markdown]
# # Setup Wandb

# %%

# time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# run_name = f"{cfg.wandb_name}_{time_str}" if len(cfg.wandb_name) > 0 else time_str

# wandb.init(entity=cfg.wandb_entity,
#            project=cfg.wandb_project,
#            name=run_name,
#            group=cfg.wandb_group if len(cfg.wandb_group) > 0 else None,
#            job_type=cfg.wandb_job_type if len(cfg.wandb_job_type) > 0 else None,
#            config=OmegaConf.to_container(cfg),
#            reinit=True)

# %%


@localscope.mfc
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


# set_seed(cfg.RANDOM_SEED)

# %% [markdown]
# # Load Data From Files

# %%
# TODO: Need way to connect an acronym file to a nerf model nicely
nerf_model_workspace = "isaac_Mug_10f6e09036350e92b3f21f1137c3c347_0.0002682458/"
acronym_filepath = "/juno/u/tylerlum/github_repos/acronym/data/grasps/Mug_10f6e09036350e92b3f21f1137c3c347_0.0002682457830986903.h5"
assets_filepath = "/juno/u/tylerlum/github_repos/nerf_grasping/assets/objects"
urdf_filepath = os.path.join(
    assets_filepath, "urdf", "Mug_10f6e09036350e92b3f21f1137c3c347.urdf"
)
obj_filepath = os.path.join(
    assets_filepath, "meshes", "Mug", "10f6e09036350e92b3f21f1137c3c347.obj"
)

# %%
acronym_data = h5py.File(acronym_filepath, "r")
mesh_scale = float(acronym_data["object/scale"][()])

grasp_transforms = np.array(acronym_data["grasps/transforms"])
grasp_successes = np.array(acronym_data["grasps/qualities/flex/object_in_gripper"])

# %%
print(f"{grasp_transforms.shape = }")
print(f"{grasp_successes.shape = }")

# %%
LEFT_TIP_POSITION_GRASP_FRAME = np.array(
    [4.10000000e-02, -7.27595772e-12, 1.12169998e-01]
)
RIGHT_TIP_POSITION_GRASP_FRAME = np.array(
    [-4.10000000e-02, -7.27595772e-12, 1.12169998e-01]
)


# %%
@localscope.mfc
def position_to_transformed_positions(position, transforms):
    assert position.shape == (3,)
    assert len(transforms.shape) == 3 and transforms.shape[1:] == (4, 4)
    num_transforms = transforms.shape[0]

    transformed_positions = (transforms @ np.array([*position, 1.0]).reshape(1, 4, 1))[
        :, :3, :
    ].squeeze()
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
run_sanity_check(position=LEFT_TIP_POSITION_GRASP_FRAME, transforms=grasp_transforms)


# %%
@localscope.mfc
def plot_obj(obj_filepath, scale=1.0):
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

    # Create the mesh3d trace
    mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color="lightpink",
        opacity=0.5,
        name=f"Mesh: {os.path.basename(obj_filepath)}",
    )

    # Create the layout
    layout = go.Layout(
        scene=dict(xaxis=dict(title="X"), yaxis=dict(title="Y"), zaxis=dict(title="Z")),
        showlegend=True,
    )

    # Create the figure
    fig = go.Figure(data=[mesh], layout=layout)

    # Return the figure
    return fig


# %%
fig = plot_obj(obj_filepath, scale=mesh_scale)
fig.show()


# %%
mesh = trimesh.load(obj_filepath, force="mesh")

@localscope.mfc
def get_mesh_centroid_scatter(mesh, scale=1.0):
    mesh_centroid = np.array(mesh.centroid) * scale
    scatter = go.Scatter3d(
        x=[mesh_centroid[0]],
        y=[mesh_centroid[1]],
        z=[mesh_centroid[2]],
        mode="markers",
        marker=dict(size=10, color="black"),
        name="Mesh Centroid",
    )
    return scatter


@localscope.mfc
def get_mesh_origin_lines():
    lines = [
        go.Scatter3d(
            x=[0.0, 0.1],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
            mode="lines",
            line=dict(width=2, color="red"),
            name="Mesh Origin X Axis",
        ),
        go.Scatter3d(
            x=[0.0, 0.0],
            y=[0.0, 0.1],
            z=[0.0, 0.0],
            mode="lines",
            line=dict(width=2, color="green"),
            name="Mesh Origin Y Axis",
        ),
        go.Scatter3d(
            x=[0.0, 0.0],
            y=[0.0, 0.0],
            z=[0.0, 0.1],
            mode="lines",
            line=dict(width=2, color="blue"),
            name="Mesh Origin Z Axis",
        ),
    ]
    return lines


# %%
fig = plot_obj(obj_filepath, scale=mesh_scale)
mesh_centroid_scatter = get_mesh_centroid_scatter(mesh, scale=mesh_scale)
mesh_origin_lines = get_mesh_origin_lines()
fig.add_trace(mesh_centroid_scatter)
for mesh_origin_line in mesh_origin_lines:
    fig.add_trace(mesh_origin_line)
fig.show()


# %%
@localscope.mfc
def get_grasp_gripper_lines(grasp_transforms, grasp_successes):
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
fig = plot_obj(obj_filepath, scale=mesh_scale)
grasp_lines = get_grasp_gripper_lines(grasp_transforms[:6], grasp_successes[:6])
for grasp_line in grasp_lines:
    fig.add_trace(grasp_line)
fig.show()


# %%
@localscope.mfc
def get_grasp_ray_lines(grasp_transforms, grasp_successes):
    raw_left_tip = np.array([4.10000000e-02, -7.27595772e-12, 1.12169998e-01])
    raw_right_tip = np.array([-4.10000000e-02, -7.27595772e-12, 1.12169998e-01])
    left_tips = position_to_transformed_positions(
        position=raw_left_tip, transforms=grasp_transforms
    )
    right_tips = position_to_transformed_positions(
        position=raw_right_tip, transforms=grasp_transforms
    )

    assert left_tips.shape == right_tips.shape == (len(grasp_successes), 3)

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
fig = plot_obj(obj_filepath, scale=mesh_scale)
grasp_lines = get_grasp_ray_lines(grasp_transforms[:6], grasp_successes[:6])
for grasp_line in grasp_lines:
    fig.add_trace(grasp_line)
fig.show()

# %% [markdown]
# ## TEMP: Visualize NeRF

# %%
@localscope.mfc(allowed=["LEFT_TIP_POSITION_GRASP_FRAME", "RIGHT_TIP_POSITION_GRASP_FRAME"])
def get_grasp_query_points_grasp_frame():
    # Grid of points in grasp frame (x, y, z)
    GRIPPER_WIDTH_MM = 82.0
    GRIPPER_FINGER_WIDTH_MM = 20.0
    GRIPPER_FINGER_HEIGHT_MM = 35.0

    # Want points equally spread out in space
    DIST_BTWN_PTS_MM = 1.0

    GRIPPER_WIDTH_M = GRIPPER_WIDTH_MM / 1000.0
    GRIPPER_FINGER_WIDTH_M = GRIPPER_FINGER_WIDTH_MM / 1000.0
    GRIPPER_FINGER_HEIGHT_M = GRIPPER_FINGER_HEIGHT_MM / 1000.0
    DIST_BTWN_PTS_M = DIST_BTWN_PTS_MM / 1000.0

    NUM_PTS_X = int(GRIPPER_WIDTH_M / DIST_BTWN_PTS_M)
    NUM_PTS_Y = int(GRIPPER_FINGER_WIDTH_M / DIST_BTWN_PTS_M)
    NUM_PTS_Z = int(GRIPPER_FINGER_HEIGHT_M / DIST_BTWN_PTS_M)

    assert NUM_PTS_X * DIST_BTWN_PTS_M == GRIPPER_WIDTH_M
    assert NUM_PTS_Y * DIST_BTWN_PTS_M == GRIPPER_FINGER_WIDTH_M
    assert NUM_PTS_Z * DIST_BTWN_PTS_M == GRIPPER_FINGER_HEIGHT_M
    num_pts = (NUM_PTS_X + 1) * (NUM_PTS_Y + 1) * (NUM_PTS_Z + 1)
    print(f"num_pts: {num_pts}")

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


grasp_query_points_grasp_frame = get_grasp_query_points_grasp_frame()


# %%
# Transform query points to object frame

@localscope.mfc
def get_trasformed_points(points, transform):
    assert len(points.shape) == 2 and points.shape[1] == 3
    assert transform.shape == (4, 4)

    points_homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)

    # First (4, 4) @ (4, N) = (4, N)
    # Then transpose to get (N, 4)
    transformed_points = np.matmul(transform, points_homogeneous.T).T

    return transformed_points[:, :3]


transform = grasp_transforms[4]
grasp_query_points_object_frame = get_trasformed_points(grasp_query_points_grasp_frame.reshape(-1, 3), transform).reshape(grasp_query_points_grasp_frame.shape)

# %%
# Visualize points
@localscope.mfc
def get_points_scatter(points):
    assert len(points.shape) == 2 and points.shape[1] == 3

    # Use plotly to make scatter3d plot
    scatter = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(size=5, color="blue"),
        name="Query Points"
    )

    return scatter

# %%
scatter = get_points_scatter(grasp_query_points_object_frame.reshape(-1, 3))

# Add the scatter plot to a figure and display it
fig = plot_obj(obj_filepath, scale=mesh_scale)
fig.add_trace(scatter)
fig.show()

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


nerf_model = load_nerf(workspace=nerf_model_workspace, bound=2, scale=1)


# %%
nerf_model


# %%
@localscope.mfc
def get_nerf_densities(nerf_model, query_points):
    """
    Evaluates density of a batch of grasp points, shape [B, n_f, 3].
    """
    B, n_f, _ = query_points.shape
    query_points = query_points.reshape(1, -1, 3)

    return nerf_model.density(query_points).reshape(B, n_f)

nerf_densities_torch = get_nerf_densities(nerf_model, torch.from_numpy(grasp_query_points_object_frame).reshape(1, -1, 3).float().cuda()).reshape(grasp_query_points_object_frame.shape[:-1])

# %%
nerf_densities = nerf_densities_torch.detach().cpu().numpy()

print(f"np.max(nerf_densities) = {np.max(nerf_densities)}")
print(f"np.min(nerf_densities) = {np.min(nerf_densities)}")


# %%
# Visualize points
@localscope.mfc
def get_colored_points_scatter(points, colors):
    assert len(points.shape) == 2 and points.shape[1] == 3

    # Use plotly to make scatter3d plot
    scatter = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(size=5, color=colors, colorscale='viridis', colorbar=dict(title="Density Scale"),),
        name="Query Points Densities",
    )

    return scatter


# %%
colored_points_scatter = get_colored_points_scatter(points=grasp_query_points_object_frame.reshape(-1, 3),
                                                   colors=nerf_densities.reshape(-1))

# Add the scatter plot to a figure and display it
fig = plot_obj(obj_filepath, scale=mesh_scale)
fig.add_trace(colored_points_scatter)
# fig.update_layout(width=800, height=600, coloraxis_colorbar=dict(len=0.2, yanchor='middle', y=0.1))
# fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=0,
#                                           ticks="outside",
#                                           ticksuffix=" bills"))
fig.update_layout(legend_orientation="h")

fig.show()


# %%
# Get bounds of mesh
@localscope.mfc
def get_mesh_bounds(mesh, scale=1):
    min_points, max_points = mesh.bounds
    return min_points * scale, max_points * scale

min_points, max_points = get_mesh_bounds(mesh, scale=mesh_scale)
print(min_points, max_points)



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
