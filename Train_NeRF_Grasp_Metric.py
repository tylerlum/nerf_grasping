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
nerf_model_workspace = (
    "torch-ngp/isaac_Mug_10f6e09036350e92b3f21f1137c3c347_0.0002682458/"
)
acronym_filepath = "/juno/u/tylerlum/github_repos/acronym/data/grasps/Mug_10f6e09036350e92b3f21f1137c3c347_0.0002682457830986903.h5"
assets_filepath = "/juno/u/tylerlum/github_repos/nerf_grasping/assets/objects"
urdf_filepath = os.path.join(assets_filepath, "urdf", "Mug_10f6e09036350e92b3f21f1137c3c347.urdf")
obj_filepath = os.path.join(assets_filepath, "meshes", "Mug", "10f6e09036350e92b3f21f1137c3c347.obj")

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
def plot_obj(filepath, scale=1.0):
    # Read in the OBJ file
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Extract the vertex coordinates and faces from the OBJ file
    vertices = []
    faces = []
    for line in lines:
        if line.startswith('v '):
            vertex = [float(i)*scale for i in line.split()[1:4]]
            vertices.append(vertex)
        elif line.startswith('f '):
            face = [int(i.split('/')[0])-1 for i in line.split()[1:4]]
            faces.append(face)

    # Convert the vertex coordinates and faces to numpy arrays
    vertices = np.array(vertices)
    faces = np.array(faces)

    # Create the mesh3d trace
    mesh = go.Mesh3d(
        x=vertices[:,0],
        y=vertices[:,1],
        z=vertices[:,2],
        i=faces[:,0],
        j=faces[:,1],
        k=faces[:,2],
        color='lightpink',
        opacity=0.5
    )

    # Create the layout
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        )
    )

    # Create the figure
    fig = go.Figure(data=[mesh], layout=layout)

    # Return the figure
    return fig


# %%
fig = plot_obj(obj_filepath, scale=0.1)
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
