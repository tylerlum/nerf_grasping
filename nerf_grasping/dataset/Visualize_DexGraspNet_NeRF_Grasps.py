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
import nerf_grasping
from localscope import localscope
from nerf import utils
import torch
import os
import trimesh
import numpy as np
from plotly import graph_objects as go
from dataclasses import dataclass

# %%
# PARAMS
dexgraspnet_data_root = "/juno/u/tylerlum/github_repos/DexGraspNet/data"
dexgraspnet_meshdata_root = os.path.join(dexgraspnet_data_root, "meshdata")
mesh_path = os.path.join(
    dexgraspnet_meshdata_root, "ddg-gd_banana_poisson_001", "coacd", "decomposed.obj"
)
nerf_checkpoint_folder = "2023-07-21_nerf_checkpoints"
nerf_model_workspace = "ddg-gd_banana_poisson_001_0_06"
nerf_size_scale = 0.06
nerf_bound = 2.0
nerf_scale = 1.0

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
def get_bounds(mesh: trimesh.Trimesh):
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
        autosize=False,
        width=200,
        height=200,
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
    )

    # Create the figure
    fig = go.Figure(data=[mesh_plot], layout=layout)

    # Return the figure
    return fig


# %%
fig = plot_mesh(mesh)
fig.show()

# %%


@localscope.mfc()
def load_nerf(workspace: str, bound: float, scale: float):
    root_dir = nerf_grasping.get_repo_root()

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


# %%
nerf_model = load_nerf(
    workspace=nerf_model_workspace, bound=nerf_bound, scale=nerf_scale
)

# %%
dir(nerf_grasping), nerf_grasping.__file__

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
