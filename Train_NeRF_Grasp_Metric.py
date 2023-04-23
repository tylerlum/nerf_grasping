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

# %%

# %% [markdown]
# # Imports

import os
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import torch
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from localscope import localscope
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, random_split

import wandb

# %%
# # Notebook Setup

# %%
OmegaConf.register_new_resolver("eval", eval, replace=True)


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


# %% [markdown]
# # Setup Config for Static Type-Checking


# %%
@dataclass
class WandbConfig:
    entity: str
    project: str
    name: str
    group: str
    job_type: str


@dataclass
class DataConfig:
    frac_val: float
    frac_test: float
    # frac_train: float

    input_dataset_dir: str


@dataclass
class Config:
    wandb: WandbConfig
    data: DataConfig


# %%
config_store = ConfigStore.instance()
config_store.store(name="config", node=Config)


# %% [markdown]
# # Load Config

# %%
if is_notebook():
    arguments = []
else:
    arguments = sys.argv[1:]
    print(f"arguments = {arguments}")


# %%
with initialize(version_base="1.1", config_path="Train_NeRF_Grasp_Metric_cfg"):
    raw_cfg = compose(config_name="config", overrides=arguments, strict=True)

cfg = instantiate(raw_cfg)

# %%
cfg.data

# %%
type(cfg.data)

# %%
type(cfg.wandb)

# %%
OmegaConf.to_container(cfg, throw_on_missing=True)

# %% [markdown]
# # Setup Workspace and Wandb Logging

# %%
time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"{cfg.wandb.name}_{time_str}" if len(cfg.wandb.name) > 0 else time_str

# %%
# TODO: Make workspace like in NeRF training to be able to resume
# workspace_root_dir = "Train_NeRF_Grasp_Metric_workspaces"
# if not os.path.exists(workspace_root_dir):
#     os.makedirs(workspace_root_dir)

# %%
wandb.init(
    entity=cfg.wandb.entity,
    project=cfg.wandb.project,
    name=run_name,
    group=cfg.wandb.group if len(cfg.wandb.group) > 0 else None,
    job_type=cfg.wandb.job_type if len(cfg.wandb.job_type) > 0 else None,
    config=OmegaConf.to_container(cfg, throw_on_missing=True),
    reinit=True,
)

# %% [markdown]
# # Load In Data

# %%
# CONSTANTS AND PARAMS
ROOT_DIR = "/juno/u/tylerlum/github_repos/nerf_grasping"
RANDOM_SEED = 42
BATCH_SIZE = 32
DATA_LOADER_NUM_WORKERS = 4


# %%
class NeRFGrid_To_GraspSuccess_Dataset(Dataset):
    @localscope.mfc
    def __init__(self, input_dataset_dir):
        self.input_dataset_dir = input_dataset_dir
        self.filepaths = [
            os.path.join(self.input_dataset_dir, object_dir, filepath)
            for object_dir in os.listdir(self.input_dataset_dir)
            for filepath in os.listdir(os.path.join(self.input_dataset_dir, object_dir))
            if filepath.endswith(".pkl")
        ]

    @localscope.mfc
    def __len__(self):
        return len(self.filepaths)

    @localscope.mfc
    def __getitem__(self, idx):
        # Read pickle ifle
        with open(self.filepaths[idx], "rb") as f:
            data_dict = pickle.load(f)

        nerf_grid_input = data_dict["nerf_grid_input"]
        grasp_success = np.array([data_dict["grasp_success"]])
        assert nerf_grid_input.shape == (4, 83, 21, 37)
        assert grasp_success.shape == (1,)

        return data_dict["nerf_grid_input"], data_dict["grasp_success"]


# %%
full_dataset = NeRFGrid_To_GraspSuccess_Dataset(
    os.path.join(ROOT_DIR, cfg.data.input_dataset_dir)
)
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [cfg.data.frac_train, cfg.data.frac_val, cfg.data.frac_test],
    generator=torch.Generator().manual_seed(RANDOM_SEED),
)

# %%
print(f"Train dataset size: {len(train_dataset)}")
print(f"Val dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# %%
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=DATA_LOADER_NUM_WORKERS,
)
val_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=DATA_LOADER_NUM_WORKERS,
)
test_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=DATA_LOADER_NUM_WORKERS,
)


# %%
@localscope.mfc
def get_isaac_origin_lines():
    x_line_np = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    y_line_np = np.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]])
    z_line_np = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]])

    lines = []
    for line_np, name, color in [
        (x_line_np, "X", "red"),
        (y_line_np, "Y", "green"),
        (z_line_np, "Z", "blue"),
    ]:
        lines.append(
            go.Scatter3d(
                x=line_np[:, 0],
                y=line_np[:, 1],
                z=line_np[:, 2],
                mode="lines",
                line=dict(width=2, color=color),
                name=f"Isaac Origin {name} Axis",
            )
        )
    return lines


@localscope.mfc
def get_colored_points_scatter(points, colors):
    assert len(points.shape) == 2 and points.shape[1] == 3
    assert len(colors.shape) == 1

    # Use plotly to make scatter3d plot
    scatter = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(
            size=5,
            color=colors,
            colorscale="viridis",
            colorbar=dict(title="Density Scale"),
        ),
        name="Query Point Densities",
    )

    return scatter


# %%
idx_to_visualize = 0
for nerf_grid_input, grasp_success in train_loader:
    assert nerf_grid_input.shape == (BATCH_SIZE, 4, 83, 21, 37)
    assert grasp_success.shape == (BATCH_SIZE,)

    nerf_densities = nerf_grid_input[idx_to_visualize, -1, :, :, :]
    assert nerf_densities.shape == (83, 21, 37)
    nerf_points = nerf_grid_input[idx_to_visualize, :3:, :, :].permute(1, 2, 3, 0)
    assert nerf_points.shape == (83, 21, 37, 3)
    isaac_origin_lines = get_isaac_origin_lines()
    colored_points_scatter = get_colored_points_scatter(
        nerf_points.reshape(-1, 3), nerf_densities.reshape(-1)
    )

    layout = go.Layout(
        scene=dict(xaxis=dict(title="X"), yaxis=dict(title="Y"), zaxis=dict(title="Z")),
        showlegend=True,
        title=f"Training datapoint: success={grasp_success[idx_to_visualize].item()}",
        width=800,
        height=800,
    )

    # Create the figure
    fig = go.Figure(layout=layout)
    for line in isaac_origin_lines:
        fig.add_trace(line)
    fig.add_trace(colored_points_scatter)
    fig.update_layout(legend_orientation="h")
    fig.show()
    break
