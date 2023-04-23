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

from typing import List
import os
import random
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from localscope import localscope
from omegaconf import OmegaConf, DictConfig, MISSING
from torch.utils.data import DataLoader, Dataset, random_split
from enum import Enum, auto
from tqdm import tqdm

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
    entity: str = MISSING
    project: str = MISSING
    name: str = MISSING
    group: str = MISSING
    job_type: str = MISSING


@dataclass
class DataConfig:
    frac_val: float = MISSING
    frac_test: float = MISSING
    frac_train: float = MISSING

    input_dataset_dir: str = MISSING


@dataclass
class TrainingConfig:
    batch_size: int = MISSING
    dataloader_num_workers: int = MISSING


class ConvOutputTo1D(Enum):
    FLATTEN = auto()  # (N, C, H, W) -> (N, C*H*W)
    AVG_POOL_SPATIAL = auto()  # (N, C, H, W) -> (N, C, 1, 1) -> (N, C)
    AVG_POOL_CHANNEL = auto()  # (N, C, H, W) -> (N, 1, H, W) -> (N, H*W)
    MAX_POOL_SPATIAL = auto()  # (N, C, H, W) -> (N, C, 1, 1) -> (N, C)
    MAX_POOL_CHANNEL = auto()  # (N, C, H, W) -> (N, 1, H, W) -> (N, H*W)


class PoolType(Enum):
    MAX = auto()
    AVG = auto()


@dataclass
class NeuralNetworkConfig:
    conv_channels: List[int] = MISSING
    pool_type: PoolType = MISSING
    dropout_prob: float = MISSING
    conv_output_to_1d: ConvOutputTo1D = MISSING

    mlp_hidden_layers: List[int] = MISSING


@dataclass
class Config:
    data: DataConfig = MISSING
    wandb: WandbConfig = MISSING
    training: TrainingConfig = MISSING
    neural_network: NeuralNetworkConfig = MISSING
    random_seed: int = MISSING


# %%
# Do I need this?
# config_store = ConfigStore.instance()
# config_store.store(name="config", node=Config)


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
    raw_cfg = compose(config_name="config", overrides=arguments)

# %%
# Runtime type-checking
cfg: Config = instantiate(raw_cfg)

# %%
print(f"Config:\n{OmegaConf.to_yaml(cfg)}")

# %% [markdown]
# # Set Random Seed


# %%
@localscope.mfc
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


set_seed(cfg.random_seed)


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
# # Dataset and Dataloader

# %%
# CONSTANTS AND PARAMS
ROOT_DIR = "/juno/u/tylerlum/github_repos/nerf_grasping"
RANDOM_SEED = 42


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
    batch_size=cfg.training.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=cfg.training.dataloader_num_workers,
)
val_loader = DataLoader(
    train_dataset,
    batch_size=cfg.training.batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=cfg.training.dataloader_num_workers,
)
test_loader = DataLoader(
    train_dataset,
    batch_size=cfg.training.batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=cfg.training.dataloader_num_workers,
)


# %% [markdown]
# # Visualize Dataset


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
    assert nerf_grid_input.shape == (cfg.training.batch_size, 4, 83, 21, 37)
    assert grasp_success.shape == (cfg.training.batch_size,)

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

# %%
idx_to_visualize = 0
for nerf_grid_input, grasp_success in val_loader:
    assert nerf_grid_input.shape == (cfg.training.batch_size, 4, 83, 21, 37)
    assert grasp_success.shape == (cfg.training.batch_size,)

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
        title=f"Validation datapoint: success={grasp_success[idx_to_visualize].item()}",
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

# %% [markdown]
# # Create Neural Network Model


# %%
@localscope.mfc
def mlp(
    num_inputs,
    num_outputs,
    hidden_layers,
    activation=nn.ReLU,
    output_activation=nn.Identity,
):
    layers = []
    layer_sizes = [num_inputs] + hidden_layers + [num_outputs]
    for i in range(len(layer_sizes) - 1):
        act = activation if i < len(layer_sizes) - 2 else output_activation
        layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class Mean(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.mean(x, dim=self.dim)


class Max(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.max(x, dim=self.dim)


@localscope.mfc
def conv_encoder(
    input_shape,
    conv_channels,
    pool_type=PoolType.MAX,
    dropout_prob=0.0,
    conv_output_to_1d=ConvOutputTo1D.FLATTEN,
    activation=nn.ReLU,
):
    # Input: Either (n_channels, n_dims) or (n_channels, height, width) or (n_channels, depth, height, width)

    # Validate input
    assert 2 <= len(input_shape) <= 4
    n_input_channels = input_shape[0]
    n_spatial_dims = len(input_shape[1:])

    # Layers for different input sizes
    n_spatial_dims_to_conv_layer_map = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
    n_spatial_dims_to_maxpool_layer_map = {
        1: nn.MaxPool1d,
        2: nn.MaxPool2d,
        3: nn.MaxPool3d,
    }
    n_spatial_dims_to_avgpool_layer_map = {
        1: nn.AvgPool1d,
        2: nn.AvgPool2d,
        3: nn.AvgPool3d,
    }
    n_spatial_dims_to_dropout_layer_map = {
        # 1: nn.Dropout1d,  # Not in some versions of torch
        2: nn.Dropout2d,
        3: nn.Dropout3d,
    }
    n_spatial_dims_to_adaptivemaxpool_layer_map = {
        1: nn.AdaptiveMaxPool1d,
        2: nn.AdaptiveMaxPool2d,
        3: nn.AdaptiveMaxPool3d,
    }
    n_spatial_dims_to_adaptiveavgpool_layer_map = {
        1: nn.AdaptiveMaxPool1d,
        2: nn.AdaptiveMaxPool2d,
        3: nn.AdaptiveMaxPool3d,
    }

    # Setup layer types
    conv_layer = n_spatial_dims_to_conv_layer_map[n_spatial_dims]
    if pool_type == PoolType.MAX:
        pool_layer = n_spatial_dims_to_maxpool_layer_map[n_spatial_dims]
    elif pool_type == PoolType.AVG:
        pool_layer = n_spatial_dims_to_avgpool_layer_map[n_spatial_dims]
    else:
        raise ValueError(f"Invalid pool_type = {pool_type}")
    dropout_layer = n_spatial_dims_to_dropout_layer_map[n_spatial_dims]

    # Conv layers
    layers = []
    n_channels = [n_input_channels] + conv_channels
    for i in range(len(n_channels) - 1):
        layers += [
            conv_layer(
                in_channels=n_channels[i],
                out_channels=n_channels[i + 1],
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            activation(),
            pool_layer(kernel_size=2, stride=2),
        ]
        if dropout_prob != 0.0:
            layers += [dropout_layer(p=dropout_prob)]

    # Convert from (n_channels, X) => (Y,)
    if conv_output_to_1d == ConvOutputTo1D.FLATTEN:
        layers.append(nn.Flatten(start_dim=1))
    elif conv_output_to_1d == ConvOutputTo1D.AVG_POOL_SPATIAL:
        adaptiveavgpool_layer = n_spatial_dims_to_adaptiveavgpool_layer_map[
            n_spatial_dims
        ]
        layers.append(
            adaptiveavgpool_layer(output_size=tuple([1 for _ in range(n_spatial_dims)]))
        )
        layers.append(nn.Flatten(start_dim=1))
    elif conv_output_to_1d == ConvOutputTo1D.MAX_POOL_SPATIAL:
        adaptivemaxpool_layer = n_spatial_dims_to_adaptivemaxpool_layer_map[
            n_spatial_dims
        ]
        layers.append(
            adaptivemaxpool_layer(output_size=tuple([1 for _ in range(n_spatial_dims)]))
        )
        layers.append(nn.Flatten(start_dim=1))
    elif conv_output_to_1d == ConvOutputTo1D.AVG_POOL_CHANNEL:
        channel_dim = 1
        layers.append(Mean(dim=channel_dim))
        layers.append(nn.Flatten(start_dim=1))
    elif conv_output_to_1d == ConvOutputTo1D.MAX_POOL_CHANNEL:
        channel_dim = 1
        layers.append(Max(dim=channel_dim))
        layers.append(nn.Flatten(start_dim=1))
    else:
        raise ValueError(f"Invalid conv_output_to_1d = {conv_output_to_1d}")

    return nn.Sequential(*layers)


# %%
class NeRF_to_Grasp_Success_Model(nn.Module):
    @localscope.mfc
    def __init__(self, input_example_shape, neural_network_config: NeuralNetworkConfig):
        super().__init__()
        self.input_example_shape = input_example_shape
        self.neural_network_config = neural_network_config

        self.conv = conv_encoder(
            input_shape=input_example_shape,
            conv_channels=neural_network_config.conv_channels,
            pool_type=neural_network_config.pool_type,
            dropout_prob=neural_network_config.dropout_prob,
            conv_output_to_1d=neural_network_config.conv_output_to_1d,
        )

        # Get conv output shape
        example_batch_size = 1
        example_input = torch.zeros((example_batch_size, *input_example_shape))
        conv_output = self.conv(example_input)
        assert (
            len(conv_output.shape) == 2 and conv_output.shape[0] == example_batch_size
        )
        _, conv_output_dim = conv_output.shape

        self.mlp = mlp(
            num_inputs=conv_output_dim,
            num_outputs=1,
            hidden_layers=neural_network_config.mlp_hidden_layers,
        )

    @localscope.mfc
    def forward(self, x):
        x = self.conv(x)
        x = self.mlp(x)
        return x

    @localscope.mfc
    def get_logit(self, x):
        return self.forward(x)

    @localscope.mfc
    def get_probability(self, x):
        return torch.sigmoid(self.get_logit(x))


# %%
