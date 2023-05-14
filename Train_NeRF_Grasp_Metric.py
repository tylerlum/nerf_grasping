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
# # Imports

# %%
import math
import os
import pickle
import random
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from enum import Enum, auto
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type
import matplotlib.pyplot as plt

import h5py
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from localscope import localscope
from omegaconf import OmegaConf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
    random_split,
    SubsetRandomSampler,
)
from torchinfo import summary
from torchviz import make_dot
from wandb.util import generate_id
from torch.profiler import profile, record_function, ProfilerActivity
from tyler_new_models import (
    PoolType,
    ConvOutputTo1D,
    Condition2D1D_ConcatFingersAfter1D,
    CNN_3D_Classifier,
    Encoder1DType,
    ClassifierConfig,
    dataclass_to_kwargs,
    get_scheduler,
)
from Train_NeRF_Grasp_Metric_Config import (
    Config,
    DataConfig,
    DataLoaderConfig,
    PreprocessConfig,
    TrainingConfig,
    CheckpointWorkspaceConfig,
    WandbConfig,
    PreprocessDensityType,
)

import wandb

# %%
if is_notebook():
    from tqdm.notebook import tqdm as std_tqdm
else:
    from tqdm import tqdm as std_tqdm

tqdm = partial(std_tqdm, dynamic_ncols=True)


# %% [markdown]
# # Setup Config for Static Type-Checking


# %%
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver(
    "datetime_str", lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), replace=True
)


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
from hydra.errors import ConfigCompositionException

try:
    with initialize(version_base="1.1", config_path="Train_NeRF_Grasp_Metric_cfg"):
        raw_cfg = compose(config_name="config", overrides=arguments)

    # Runtime type-checking
    cfg: Config = instantiate(raw_cfg)
except ConfigCompositionException as e:
    print(f"ConfigCompositionException: {e}")
    print()
    print(f"e.__cause__ = {e.__cause__}")
    raise e.__cause__

# %%
print(f"Config:\n{OmegaConf.to_yaml(cfg)}")

# %%
if cfg.dry_run:
    print("Dry run passed. Exiting.")
    exit()

# %% [markdown]
# # Set Random Seed


# %%
@localscope.mfc
def set_seed(seed) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)  # TODO: Is this slowing things down?

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Set random seed to {seed}")


set_seed(cfg.random_seed)

# %% [markdown]
# # Setup Checkpoint Workspace and Maybe Resume Previous Run


# %%
@localscope.mfc
def load_checkpoint(checkpoint_workspace_dir_path: str) -> Optional[Dict[str, Any]]:
    checkpoint_filepaths = sorted(
        [
            os.path.join(checkpoint_workspace_dir_path, filename)
            for filename in os.listdir(checkpoint_workspace_dir_path)
            if filename.endswith(".pt")
        ]
    )
    if len(checkpoint_filepaths) == 0:
        print("No checkpoint found")
        return None
    return torch.load(checkpoint_filepaths[-1])


# %%
# Set up checkpoint_workspace
if not os.path.exists(cfg.checkpoint_workspace.root_dir):
    os.makedirs(cfg.checkpoint_workspace.root_dir)

checkpoint_workspace_dir_path = os.path.join(
    cfg.checkpoint_workspace.root_dir, cfg.checkpoint_workspace.leaf_dir
)

# Remove checkpoint_workspace directory if force_no_resume is set
if (
    os.path.exists(checkpoint_workspace_dir_path)
    and cfg.checkpoint_workspace.force_no_resume
):
    print(f"force_no_resume = {cfg.checkpoint_workspace.force_no_resume}")
    print(f"Removing checkpoint_workspace directory at {checkpoint_workspace_dir_path}")
    shutil.rmtree(checkpoint_workspace_dir_path)
    print("Done removing checkpoint_workspace directory")

# Read wandb_run_id from checkpoint_workspace if it exists
wandb_run_id_filepath = os.path.join(checkpoint_workspace_dir_path, "wandb_run_id.txt")
if os.path.exists(checkpoint_workspace_dir_path):
    print(
        f"checkpoint_workspace directory already exists at {checkpoint_workspace_dir_path}"
    )

    print(f"Loading wandb_run_id from {wandb_run_id_filepath}")
    with open(wandb_run_id_filepath, "r") as f:
        wandb_run_id = f.read()
    print(f"Done loading wandb_run_id = {wandb_run_id}")

else:
    print(f"Creating checkpoint_workspace directory at {checkpoint_workspace_dir_path}")
    os.makedirs(checkpoint_workspace_dir_path)
    print("Done creating checkpoint_workspace directory")

    wandb_run_id = generate_id()
    print(f"Saving wandb_run_id = {wandb_run_id} to {wandb_run_id_filepath}")
    with open(wandb_run_id_filepath, "w") as f:
        f.write(wandb_run_id)
    print("Done saving wandb_run_id")

# %% [markdown]
# # Setup Wandb Logging

# %%
# Add to config
wandb.init(
    entity=cfg.wandb.entity,
    project=cfg.wandb.project,
    name=cfg.wandb.name,
    group=cfg.wandb.group if len(cfg.wandb.group) > 0 else None,
    job_type=cfg.wandb.job_type if len(cfg.wandb.job_type) > 0 else None,
    config=OmegaConf.to_container(cfg, throw_on_missing=True),
    id=wandb_run_id,
    resume="never" if cfg.checkpoint_workspace.force_no_resume else "allow",
    reinit=True,
)

# %% [markdown]
# # Dataset and Dataloader

# %%
# CONSTANTS AND PARAMS
NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z = 83, 21, 37
NUM_XYZ = 3
NUM_DENSITY = 1
NUM_CHANNELS = NUM_XYZ + NUM_DENSITY
INPUT_EXAMPLE_SHAPE = (NUM_CHANNELS, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z)
NERF_COORDINATE_START_IDX, NERF_COORDINATE_END_IDX = 0, 3
NERF_DENSITY_START_IDX, NERF_DENSITY_END_IDX = 3, 4
DELTA = 0.001  # 1mm between grid points

assert NERF_COORDINATE_END_IDX == NERF_COORDINATE_START_IDX + NUM_XYZ
assert NERF_DENSITY_END_IDX == NERF_DENSITY_START_IDX + NUM_DENSITY


# %%


class NeRFGrid_To_GraspSuccess_HDF5_Dataset(Dataset):
    # @localscope.mfc  # ValueError: Cell is empty
    def __init__(
        self,
        input_hdf5_filepath: str,
        max_num_data_points: Optional[int] = None,
        load_nerf_grid_inputs_in_ram: bool = False,
        load_grasp_successes_in_ram: bool = False,
    ) -> None:
        super().__init__()
        self.input_hdf5_filepath = input_hdf5_filepath

        # Recommended in https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/13
        self.hdf5_file = None

        with h5py.File(self.input_hdf5_filepath, "r") as hdf5_file:
            self.len = self._set_length(
                hdf5_file=hdf5_file, max_num_data_points=max_num_data_points
            )

            # Check that the data is in the expected format
            assert len(hdf5_file["/grasp_success"].shape) == 1
            assert hdf5_file["/nerf_grid_input"].shape[1:] == INPUT_EXAMPLE_SHAPE

            # This is usually too big for RAM
            self.nerf_grid_inputs = (
                torch.from_numpy(hdf5_file["/nerf_grid_input"][()]).float()
                if load_nerf_grid_inputs_in_ram
                else None
            )

            # This is small enough to fit in RAM
            self.grasp_successes = (
                torch.from_numpy(hdf5_file["/grasp_success"][()]).long()
                if load_grasp_successes_in_ram
                else None
            )

    @localscope.mfc
    def _set_length(
        self, hdf5_file: h5py.File, max_num_data_points: Optional[int]
    ) -> int:
        length = (
            hdf5_file.attrs["num_data_points"]
            if "num_data_points" in hdf5_file.attrs
            else hdf5_file["/grasp_success"].shape[0]
        )
        if length != hdf5_file["/grasp_success"].shape[0]:
            print(
                f"WARNING: num_data_points = {length} != grasp_success.shape[0] = {hdf5_file['/grasp_success'].shape[0]}"
            )

        # Constrain length of dataset if max_num_data_points is set
        if max_num_data_points is not None:
            print(f"Constraining dataset length to {max_num_data_points}")
            length = max_num_data_points

        return length

    @localscope.mfc
    def __len__(self) -> int:
        return self.len

    @localscope.mfc(
        allowed=[
            "INPUT_EXAMPLE_SHAPE",
            "NERF_DENSITY_START_IDX",
            "NERF_DENSITY_END_IDX",
        ]
    )
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.hdf5_file is None:
            # Hope to speed up with rdcc params
            self.hdf5_file = h5py.File(
                self.input_hdf5_filepath,
                "r",
                rdcc_nbytes=1024**2 * 4_000,
                rdcc_w0=0.75,
                rdcc_nslots=4_000,
            )

        nerf_grid_input = (
            torch.from_numpy(self.hdf5_file["/nerf_grid_input"][idx]).float()
            if self.nerf_grid_inputs is None
            else self.nerf_grid_inputs[idx]
        )

        grasp_success = (
            torch.from_numpy(np.array(self.hdf5_file["/grasp_success"][idx])).long()
            if self.grasp_successes is None
            else self.grasp_successes[idx]
        )

        assert nerf_grid_input.shape == INPUT_EXAMPLE_SHAPE
        assert grasp_success.shape == ()

        return nerf_grid_input, grasp_success


# %%
@localscope.mfc(
    allowed=[
        "ctx_factory",  # global from torch.no_grad
        "DELTA",
    ]
)
@torch.no_grad()
def preprocess_to_alpha(nerf_densities: torch.Tensor) -> torch.Tensor:
    # alpha = 1 - exp(-delta * sigma)
    #       = probability of collision within this segment starting from beginning of segment
    return 1.0 - torch.exp(-DELTA * nerf_densities)


@localscope.mfc(
    allowed=[
        "ctx_factory",  # global from torch.no_grad
        "INPUT_EXAMPLE_SHAPE",
        "NERF_DENSITY_START_IDX",
        "NERF_DENSITY_END_IDX",
        "NUM_PTS_X",
    ]
)
@torch.no_grad()
def preprocess_to_weight(nerf_densities: torch.Tensor) -> torch.Tensor:
    # alpha_j = 1 - exp(-delta_j * sigma_j)
    #         = probability of collision within this segment starting from beginning of segment
    # weight_j = alpha_j * (1 - alpha_{j-1}) * ... * (1 - alpha_1))
    #          = probability of collision within j-th segment starting from left edge

    @localscope.mfc
    def compute_weight(alpha: torch.Tensor) -> torch.Tensor:
        x_axis_dim = -3
        # [1 - alpha_1, (1 - alpha_1) * (1 - alpha_2), ..., (1 - alpha_1) * ... * (1 - alpha_{NUM_PTS_X}))]
        cumprod_1_minus_alpha_from_left = (1 - alpha).cumprod(dim=x_axis_dim)

        # [1, 1 - alpha_1, (1 - alpha_1) * (1 - alpha_2), ..., (1 - alpha_1) * ... * (1 - alpha_{NUM_PTS_X-1}))]
        cumprod_1_minus_alpha_from_left_shifted = torch.cat(
            [
                torch.ones_like(
                    cumprod_1_minus_alpha_from_left[:, :1],
                    dtype=alpha.dtype,
                    device=alpha.device,
                ),
                cumprod_1_minus_alpha_from_left[:, :-1],
            ],
            dim=x_axis_dim,
        )

        # weight_j = alpha_j * (1 - alpha_{j-1}) * ... * (1 - alpha_1))
        weight = alpha * cumprod_1_minus_alpha_from_left_shifted
        return weight

    assert nerf_densities.shape[-3:] == (NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z)

    # [alpha_1, alpha_2, ..., alpha_{NUM_PTS_X}]
    alpha = preprocess_to_alpha(nerf_densities)

    # weight_j = alpha_j * (1 - alpha_{j-1}) * ... * (1 - alpha_1))
    weight = compute_weight(alpha)

    return weight


# %%
@localscope.mfc(
    allowed=[
        "cfg",
        "NUM_PTS_X",
        "NUM_PTS_Y",
        "NUM_PTS_Z",
        "NUM_XYZ",
        "NUM_CHANNELS",
        "NERF_DENSITY_START_IDX",
        "NERF_DENSITY_END_IDX",
        "NERF_COORDINATE_START_IDX",
        "NERF_COORDINATE_END_IDX",
    ]
)
def get_nerf_densities_and_points(
    nerf_grid_inputs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = nerf_grid_inputs.shape[0]
    assert (
        len(nerf_grid_inputs.shape) == 5
        and nerf_grid_inputs.shape[0] == batch_size
        and nerf_grid_inputs.shape[1] == NUM_CHANNELS
        and nerf_grid_inputs.shape[2] in [NUM_PTS_X // 2, NUM_PTS_X]
        and nerf_grid_inputs.shape[3] == NUM_PTS_Y
        and nerf_grid_inputs.shape[4] == NUM_PTS_Z
    )

    assert torch.is_tensor(nerf_grid_inputs)

    nerf_densities = nerf_grid_inputs[
        :, NERF_DENSITY_START_IDX:NERF_DENSITY_END_IDX
    ].squeeze(dim=1)
    nerf_points = nerf_grid_inputs[:, NERF_COORDINATE_START_IDX:NERF_COORDINATE_END_IDX]

    return nerf_densities, nerf_points


@localscope.mfc(
    allowed=[
        "cfg",
        "NUM_PTS_X",
        "NUM_PTS_Y",
        "NUM_PTS_Z",
        "NUM_XYZ",
    ]
)
def get_global_params(
    nerf_points: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = nerf_points.shape[0]
    assert (
        len(nerf_points.shape) == 5
        and nerf_points.shape[0] == batch_size
        and nerf_points.shape[1] == NUM_XYZ
        and nerf_points.shape[2] in [NUM_PTS_X // 2, NUM_PTS_X]
        and nerf_points.shape[3] == NUM_PTS_Y
        and nerf_points.shape[4] == NUM_PTS_Z
    )
    assert torch.is_tensor(nerf_points)

    new_origin_x_idx, new_origin_y_idx, new_origin_z_idx = (
        0,
        NUM_PTS_Y // 2,
        NUM_PTS_Z // 2,
    )

    new_origin = nerf_points[:, :, new_origin_x_idx, new_origin_y_idx, new_origin_z_idx]
    assert new_origin.shape == (batch_size, NUM_XYZ)

    new_x_axis = nn.functional.normalize(
        nerf_points[:, :, new_origin_x_idx + 1, new_origin_y_idx, new_origin_z_idx]
        - new_origin,
        dim=-1,
    )
    new_y_axis = nn.functional.normalize(
        nerf_points[:, :, new_origin_x_idx, new_origin_y_idx + 1, new_origin_z_idx]
        - new_origin,
        dim=-1,
    )
    new_z_axis = nn.functional.normalize(
        nerf_points[:, :, new_origin_x_idx, new_origin_y_idx, new_origin_z_idx + 1]
        - new_origin,
        dim=-1,
    )

    assert torch.isclose(
        torch.cross(new_x_axis, new_y_axis, dim=-1), new_z_axis, rtol=1e-3, atol=1e-3
    ).all()

    # new_z_axis is implicit from the cross product of new_x_axis and new_y_axis
    return new_origin, new_x_axis, new_y_axis


@localscope.mfc(
    allowed=[
        "cfg",
        "NUM_XYZ",
    ]
)
def invariance_transformation(
    left_global_params: Tuple[torch.Tensor, ...],
    right_global_params: Tuple[torch.Tensor, ...],
    rotate_polar_angle: bool = False,
    reflect_around_xz_plane_randomly: bool = False,
    reflect_around_xy_plane_randomly: bool = False,
    remove_y_axis: bool = False,
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
    left_origin, left_x_axis, left_y_axis = left_global_params
    right_origin, right_x_axis, right_y_axis = right_global_params

    batch_size = left_origin.shape[0]

    assert (
        left_origin.shape
        == right_origin.shape
        == left_x_axis.shape
        == right_x_axis.shape
        == left_y_axis.shape
        == right_y_axis.shape
        == (batch_size, NUM_XYZ)
    )

    # Always do rotation wrt left
    # Get azimuth angle of left_origin
    azimuth_angle = torch.atan2(left_origin[:, 1], left_origin[:, 0])
    assert azimuth_angle.shape == (batch_size,)

    # Reverse azimuth angle around z to get back to xz plane (left_origin_y = 0)
    # This handles invariance in both xy and yaw (angle around z)
    transformation_matrix_around_z = torch.stack(
        [
            torch.stack(
                [
                    torch.cos(-azimuth_angle),
                    -torch.sin(-azimuth_angle),
                    torch.zeros_like(azimuth_angle),
                    torch.zeros_like(azimuth_angle),
                ],
                dim=0,
            ),
            torch.stack(
                [
                    torch.sin(-azimuth_angle),
                    torch.cos(-azimuth_angle),
                    torch.zeros_like(azimuth_angle),
                    torch.zeros_like(azimuth_angle),
                ],
                dim=0,
            ),
            torch.stack(
                [
                    torch.zeros_like(azimuth_angle),
                    torch.zeros_like(azimuth_angle),
                    torch.ones_like(azimuth_angle),
                    torch.zeros_like(azimuth_angle),
                ],
                dim=0,
            ),
            torch.stack(
                [
                    torch.zeros_like(azimuth_angle),
                    torch.zeros_like(azimuth_angle),
                    torch.zeros_like(azimuth_angle),
                    torch.ones_like(azimuth_angle),
                ],
                dim=0,
            ),
        ],
        dim=0,
    )
    assert transformation_matrix_around_z.shape == (4, 4, batch_size)
    transformation_matrix_around_z = transformation_matrix_around_z.permute(2, 0, 1)
    assert transformation_matrix_around_z.shape == (batch_size, 4, 4)

    @localscope.mfc(allowed=["batch_size"])
    def transform(
        transformation_matrix: torch.Tensor, point: torch.Tensor
    ) -> torch.Tensor:
        assert transformation_matrix.shape == (batch_size, 4, 4)
        assert point.shape == (batch_size, 3)

        # Concat 1s to point to make it homogenous
        point_homo = torch.cat(
            [point, torch.ones((batch_size, 1), device=point.device)], dim=1
        )
        assert point_homo.shape == (batch_size, 4)

        # Add dimension so we can do batched matrix multiply
        point_homo = point_homo[:, :, None]
        assert point_homo.shape == (batch_size, 4, 1)

        # Transform (batch_size, 4, 4) @ (batch_size, 4, 1) => (batch_size, 4, 1)
        transformed_point = transformation_matrix @ point_homo
        assert transformed_point.shape == (batch_size, 4, 1)

        # Extract points
        transformed_point = transformed_point[:, :3, 0]
        assert transformed_point.shape == (batch_size, 3)

        return transformed_point

    # Transform all
    left_origin = transform(transformation_matrix_around_z, left_origin)
    left_x_axis = transform(transformation_matrix_around_z, left_x_axis)
    left_y_axis = transform(transformation_matrix_around_z, left_y_axis)
    right_origin = transform(transformation_matrix_around_z, right_origin)
    right_x_axis = transform(transformation_matrix_around_z, right_x_axis)
    right_y_axis = transform(transformation_matrix_around_z, right_y_axis)

    assert torch.allclose(
        left_origin[:, 1], torch.zeros_like(left_origin[:, 1]), rtol=1e-3, atol=1e-3
    )  # left_origin_y = 0

    if rotate_polar_angle:
        # Always do rotation wrt left
        # Get polar angle of left_origin
        polar_angle = torch.atan2(
            torch.sqrt(left_origin[:, 0] ** 2 + left_origin[:, 1] ** 2),
            left_origin[:, 2],
        )

        # Angle between x axis and left_origin
        polar_angle = torch.pi / 2 - polar_angle
        assert polar_angle.shape == (batch_size,)

        # Rotation around y, positive to bring down to z = 0
        transformation_matrix_around_y = torch.stack(
            [
                torch.stack(
                    [
                        torch.cos(polar_angle),
                        torch.zeros_like(polar_angle),
                        torch.sin(polar_angle),
                        torch.zeros_like(polar_angle),
                    ],
                    dim=0,
                ),
                torch.stack(
                    [
                        torch.zeros_like(polar_angle),
                        torch.ones_like(polar_angle),
                        torch.zeros_like(polar_angle),
                        torch.zeros_like(polar_angle),
                    ],
                    dim=0,
                ),
                torch.stack(
                    [
                        -torch.sin(polar_angle),
                        torch.zeros_like(polar_angle),
                        torch.cos(polar_angle),
                        torch.zeros_like(polar_angle),
                    ],
                    dim=0,
                ),
                torch.stack(
                    [
                        torch.zeros_like(polar_angle),
                        torch.zeros_like(polar_angle),
                        torch.zeros_like(polar_angle),
                        torch.ones_like(polar_angle),
                    ],
                    dim=0,
                ),
            ],
            dim=0,
        )
        assert transformation_matrix_around_y.shape == (4, 4, batch_size)
        transformation_matrix_around_y = transformation_matrix_around_y.permute(2, 0, 1)
        assert transformation_matrix_around_y.shape == (batch_size, 4, 4)

        # Transform all
        left_origin = transform(transformation_matrix_around_y, left_origin)
        left_x_axis = transform(transformation_matrix_around_y, left_x_axis)
        left_y_axis = transform(transformation_matrix_around_y, left_y_axis)
        right_origin = transform(transformation_matrix_around_y, right_origin)
        right_x_axis = transform(transformation_matrix_around_y, right_x_axis)
        right_y_axis = transform(transformation_matrix_around_y, right_y_axis)

        assert torch.allclose(
            left_origin[:, 2], torch.zeros_like(left_origin[:, 2]), rtol=1e-3, atol=1e-3
        )  # left_origin_z = 0

    # To handle additional invariance, we can reflect around planes of symmetry
    # yz is handled already by the rotation around z
    # xz plane is probably the best choice, as there is symmetry around moving left and right
    # xy plane probably doesn't make sense, as gravity affects this axis

    # Reflect around xz plane (-y)
    if reflect_around_xz_plane_randomly:
        reflect = torch.rand((batch_size,)) > 0.5
        left_origin = torch.where(
            reflect[:, None], left_origin * torch.tensor([1, -1, 1]), left_origin
        )
        left_x_axis = torch.where(
            reflect[:, None], left_x_axis * torch.tensor([1, -1, 1]), left_x_axis
        )
        left_y_axis = torch.where(
            reflect[:, None], left_y_axis * torch.tensor([1, -1, 1]), left_y_axis
        )
        right_origin = torch.where(
            reflect[:, None], right_origin * torch.tensor([1, -1, 1]), right_origin
        )
        right_x_axis = torch.where(
            reflect[:, None], right_x_axis * torch.tensor([1, -1, 1]), right_x_axis
        )
        right_y_axis = torch.where(
            reflect[:, None], right_y_axis * torch.tensor([1, -1, 1]), right_y_axis
        )

    # Reflect around xy plane (-z)
    if reflect_around_xy_plane_randomly:
        reflect = torch.rand((batch_size,)) > 0.5
        left_origin = torch.where(
            reflect[:, None], left_origin * torch.tensor([1, 1, -1]), left_origin
        )
        left_x_axis = torch.where(
            reflect[:, None], left_x_axis * torch.tensor([1, 1, -1]), left_x_axis
        )
        left_y_axis = torch.where(
            reflect[:, None], left_y_axis * torch.tensor([1, 1, -1]), left_y_axis
        )
        right_origin = torch.where(
            reflect[:, None], right_origin * torch.tensor([1, 1, -1]), right_origin
        )
        right_x_axis = torch.where(
            reflect[:, None], right_x_axis * torch.tensor([1, 1, -1]), right_x_axis
        )
        right_y_axis = torch.where(
            reflect[:, None], right_y_axis * torch.tensor([1, 1, -1]), right_y_axis
        )

    # y-axis gives you the orientation around the approach direction, which may not be important
    if remove_y_axis:
        return ((left_origin, left_x_axis), (right_origin, right_x_axis))
    return (
        (
            left_origin,
            left_x_axis,
            left_y_axis,
        ),
        (
            right_origin,
            right_x_axis,
            right_y_axis,
        ),
    )


@localscope.mfc(allowed=["cfg", "INPUT_EXAMPLE_SHAPE", "NUM_PTS_X", "NUM_XYZ"])
def preprocess_nerf_grid_inputs(
    nerf_grid_inputs: torch.Tensor,
    flip_left_right_randomly: bool = False,
    preprocess_density_type: PreprocessDensityType = PreprocessDensityType.DENSITY,
    add_invariance_transformations: bool = False,
    rotate_polar_angle: bool = False,
    reflect_around_xz_plane_randomly: bool = False,
    reflect_around_xy_plane_randomly: bool = False,
    remove_y_axis: bool = False,
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
    batch_size = nerf_grid_inputs.shape[0]
    assert nerf_grid_inputs.shape == (
        batch_size,
        *INPUT_EXAMPLE_SHAPE,
    )
    assert torch.is_tensor(nerf_grid_inputs)

    # Split into left and right
    # Need to rotate the right side around the z-axis, so that the x-axis is pointing toward the left
    # so need to flip the x and y axes
    num_pts_per_side = NUM_PTS_X // 2
    x_dim, y_dim = -3, -2
    left_nerf_grid_inputs = nerf_grid_inputs[:, :, :num_pts_per_side, :, :]
    right_nerf_grid_inputs = nerf_grid_inputs[:, :, -num_pts_per_side:, :, :].flip(
        dims=(x_dim, y_dim)
    )

    left_nerf_densities, left_nerf_points = get_nerf_densities_and_points(
        left_nerf_grid_inputs
    )
    right_nerf_densities, right_nerf_points = get_nerf_densities_and_points(
        right_nerf_grid_inputs
    )

    # Flip which side is left and right
    if flip_left_right_randomly:
        flip = torch.rand((batch_size,)) > 0.5
        left_nerf_densities, right_nerf_densities = (
            torch.where(
                flip[:, None, None, None], right_nerf_densities, left_nerf_densities
            ),
            torch.where(
                flip[:, None, None, None], left_nerf_densities, right_nerf_densities
            ),
        )
        left_nerf_points, right_nerf_points = (
            torch.where(
                flip[:, None, None, None, None], right_nerf_points, left_nerf_points
            ),
            torch.where(
                flip[:, None, None, None, None], left_nerf_points, right_nerf_points
            ),
        )

    # Extract global params
    left_global_params = get_global_params(left_nerf_points)
    right_global_params = get_global_params(right_nerf_points)

    # Preprocess densities
    preprocess_type_to_fn = {
        PreprocessDensityType.DENSITY: lambda x: x,
        PreprocessDensityType.ALPHA: preprocess_to_alpha,
        PreprocessDensityType.WEIGHT: preprocess_to_weight,
    }
    preprocess_density_fn = preprocess_type_to_fn[preprocess_density_type]
    left_nerf_densities = preprocess_density_fn(left_nerf_densities)
    right_nerf_densities = preprocess_density_fn(right_nerf_densities)

    # Invariance transformations
    if add_invariance_transformations:
        left_global_params, right_global_params = invariance_transformation(
            left_global_params=left_global_params,
            right_global_params=right_global_params,
            rotate_polar_angle=rotate_polar_angle,
            reflect_around_xz_plane_randomly=reflect_around_xz_plane_randomly,
            reflect_around_xy_plane_randomly=reflect_around_xy_plane_randomly,
            remove_y_axis=remove_y_axis,
        )

    # Concatenate global params into a single tensor
    assert len(left_global_params) == len(right_global_params)
    assert all([param.shape == (batch_size, NUM_XYZ) for param in left_global_params])
    assert all([param.shape == (batch_size, NUM_XYZ) for param in right_global_params])
    left_global_params = torch.cat(left_global_params, dim=1)
    right_global_params = torch.cat(right_global_params, dim=1)

    return (
        (left_nerf_densities, left_global_params),
        (right_nerf_densities, right_global_params),
    )


# %%
class DatasetType(Enum):
    HDF5_FILE = auto()


assert cfg.data.input_dataset_path.endswith(".h5")
dataset_type = DatasetType.HDF5_FILE
input_dataset_full_path = os.path.join(
    cfg.data.input_dataset_root_dir, cfg.data.input_dataset_path
)

if dataset_type == DatasetType.HDF5_FILE:
    full_dataset = NeRFGrid_To_GraspSuccess_HDF5_Dataset(
        input_dataset_full_path,
        max_num_data_points=cfg.data.max_num_data_points,
        load_nerf_grid_inputs_in_ram=cfg.dataloader.load_nerf_grid_inputs_in_ram,
        load_grasp_successes_in_ram=cfg.dataloader.load_grasp_successes_in_ram,
    )
else:
    raise ValueError(f"Unknown dataset type: {dataset_type}")

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [cfg.data.frac_train, cfg.data.frac_val, cfg.data.frac_test],
    generator=torch.Generator().manual_seed(cfg.random_seed),
)

# %%
print(f"Train dataset size: {len(train_dataset)}")
print(f"Val dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# %%
assert len(set.intersection(set(train_dataset.indices), set(val_dataset.indices))) == 0
assert len(set.intersection(set(train_dataset.indices), set(test_dataset.indices))) == 0
assert len(set.intersection(set(val_dataset.indices), set(test_dataset.indices))) == 0


# %%
# Create custom class to store a batch
class BatchData:
    @localscope.mfc(allowed=["NUM_PTS_X", "NUM_PTS_Y", "NUM_PTS_Z"])
    def __init__(
        self,
        left_nerf_densities,
        left_global_params,
        right_nerf_densities,
        right_global_params,
        grasp_successes,
    ):
        assert (
            left_nerf_densities.shape
            == right_nerf_densities.shape
            == (
                left_nerf_densities.shape[0],
                NUM_PTS_X // 2,
                NUM_PTS_Y,
                NUM_PTS_Z,
            )
        )
        assert (
            left_global_params.shape
            == right_global_params.shape
            == (
                left_global_params.shape[0],
                left_global_params.shape[1],
            )
        )

        self._nerf_grid_inputs = torch.stack(
            [left_nerf_densities, right_nerf_densities], dim=1
        )
        self.global_params = torch.stack(
            [left_global_params, right_global_params], dim=1
        )
        self.grasp_successes = grasp_successes

        assert self._nerf_grid_inputs.shape == (
            left_nerf_densities.shape[0],
            2,
            NUM_PTS_X // 2,
            NUM_PTS_Y,
            NUM_PTS_Z,
        )
        assert self.global_params.shape == (
            left_global_params.shape[0],
            2,
            left_global_params.shape[1],
        )

    @localscope.mfc
    def to(self, device):
        self._nerf_grid_inputs = self._nerf_grid_inputs.to(device)
        self.global_params = self.global_params.to(device)
        self.grasp_successes = self.grasp_successes.to(device)
        return self

    @property
    @localscope.mfc(allowed=["cfg"])
    def nerf_grid_inputs(self) -> torch.Tensor:
        # TODO HACK Downsample
        return self._nerf_grid_inputs[
            :,
            :,
            :: cfg.dataloader.downsample_factor_x,
            :: cfg.dataloader.downsample_factor_y,
            :: cfg.dataloader.downsample_factor_z,
        ]

    @property
    @localscope.mfc
    def left_nerf_densities(self) -> torch.Tensor:
        return self._nerf_grid_inputs[:, 0]

    @property
    @localscope.mfc
    def right_nerf_densities(self) -> torch.Tensor:
        return self._nerf_grid_inputs[:, 1]

    @property
    @localscope.mfc
    def left_global_params(self) -> torch.Tensor:
        return self.global_params[:, 0]

    @property
    @localscope.mfc
    def right_global_params(self) -> torch.Tensor:
        return self.global_params[:, 1]


# %%
@localscope.mfc(allowed=["cfg"])
def custom_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> BatchData:
    batch = torch.utils.data.dataloader.default_collate(batch)

    nerf_grid_inputs, grasp_successes = batch
    (
        (left_nerf_densities, left_global_params),
        (right_nerf_densities, right_global_params),
    ) = preprocess_nerf_grid_inputs(
        nerf_grid_inputs=nerf_grid_inputs,
        flip_left_right_randomly=cfg.preprocess.flip_left_right_randomly,
        preprocess_density_type=cfg.preprocess.density_type,
        add_invariance_transformations=cfg.preprocess.add_invariance_transformations,
        rotate_polar_angle=cfg.preprocess.rotate_polar_angle,
        reflect_around_xz_plane_randomly=cfg.preprocess.reflect_around_xz_plane_randomly,
        reflect_around_xy_plane_randomly=cfg.preprocess.reflect_around_xy_plane_randomly,
        remove_y_axis=cfg.preprocess.remove_y_axis,
    )

    return BatchData(
        left_nerf_densities=left_nerf_densities,
        left_global_params=left_global_params,
        right_nerf_densities=right_nerf_densities,
        right_global_params=right_global_params,
        grasp_successes=grasp_successes,
    )


@localscope.mfc(allowed=["cfg"])
def custom_collate_fn_without_invariance_transformations(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> BatchData:
    batch = torch.utils.data.dataloader.default_collate(batch)

    nerf_grid_inputs, grasp_successes = batch
    (
        (left_nerf_densities, left_global_params),
        (right_nerf_densities, right_global_params),
    ) = preprocess_nerf_grid_inputs(
        nerf_grid_inputs=nerf_grid_inputs,
        flip_left_right_randomly=cfg.preprocess.flip_left_right_randomly,
        preprocess_density_type=cfg.preprocess.density_type,
        add_invariance_transformations=False,  # Force to False
        rotate_polar_angle=cfg.preprocess.rotate_polar_angle,
        reflect_around_xz_plane_randomly=cfg.preprocess.reflect_around_xz_plane_randomly,
        reflect_around_xy_plane_randomly=cfg.preprocess.reflect_around_xy_plane_randomly,
        remove_y_axis=cfg.preprocess.remove_y_axis,
    )

    return BatchData(
        left_nerf_densities=left_nerf_densities,
        left_global_params=left_global_params,
        right_nerf_densities=right_nerf_densities,
        right_global_params=right_global_params,
        grasp_successes=grasp_successes,
    )


# %%
train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.dataloader.batch_size,
    shuffle=True,
    pin_memory=cfg.dataloader.pin_memory,
    num_workers=cfg.dataloader.num_workers,
    collate_fn=custom_collate_fn,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.dataloader.batch_size,
    shuffle=False,
    pin_memory=cfg.dataloader.pin_memory,
    num_workers=cfg.dataloader.num_workers,
    collate_fn=custom_collate_fn,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=cfg.dataloader.batch_size,
    shuffle=False,
    pin_memory=cfg.dataloader.pin_memory,
    num_workers=cfg.dataloader.num_workers,
    collate_fn=custom_collate_fn,
)

# %%
example_batch_data: BatchData = next(iter(val_loader))

# %%
print(f"left_nerf_densities.shape: {example_batch_data.left_nerf_densities.shape}")
print(f"left_global_params.shape: {example_batch_data.left_global_params.shape}")
print(f"right_nerf_densities.shape: {example_batch_data.right_nerf_densities.shape}")
print(f"right_global_params.shape: {example_batch_data.right_global_params.shape}")


# %%
@localscope.mfc(allowed=["NUM_PTS_X", "NUM_PTS_Y", "NUM_PTS_Z"])
def visualize_nerf_density_imgs(
    nerf_densities: torch.Tensor,
    idx_to_visualize: int = 0,
    name: str = "Nerf Densities Grid",
    save_to_wandb: bool = False,
) -> plt.Figure:
    nerf_density = nerf_densities[idx_to_visualize]
    assert nerf_density.shape == (NUM_PTS_X // 2, NUM_PTS_Y, NUM_PTS_Z)

    # Visualize nerf densities
    nerf_density = (
        nerf_density.cpu().numpy().transpose(0, 2, 1)
    )  # Tranpose because z is last, but should be height
    num_imgs, height, width = nerf_density.shape

    # Create grid of images
    images = [nerf_density[i] for i in range(num_imgs)]
    num_rows = math.ceil(math.sqrt(num_imgs))
    num_cols = math.ceil(num_imgs / num_rows)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20))
    for i, ax in enumerate(axes.flatten()):
        ax.axis("off")

        if i >= num_imgs:
            continue

        ax.imshow(images[i])
        ax.set_title(f"Image {i}")

    fig.suptitle(name)
    fig.tight_layout()

    if save_to_wandb:
        print(f"Saving {name} to wandb")
        wandb.log({name: fig})

    return fig


# %%
if cfg.visualize_data:
    visualize_nerf_density_imgs(
        example_batch_data.left_nerf_densities,
        idx_to_visualize=0,
        name="Left Nerf Densities Grid",
        save_to_wandb=True,
    )

# %%
if cfg.visualize_data:
    visualize_nerf_density_imgs(
        example_batch_data.right_nerf_densities,
        idx_to_visualize=0,
        name="Right Nerf Densities Grid",
        save_to_wandb=True,
    )


# %%
# %%
@localscope.mfc(allowed=["NUM_PTS_X", "NUM_PTS_Y", "NUM_PTS_Z"])
def visualize_nerf_videos(
    nerf_densities: torch.Tensor,
    idx_to_visualize: int = 0,
    name: str = "Nerf Densities Video",
    save_to_wandb: bool = False,
    fps: int = 10,
) -> np.ndarray:
    nerf_density = nerf_densities[idx_to_visualize]
    assert nerf_density.shape == (NUM_PTS_X // 2, NUM_PTS_Y, NUM_PTS_Z)

    # Visualize nerf densities
    nerf_density = (
        nerf_density.cpu().numpy().transpose(0, 2, 1)
    )  # Tranpose because z is last, but should be height
    num_imgs, height, width = nerf_density.shape

    # Create grid of images
    images = [nerf_density[i] for i in range(num_imgs)]

    # need to be in shape (num_imgs, 3, H, W) and be np.uint8 in [0, 255]
    wandb_video = (
        np.array(images).reshape(num_imgs, 1, height, width).repeat(repeats=3, axis=1)
        * 255
    ).astype(np.uint8)

    if save_to_wandb:
        print(f"Saving {name} to wandb")
        wandb.log({name: wandb.Video(wandb_video, fps=fps, format="mp4")})

    return wandb_video


# %%
if cfg.visualize_data:
    visualize_nerf_videos(
        example_batch_data.left_nerf_densities,
        idx_to_visualize=0,
        name="Left Nerf Densities Video",
        save_to_wandb=True,
    )

# %%
if cfg.visualize_data:
    visualize_nerf_videos(
        example_batch_data.right_nerf_densities,
        idx_to_visualize=0,
        name="Right Nerf Densities Video",
        save_to_wandb=True,
    )


# %%
@localscope.mfc(allowed=["NUM_PTS_X", "NUM_PTS_Y", "NUM_PTS_Z"])
def visualize_1D_max_nerf_density(
    left_nerf_densities: torch.Tensor,
    right_nerf_densities: torch.Tensor,
    idx_to_visualize: int = 0,
    name: str = "Max Nerf Density",
    save_to_wandb: bool = False,
) -> plt.Figure:
    # Create 1D visualization
    left_nerf_density = left_nerf_densities[idx_to_visualize]
    right_nerf_density = right_nerf_densities[idx_to_visualize]
    assert left_nerf_density.shape == (NUM_PTS_X // 2, NUM_PTS_Y, NUM_PTS_Z)
    assert right_nerf_density.shape == (NUM_PTS_X // 2, NUM_PTS_Y, NUM_PTS_Z)

    # Visualize nerf densities
    left_nerf_density = (
        left_nerf_density.cpu().numpy().transpose(0, 2, 1)
    )  # Tranpose because z is last, but should be height
    right_nerf_density = (
        right_nerf_density.cpu().numpy().transpose(0, 2, 1)
    )  # Tranpose because z is last, but should be height
    num_imgs, _, _ = left_nerf_density.shape

    # Get maxes from both sides
    left_max_density = left_nerf_density.max(axis=(1, 2))
    right_max_density = right_nerf_density.max(axis=(1, 2))
    left_range = range(len(left_max_density))
    right_range = range(
        len(left_max_density), len(left_max_density) + len(right_max_density)
    )
    max_density = np.concatenate(
        [left_max_density, right_max_density[::-1]]
    )  # Reverse right side so they align

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(left_range, left_max_density, label="left finger")
    ax.plot(right_range, right_max_density[::-1], label="right finger")
    ax.set_title(name)
    ax.set_xlabel(f"Idx (Left = 0, Right = {len(max_density) - 1})")
    ax.set_ylabel(name)
    ax.legend()
    fig.tight_layout()

    if save_to_wandb:
        print(f"Saving {name} to wandb")
        wandb.log({name: fig})

    return fig


# %%
if cfg.visualize_data:
    visualize_1D_max_nerf_density(
        example_batch_data.left_nerf_densities,
        example_batch_data.right_nerf_densities,
        idx_to_visualize=0,
        name="Max Nerf Density",
        save_to_wandb=True,
    )

# %%
print(f"Train loader size: {len(train_loader)}")
print(f"Val loader size: {len(val_loader)}")
print(f"Test loader size: {len(test_loader)}")

# %%
assert math.ceil(len(train_dataset) / cfg.dataloader.batch_size) == len(train_loader)
assert math.ceil(len(val_dataset) / cfg.dataloader.batch_size) == len(val_loader)
assert math.ceil(len(test_dataset) / cfg.dataloader.batch_size) == len(test_loader)

# %% [markdown]
# # Visualize Datapoint

# %%


class Phase(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


# %%
@localscope.mfc
def wandb_log_plotly_fig(
    plotly_fig: go.Figure, title: str, group_name: str = "plotly"
) -> None:
    if wandb.run is None:
        print("Not logging plotly fig to wandb because wandb.run is None")
        return

    path_to_plotly_html = f"{wandb.run.dir}/{title}.html"
    print(f"Saving to {path_to_plotly_html}")

    plotly_fig.write_html(path_to_plotly_html)
    wandb_table = wandb.Table(columns=[title])
    wandb_table.add_data(wandb.Html(path_to_plotly_html))
    if group_name is not None:
        wandb.log({f"{group_name}/{title}": wandb_table})
    else:
        wandb.log({title: wandb_table})
    print(f"Successfully logged {title} to wandb")


# %%
@localscope.mfc
def get_origin_lines() -> List[go.Scatter3d]:
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


@localscope.mfc(allowed=["NUM_XYZ"])
def get_colored_points_scatter(
    points: torch.Tensor, colors: torch.Tensor
) -> go.Scatter3d:
    assert len(points.shape) == 2 and points.shape[1] == NUM_XYZ
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
@localscope.mfc(
    allowed=[
        "INPUT_EXAMPLE_SHAPE",
        "NUM_DENSITY",
        "NUM_PTS_X",
        "NUM_PTS_Y",
        "NUM_PTS_Z",
        "NERF_DENSITY_START_IDX",
        "NERF_DENSITY_END_IDX",
        "NUM_XYZ",
        "NERF_COORDINATE_START_IDX",
        "NERF_COORDINATE_END_IDX",
    ]
)
def create_datapoint_plotly_fig(
    dataset: NeRFGrid_To_GraspSuccess_HDF5_Dataset,
    datapoint_name: str,
    idx_to_visualize: int = 0,
    save_to_wandb: bool = False,
) -> go.Figure:
    nerf_grid_input, grasp_success = dataset[idx_to_visualize]

    assert nerf_grid_input.shape == INPUT_EXAMPLE_SHAPE
    assert len(np.array(grasp_success).shape) == 0

    nerf_densities = nerf_grid_input[
        NERF_DENSITY_START_IDX:NERF_DENSITY_END_IDX, :, :, :
    ]
    assert nerf_densities.shape == (NUM_DENSITY, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z)

    nerf_points = nerf_grid_input[
        NERF_COORDINATE_START_IDX:NERF_COORDINATE_END_IDX
    ].permute(1, 2, 3, 0)
    assert nerf_points.shape == (NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z, NUM_XYZ)

    isaac_origin_lines = get_origin_lines()
    colored_points_scatter = get_colored_points_scatter(
        nerf_points.reshape(-1, NUM_XYZ), nerf_densities.reshape(-1)
    )

    layout = go.Layout(
        scene=dict(xaxis=dict(title="X"), yaxis=dict(title="Y"), zaxis=dict(title="Z")),
        showlegend=True,
        title=f"{datapoint_name} datapoint: success={grasp_success}",
        width=800,
        height=800,
    )

    # Create the figure
    fig = go.Figure(layout=layout)
    for line in isaac_origin_lines:
        fig.add_trace(line)
    fig.add_trace(colored_points_scatter)
    fig.update_layout(legend_orientation="h")

    if save_to_wandb:
        print(f"Saving {datapoint_name} datapoint to wandb")
        wandb_log_plotly_fig(plotly_fig=fig, title=f"{datapoint_name}_datapoint")

    return fig


# %%
if cfg.visualize_data:
    create_datapoint_plotly_fig(
        dataset=train_dataset,
        datapoint_name=Phase.TRAIN.name.lower(),
        save_to_wandb=True,
    )

# %%
if cfg.visualize_data:
    create_datapoint_plotly_fig(
        dataset=val_dataset, datapoint_name=Phase.VAL.name.lower(), save_to_wandb=True
    )


# %%
@localscope.mfc
def create_plotly_mesh(
    obj_filepath, scale=1.0, offset=None, color="lightpink"
) -> go.Mesh3d:
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

    return mesh


# %%
@localscope.mfc
def create_detailed_plot_with_mesh(
    full_dataset: NeRFGrid_To_GraspSuccess_HDF5_Dataset,
    idx_to_visualize: int = 0,
    save_to_wandb: bool = False,
) -> go.Figure:
    # Hacky function that reads from both the input dataset and the acronym dataset
    # To create a detailed plot with the mesh and the grasp
    fig = create_datapoint_plotly_fig(
        dataset=full_dataset,
        datapoint_name="full data",
        save_to_wandb=False,
        idx_to_visualize=idx_to_visualize,
    )

    ACRONYM_ROOT_DIR = "/juno/u/tylerlum/github_repos/acronym/data/grasps"
    MESH_ROOT_DIR = "assets/objects"
    LEFT_TIP_POSITION_GRASP_FRAME = np.array(
        [4.10000000e-02, -7.27595772e-12, 1.12169998e-01]
    )
    RIGHT_TIP_POSITION_GRASP_FRAME = np.array(
        [-4.10000000e-02, -7.27595772e-12, 1.12169998e-01]
    )

    # Get acronym filename and grasp transform from input dataset
    with h5py.File(full_dataset.input_hdf5_filepath, "r") as hdf5_file:
        acronym_filename = hdf5_file["/acronym_filename"][idx_to_visualize].decode(
            "utf-8"
        )
        grasp_transform = np.array(hdf5_file["/grasp_transform"][idx_to_visualize])

    # Get mesh info from acronym dataset
    acronym_filepath = os.path.join(ACRONYM_ROOT_DIR, acronym_filename)
    with h5py.File(acronym_filepath, "r") as acronym_hdf5_file:
        mesh_filename = acronym_hdf5_file["object/file"][()].decode("utf-8")
        mesh_filepath = os.path.join(MESH_ROOT_DIR, mesh_filename)

        import trimesh

        mesh = trimesh.load(mesh_filepath, force="mesh")
        mesh_scale = float(acronym_hdf5_file["object/scale"][()])
        mesh_centroid = np.array(mesh.centroid) * mesh_scale

    left_tip = (
        np.matmul(
            grasp_transform, np.concatenate([LEFT_TIP_POSITION_GRASP_FRAME, [1.0]])
        )[:3]
        - mesh_centroid
    )
    right_tip = (
        np.matmul(
            grasp_transform, np.concatenate([RIGHT_TIP_POSITION_GRASP_FRAME, [1.0]])
        )[:3]
        - mesh_centroid
    )

    # Draw mesh, ensure -centroid offset so that mesh centroid is centered at origin
    fig.add_trace(
        create_plotly_mesh(
            obj_filepath=mesh_filepath,
            scale=mesh_scale,
            offset=-mesh_centroid,
            color="lightpink",
        )
    )

    # Draw line from left_tip to right_tip
    fig.add_trace(
        go.Scatter3d(
            x=[left_tip[0], right_tip[0]],
            y=[left_tip[1], right_tip[1]],
            z=[left_tip[2], right_tip[2]],
            mode="lines",
            line=dict(color="red", width=10),
            name="Grasp (should align with query points)",
        )
    )

    if save_to_wandb:
        print(f"Saving detailed mesh plot to wandb")
        wandb_log_plotly_fig(
            plotly_fig=fig, title=f"Detailed Mesh Plot idx={idx_to_visualize}"
        )

    return fig


# %%
if cfg.visualize_data:
    create_datapoint_plotly_fig(
        dataset=full_dataset,
        datapoint_name="full",
        idx_to_visualize=0,
        save_to_wandb=True,
    )

# %%
if cfg.visualize_data:
    create_detailed_plot_with_mesh(
        full_dataset=full_dataset, idx_to_visualize=0, save_to_wandb=True
    )


# %%
@localscope.mfc(allowed=["cfg"])
def create_before_and_after_invariance_transformations_fig(
    train_dataset: Subset,
    batch_idx_to_visualize: int = 0,
    save_to_wandb: bool = False,
) -> go.Figure:
    # Load data without invariance transformations
    no_invariance_transforms_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        pin_memory=cfg.dataloader.pin_memory,
        num_workers=cfg.dataloader.num_workers,
        collate_fn=custom_collate_fn_without_invariance_transformations,
    )
    example_batch_data: BatchData = next(iter(no_invariance_transforms_loader))

    before_left_global_params = torch.chunk(
        example_batch_data.left_global_params, chunks=3, dim=1
    )
    before_right_global_params = torch.chunk(
        example_batch_data.right_global_params, chunks=3, dim=1
    )

    assert len(before_left_global_params) == len(before_right_global_params) == 3

    # Run invariance transformations
    after_left_global_params, after_right_global_params = invariance_transformation(
        left_global_params=before_left_global_params,
        right_global_params=before_right_global_params,
        rotate_polar_angle=False,
        reflect_around_xz_plane_randomly=True,
        reflect_around_xy_plane_randomly=False,
        remove_y_axis=False,
    )

    # Create the figure
    title = f"Before and After Invariance Transformations idx={batch_idx_to_visualize}"
    layout = go.Layout(
        scene=dict(xaxis=dict(title="X"), yaxis=dict(title="Y"), zaxis=dict(title="Z")),
        showlegend=True,
        title=title,
        width=800,
        height=800,
    )

    fig = go.Figure(layout=layout)
    for line in get_origin_lines():
        fig.add_trace(line)

    (
        before_left_origin,
        before_left_x_axis,
        before_left_y_axis,
    ) = before_left_global_params
    right_origin, right_x_axis, right_y_axis = before_right_global_params
    after_left_origin, after_left_x_axis, after_left_y_axis = after_left_global_params
    (
        after_right_origin,
        after_right_x_axis,
        after_right_y_axis,
    ) = after_right_global_params

    @localscope.mfc
    def create_line(origin, axis, name, color, length=0.02):
        return go.Scatter3d(
            x=[origin[0], origin[0] + axis[0] * length],
            y=[origin[1], origin[1] + axis[1] * length],
            z=[origin[2], origin[2] + axis[2] * length],
            mode="lines",
            line=dict(width=2, color=color),
            name=name,
        )

    # Get the idx to visualize
    before_left_origin = before_left_origin[batch_idx_to_visualize].cpu().numpy()
    before_left_x_axis = before_left_x_axis[batch_idx_to_visualize].cpu().numpy()
    before_left_y_axis = before_left_y_axis[batch_idx_to_visualize].cpu().numpy()
    right_origin = right_origin[batch_idx_to_visualize].cpu().numpy()
    right_x_axis = right_x_axis[batch_idx_to_visualize].cpu().numpy()
    right_y_axis = right_y_axis[batch_idx_to_visualize].cpu().numpy()
    after_left_origin = after_left_origin[batch_idx_to_visualize].cpu().numpy()
    after_left_x_axis = after_left_x_axis[batch_idx_to_visualize].cpu().numpy()
    after_left_y_axis = after_left_y_axis[batch_idx_to_visualize].cpu().numpy()
    after_right_origin = after_right_origin[batch_idx_to_visualize].cpu().numpy()
    after_right_x_axis = after_right_x_axis[batch_idx_to_visualize].cpu().numpy()
    after_right_y_axis = after_right_y_axis[batch_idx_to_visualize].cpu().numpy()

    # Draw before and after lines, wrt origin
    fig.add_trace(
        create_line(
            before_left_origin, before_left_x_axis, "Before Left X Axis", "black"
        )
    )
    fig.add_trace(
        create_line(
            before_left_origin, before_left_y_axis, "Before Left Y Axis", "black"
        )
    )
    fig.add_trace(
        create_line(right_origin, right_x_axis, "Before Right X Axis", "magenta")
    )
    fig.add_trace(
        create_line(right_origin, right_y_axis, "Before Right Y Axis", "magenta")
    )
    fig.add_trace(
        create_line(after_left_origin, after_left_x_axis, "After Left X Axis", "gray")
    )
    fig.add_trace(
        create_line(after_left_origin, after_left_y_axis, "After Left Y Axis", "gray")
    )
    fig.add_trace(
        create_line(
            after_right_origin, after_right_x_axis, "After Right X Axis", "olive"
        )
    )
    fig.add_trace(
        create_line(
            after_right_origin, after_right_y_axis, "After Right Y Axis", "olive"
        )
    )

    if save_to_wandb:
        print(f"Saving {title} to wandb")
        wandb.log({title: fig})

    return fig


# %%
if cfg.visualize_data:
    for i in range(3):
        create_before_and_after_invariance_transformations_fig(
            train_dataset=train_dataset, batch_idx_to_visualize=i, save_to_wandb=True
        )


# %% [markdown]
# # Visualize Dataset Distribution


# %%
@localscope.mfc
def create_grasp_success_distribution_fig(
    train_dataset: Subset, input_dataset_full_path: str, save_to_wandb: bool = False
) -> Optional[plt.Figure]:
    try:
        print("Loading grasp success data for grasp success distribution...")
        t1 = time.time()
        with h5py.File(input_dataset_full_path, "r") as hdf5_file:
            grasp_successes_np = np.array(hdf5_file["/grasp_success"][()])
        t2 = time.time()
        print(f"Loaded grasp success data in {t2 - t1:.2f} s")

        print("Extracting training indices...")
        t3 = time.time()
        grasp_successes_np = grasp_successes_np[train_dataset.indices]
        t4 = time.time()
        print(f"Extracted training indices in {t4 - t3:.2f} s")

        # Plot histogram
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.hist(grasp_successes_np, bins=100)
        ax.set_title("Distribution of Grasp Successes")
        ax.set_xlabel("Grasp Success")
        ax.set_ylabel("Frequency")

        if save_to_wandb:
            print("Saving grasp success distribution to wandb")
            wandb.log({"Distribution of Grasp Successes": fig})

        return fig

    except Exception as e:
        print(f"Error: {e}")
        print("Skipping visualization of grasp success distribution")


if cfg.visualize_data:
    create_grasp_success_distribution_fig(
        train_dataset=train_dataset,
        input_dataset_full_path=input_dataset_full_path,
        save_to_wandb=True,
    )


# %%
@localscope.mfc(
    allowed=[
        "INPUT_EXAMPLE_SHAPE",
        "NERF_COORDINATE_START_IDX",
        "NERF_COORDINATE_END_IDX",
        "NERF_DENSITY_START_IDX",
        "NERF_DENSITY_END_IDX",
        "tqdm",
    ]
)
def create_batch_input_distribution_fig(
    train_loader: DataLoader,
    save_to_wandb: bool = False,
    max_num_batches: Optional[int] = None,
) -> plt.Figure:
    nerf_densities, nerf_global_params = [], []
    max_num_batches = (
        max_num_batches if max_num_batches is not None else len(train_loader)
    )
    max_num_batches = min(
        max_num_batches, len(train_loader)
    )  # Ensure max_num_batches is not too large

    for batch_idx, batch_data in tqdm(
        enumerate(train_loader),
        desc="Calculating batch input dataset statistics",
        total=max_num_batches,
    ):
        if batch_idx >= max_num_batches:
            break

        batch_data: BatchData = batch_data
        nerf_densities.extend(batch_data.left_nerf_densities.flatten().tolist())
        nerf_densities.extend(batch_data.right_nerf_densities.flatten().tolist())
        nerf_global_params.extend(batch_data.left_global_params.flatten().tolist())
        nerf_global_params.extend(batch_data.right_global_params.flatten().tolist())

    nerf_densities, nerf_global_params = np.array(nerf_densities), np.array(
        nerf_global_params
    )
    print(f"nerf_density_min: {nerf_densities.min()}")
    print(f"nerf_density_mean: {nerf_densities.mean()}")
    print(f"nerf_density_max: {nerf_densities.max()}")
    print(f"nerf_global_param_min: {nerf_global_params.min()}")
    print(f"nerf_global_param_mean: {nerf_global_params.mean()}")
    print(f"nerf_global_param_max: {nerf_global_params.max()}")

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes = axes.flatten()
    axes[0].hist(nerf_densities, bins=100)
    axes[0].set_title("Distribution of nerf_densities")
    axes[0].set_xlabel("Density")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(nerf_global_params, bins=100)
    axes[1].set_title("Distribution of nerf_global_params")
    axes[1].set_xlabel("Global Param")
    axes[1].set_ylabel("Frequency")

    title = f"Distribution of Inputs ({max_num_batches} of {len(train_loader)} batches)"
    fig.suptitle(title)

    fig.tight_layout()

    if save_to_wandb:
        print("Saving batch input distribution to wandb")
        wandb.log({title: fig})

    return fig


# %%
if cfg.visualize_data:
    create_batch_input_distribution_fig(
        train_loader=train_loader,
        save_to_wandb=True,
        max_num_batches=50,
    )

# %% [markdown]
# # Create Neural Network Model

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
input_shape = example_batch_data.nerf_grid_inputs.shape[-3:]
conditioning_dim = example_batch_data.left_global_params.shape[1]
print(f"input_shape = {input_shape}")
print(f"conditioning_dim = {conditioning_dim}")

nerf_to_grasp_success_model = Condition2D1D_ConcatFingersAfter1D(
    input_shape=input_shape,
    n_fingers=2,
    conditioning_dim=conditioning_dim,
    # **dataclass_to_kwargs(cfg.classifier),
    conv_encoder_2d_config=cfg.classifier.conv_encoder_2d_config,
    use_conditioning_2d=cfg.classifier.use_conditioning_2d,
    conv_encoder_2d_embed_dim=cfg.classifier.conv_encoder_2d_embed_dim,
    conv_encoder_2d_mlp_hidden_layers=cfg.classifier.conv_encoder_2d_mlp_hidden_layers,
    conv_encoder_1d_config=cfg.classifier.conv_encoder_1d_config,
    transformer_encoder_1d_config=cfg.classifier.transformer_encoder_1d_config,
    encoder_1d_type=cfg.classifier.encoder_1d_type,
    use_conditioning_1d=cfg.classifier.use_conditioning_1d,
    head_mlp_hidden_layers=cfg.classifier.head_mlp_hidden_layers,
).to(device)

start_epoch = 0
optimizer = torch.optim.AdamW(
    params=nerf_to_grasp_success_model.parameters(),
    lr=cfg.training.lr,
    betas=cfg.training.betas,
    weight_decay=cfg.training.weight_decay,
)
lr_scheduler = get_scheduler(
    name=cfg.training.lr_scheduler_name,
    optimizer=optimizer,
    num_warmup_steps=cfg.training.lr_scheduler_num_warmup_steps,
    num_training_steps=(
        len(train_loader) * cfg.training.n_epochs
    ),
    last_epoch=start_epoch - 1,
)


# %% [markdown]
# # Load Checkpoint

# %%
checkpoint = load_checkpoint(checkpoint_workspace_dir_path)
if checkpoint is not None:
    print("Loading checkpoint...")
    nerf_to_grasp_success_model.load_state_dict(
        checkpoint["nerf_to_grasp_success_model"]
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    start_epoch = checkpoint["epoch"]
    print("Done loading checkpoint")


# %% [markdown]
# # Visualize Neural Network Model

# %%
print(f"nerf_to_grasp_success_model = {nerf_to_grasp_success_model}")
print(f"optimizer = {optimizer}")
print(f"lr_scheduler = {lr_scheduler}")

# %%
example_batch_data = example_batch_data.to(device)

summary(
    nerf_to_grasp_success_model,
    input_data=[example_batch_data.nerf_grid_inputs, example_batch_data.global_params],
    device=device,
    depth=5,
)

# %%
example_batch_nerf_grid_inputs, example_batch_global_params = (
    example_batch_data.nerf_grid_inputs,
    example_batch_data.global_params,
)
example_batch_nerf_grid_inputs = example_batch_nerf_grid_inputs.requires_grad_(True).to(
    device
)
example_batch_global_params = example_batch_global_params.requires_grad_(True).to(
    device
)
example_grasp_success_predictions = nerf_to_grasp_success_model(
    example_batch_nerf_grid_inputs, conditioning=example_batch_global_params
)

if cfg.visualize_data:
    dot = None
    try:
        dot = make_dot(
            example_grasp_success_predictions,
            params={
                **dict(nerf_to_grasp_success_model.named_parameters()),
                **{"NERF INPUT": example_batch_nerf_grid_inputs},
                **{"GLOBAL PARAMS": example_batch_global_params},
                **{"GRASP SUCCESS": example_grasp_success_predictions},
            },
        )
        model_graph_filename = "model_graph.png"
        model_graph_filename_split = model_graph_filename.split(".")
        print(f"Saving to {model_graph_filename}...")
        dot.render(model_graph_filename_split[0], format=model_graph_filename_split[1])
        print(f"Done saving to {model_graph_filename}")
    except:
        print("Failed to save model graph to file.")

    dot


# %% [markdown]
# # Training Setup


# %%
@localscope.mfc
def save_checkpoint(
    checkpoint_workspace_dir_path: str,
    epoch: int,
    nerf_to_grasp_success_model: Condition2D1D_ConcatFingersAfter1D,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    checkpoint_filepath = os.path.join(
        checkpoint_workspace_dir_path, f"checkpoint_{epoch:04}.pt"
    )
    print(f"Saving checkpoint to {checkpoint_filepath}")
    torch.save(
        {
            "epoch": epoch,
            "nerf_to_grasp_success_model": nerf_to_grasp_success_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        },
        checkpoint_filepath,
    )
    print("Done saving checkpoint")


# %%


@localscope.mfc
def create_dataloader_subset(
    original_dataloader: DataLoader,
    fraction: Optional[float] = None,
    subset_size: Optional[int] = None,
) -> DataLoader:
    if fraction is not None and subset_size is None:
        smaller_dataset_size = int(len(original_dataloader.dataset) * fraction)
    elif fraction is None and subset_size is not None:
        smaller_dataset_size = subset_size
    else:
        raise ValueError(f"Must specify either fraction or subset_size")

    sampled_indices = random.sample(
        range(len(original_dataloader.dataset.indices)), smaller_dataset_size
    )
    dataloader = DataLoader(
        original_dataloader.dataset,
        batch_size=original_dataloader.batch_size,
        sampler=SubsetRandomSampler(
            sampled_indices,
        ),
        pin_memory=original_dataloader.pin_memory,
        num_workers=original_dataloader.num_workers,
        collate_fn=custom_collate_fn,
    )

    return dataloader


@localscope.mfc(allowed=["tqdm"])
def iterate_through_dataloader(
    phase: Phase,
    dataloader: DataLoader,
    nerf_to_grasp_success_model: Condition2D1D_ConcatFingersAfter1D,
    device: str,
    ce_loss_fn: nn.CrossEntropyLoss,
    wandb_log_dict: Dict[str, Any],
    cfg: Optional[TrainingConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    log_loss: bool = True,
    log_grad: bool = False,
    gather_predictions: bool = False,
    log_confusion_matrix: bool = False,
    log_each_batch: bool = False,
) -> None:
    assert phase in [Phase.TRAIN, Phase.VAL, Phase.TEST]
    if phase == Phase.TRAIN:
        nerf_to_grasp_success_model.train()
        assert cfg is not None and optimizer is not None

    else:
        nerf_to_grasp_success_model.eval()
        assert cfg is None and optimizer is None

    with torch.set_grad_enabled(phase == Phase.TRAIN):
        losses_dict = defaultdict(list)
        grads_dict = defaultdict(list)

        batch_total_time_taken = 0.0
        dataload_total_time_taken = 0.0
        forward_pass_total_time_taken = 0.0
        backward_pass_total_time_taken = 0.0
        grad_log_total_time_taken = 0.0
        loss_log_total_time_taken = 0.0
        gather_predictions_total_time_taken = 0.0
        log_each_batch_total_time_taken = 0.0

        all_predictions, all_ground_truths = [], []

        end_time = time.time()
        for batch_data in (pbar := tqdm(dataloader)):
            dataload_time_taken = time.time() - end_time

            # Forward pass
            start_forward_pass_time = time.time()
            batch_data: BatchData = batch_data
            batch_data = batch_data.to(device)

            grasp_success_logits = nerf_to_grasp_success_model.get_success_logits(
                batch_data.nerf_grid_inputs, conditioning=batch_data.global_params
            )
            ce_loss = ce_loss_fn(
                input=grasp_success_logits, target=batch_data.grasp_successes
            )
            total_loss = ce_loss
            forward_pass_time_taken = time.time() - start_forward_pass_time

            # Gradient step
            start_backward_pass_time = time.time()
            if phase == Phase.TRAIN and optimizer is not None:
                optimizer.zero_grad()
                total_loss.backward()

                if cfg is not None and cfg.grad_clip_val is not None:
                    torch.nn.utils.clip_grad_value_(
                        nerf_to_grasp_success_model.parameters(),
                        cfg.grad_clip_val,
                    )

                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
            backward_pass_time_taken = time.time() - start_backward_pass_time

            # Loss logging
            start_loss_log_time = time.time()
            if log_loss:
                losses_dict[f"{phase.name.lower()}_loss"].append(total_loss.item())
            loss_log_time_taken = time.time() - start_loss_log_time

            # Gradient logging
            start_grad_log_time = time.time()
            if phase == Phase.TRAIN and log_grad:
                with torch.no_grad():  # not sure if need this
                    grad_abs_values = torch.concat(
                        [
                            p.grad.data.abs().flatten()
                            for p in nerf_to_grasp_success_model.parameters()
                            if p.grad is not None and p.requires_grad
                        ]
                    )
                    grads_dict[f"{phase.name.lower()}_max_grad_abs_value"].append(
                        torch.max(grad_abs_values).item()
                    )
                    grads_dict[f"{phase.name.lower()}_median_grad_abs_value"].append(
                        torch.median(grad_abs_values).item()
                    )
                    grads_dict[f"{phase.name.lower()}_mean_grad_abs_value"].append(
                        torch.mean(grad_abs_values).item()
                    )
                    grads_dict[f"{phase.name.lower()}_mean_grad_norm_value"].append(
                        torch.norm(grad_abs_values).item()
                    )
            grad_log_time_taken = time.time() - start_grad_log_time

            # Gather predictions and ground truths
            start_gather_predictions_time = time.time()
            if gather_predictions:
                with torch.no_grad():
                    predictions = grasp_success_logits.argmax(axis=1).tolist()
                    ground_truths = batch_data.grasp_successes.tolist()
                    all_predictions = all_predictions + predictions
                    all_ground_truths = all_ground_truths + ground_truths
            gather_predictions_time_taken = time.time() - start_gather_predictions_time

            # Log each batch
            start_log_batch_time = time.time()
            if log_each_batch:
                batch_log_dict = {}
                if optimizer is not None:
                    batch_log_dict[f"batch_{phase.name.lower()}_lr"] = optimizer.param_groups[0].['lr']
                if lr_scheduler is not None:
                    batch_log_dict[f"batch_{phase.name.lower()}_scheduler_lr"] = lr_scheduler.get_last_lr()[0]
                for loss_name, losses in losses_dict.items():
                    if len(losses) == 0:
                        continue
                    batch_log_dict[f"batch_{loss_name}"] = losses[-1]

                if len(all_predictions) > 0 and len(all_ground_truths) > 0:
                    # Can add more metrics here
                    batch_log_dict[f"batch_{phase.name.lower()}_accuracy"] = 100.0 * accuracy_score(
                        y_true=all_ground_truths, y_pred=all_predictions
                    )

                # Extra debugging
                for grad_name, grad_vals in grads_dict.items():
                    if len(grad_vals) == 0:
                        continue
                    if "_max_" in grad_name:
                        batch_log_dict[f"batch_{grad_name}"] = np.max(grad_vals)
                    elif "_mean_" in grad_name:
                        batch_log_dict[f"batch_{grad_name}"] = np.mean(grad_vals)
                    elif "_median_" in grad_name:
                        batch_log_dict[f"batch_{grad_name}"] = np.median(grad_vals)
                    else:
                        print(f"WARNING: grad_name = {grad_name} will not be logged")

                if len(batch_log_dict) > 0:
                    wandb.log(batch_log_dict)
            log_batch_time_taken = time.time() - start_log_batch_time

            batch_time_taken = time.time() - end_time

            # Set description
            loss_log_str = (
                f"loss: {np.mean(losses_dict[f'{phase.name.lower()}_loss']):.5f}"
                if len(losses_dict[f"{phase.name.lower()}_loss"]) > 0
                else "loss: N/A"
            )
            description = " | ".join(
                [
                    f"{phase.name.lower()} (ms)",
                    f"Batch: {1000*batch_time_taken:.0f}",
                    f"Data: {1000*dataload_time_taken:.0f}",
                    f"Fwd: {1000*forward_pass_time_taken:.0f}",
                    f"Bwd: {1000*backward_pass_time_taken:.0f}",
                    f"Loss: {1000*loss_log_time_taken:.0f}",
                    f"Grad: {1000*grad_log_time_taken:.0f}",
                    f"Gather: {1000*gather_predictions_time_taken:.0f}",
                    f"Log: {1000*log_batch_time_taken:.0f}",
                    loss_log_str,
                ]
            )
            pbar.set_description(description)

            batch_total_time_taken += batch_time_taken
            dataload_total_time_taken += dataload_time_taken
            forward_pass_total_time_taken += forward_pass_time_taken
            backward_pass_total_time_taken += backward_pass_time_taken
            loss_log_total_time_taken += loss_log_time_taken
            grad_log_total_time_taken += grad_log_time_taken
            gather_predictions_total_time_taken += gather_predictions_time_taken
            log_each_batch_total_time_taken += log_batch_time_taken

            end_time = time.time()

    print(
        f"Total time taken for {phase.name.lower()} phase: {batch_total_time_taken:.2f} s"
    )
    print(f"Time taken for dataload: {dataload_total_time_taken:.2f} s")
    print(f"Time taken for forward pass: {forward_pass_total_time_taken:.2f} s")
    print(f"Time taken for backward pass: {backward_pass_total_time_taken:.2f} s")
    print(f"Time taken for loss logging: {loss_log_total_time_taken:.2f} s")
    print(f"Time taken for grad logging: {grad_log_total_time_taken:.2f} s")
    print(
        f"Time taken for gather predictions: {gather_predictions_total_time_taken:.2f} s"
    )
    print(f"Time taken for log each batch: {log_each_batch_total_time_taken:.2f} s")
    print()

    # In percentage of batch_total_time_taken
    print("In percentage of batch_total_time_taken:")
    print(f"dataload: {100*dataload_total_time_taken/batch_total_time_taken:.2f} %")
    print(
        f"forward pass: {100*forward_pass_total_time_taken/batch_total_time_taken:.2f} %"
    )
    print(
        f"backward pass: {100*backward_pass_total_time_taken/batch_total_time_taken:.2f} %"
    )
    print(f"loss logging: {100*loss_log_total_time_taken/batch_total_time_taken:.2f} %")
    print(f"grad logging: {100*grad_log_total_time_taken/batch_total_time_taken:.2f} %")
    print(
        f"gather predictions: {100*gather_predictions_total_time_taken/batch_total_time_taken:.2f} %"
    )
    print(f"log each batch: {100*log_each_batch_total_time_taken/batch_total_time_taken:.2f} %")
    print()
    print()

    if log_confusion_matrix and len(all_predictions) > 0 and len(all_ground_truths) > 0:
        wandb_log_dict[
            f"{phase.name.lower()}_confusion_matrix"
        ] = wandb.plot.confusion_matrix(
            preds=all_predictions,
            y_true=all_ground_truths,
            class_names=["Fail", "Success"],
            title=f"{phase.name.lower()} Confusion Matrix",
        )

    if optimizer is not None:
        wandb_log_dict[f"{phase.name.lower()}_lr"] = optimizer.param_groups[0].['lr']
    if lr_scheduler is not None:
        wandb_log_dict[f"{phase.name.lower()}_scheduler_lr"] = lr_scheduler.get_last_lr()[0]

    for loss_name, losses in losses_dict.items():
        wandb_log_dict[loss_name] = np.mean(losses)

    if len(all_predictions) > 0 and len(all_ground_truths) > 0:
        # Can add more metrics here
        wandb_log_dict[f"{phase.name.lower()}_accuracy"] = 100.0 * accuracy_score(
            y_true=all_ground_truths, y_pred=all_predictions
        )

    # Extra debugging
    for grad_name, grad_vals in grads_dict.items():
        if "_max_" in grad_name:
            wandb_log_dict[grad_name] = np.max(grad_vals)
        elif "_mean_" in grad_name:
            wandb_log_dict[grad_name] = np.mean(grad_vals)
        elif "_median_" in grad_name:
            wandb_log_dict[grad_name] = np.median(grad_vals)
        else:
            print(f"WARNING: grad_name = {grad_name} will not be logged")

    return


# %% [markdown]
# # Training


# %%
@localscope.mfc(allowed=["tqdm"])
def run_training_loop(
    cfg: TrainingConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    nerf_to_grasp_success_model: Condition2D1D_ConcatFingersAfter1D,
    device: str,
    ce_loss_fn: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    start_epoch: int,
    checkpoint_workspace_dir_path: str,
) -> None:
    training_loop_base_description = "Training Loop"
    for epoch in (
        pbar := tqdm(
            range(start_epoch, cfg.n_epochs), desc=training_loop_base_description
        )
    ):
        wandb_log_dict = {}
        wandb_log_dict["epoch"] = epoch

        # Save checkpoint
        start_save_checkpoint_time = time.time()
        if epoch % cfg.save_checkpoint_freq == 0 and (
            epoch != 0 or cfg.save_checkpoint_on_epoch_0
        ):
            save_checkpoint(
                checkpoint_workspace_dir_path=checkpoint_workspace_dir_path,
                epoch=epoch,
                nerf_to_grasp_success_model=nerf_to_grasp_success_model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )
        save_checkpoint_time_taken = time.time() - start_save_checkpoint_time

        log_confusion_matrix = epoch % cfg.confusion_matrix_freq == 0 and (
            epoch != 0 or cfg.save_confusion_matrix_on_epoch_0
        )
        gather_predictions = log_confusion_matrix

        # Train
        start_train_time = time.time()
        log_grad = epoch % cfg.log_grad_freq == 0 and (
            epoch != 0 or cfg.log_grad_on_epoch_0
        )
        if cfg.use_dataloader_subset:
            # subset_fraction = 0.2
            # subset_train_loader = create_dataloader_subset(
            #     train_loader, fraction=subset_fraction
            # )
            # num_passes = int(1 / subset_fraction)
            subset_train_loader = create_dataloader_subset(
                train_loader,
                subset_size=32_000,  # 2023-04-28 each datapoint is 1MB
            )
            num_passes = 3
            for subset_pass in range(num_passes):
                print(f"Subset pass {subset_pass + 1}/{num_passes}")
                iterate_through_dataloader(
                    phase=Phase.TRAIN,
                    dataloader=subset_train_loader,
                    nerf_to_grasp_success_model=nerf_to_grasp_success_model,
                    device=device,
                    ce_loss_fn=ce_loss_fn,
                    wandb_log_dict=wandb_log_dict,
                    cfg=cfg,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    log_grad=log_grad,
                    gather_predictions=False,  # Doesn't make sense to gather predictions for a subset
                    log_confusion_matrix=False,
                    log_each_batch=True,
                )
        else:
            iterate_through_dataloader(
                phase=Phase.TRAIN,
                dataloader=train_loader,
                nerf_to_grasp_success_model=nerf_to_grasp_success_model,
                device=device,
                ce_loss_fn=ce_loss_fn,
                wandb_log_dict=wandb_log_dict,
                cfg=cfg,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                log_grad=log_grad,
                gather_predictions=gather_predictions,
                log_confusion_matrix=log_confusion_matrix,
                log_each_batch=True,
            )
        train_time_taken = time.time() - start_train_time

        # Val
        # Can do this before or after training (decided on after since before it was always at -ln(1/N_CLASSES) ~ 0.69)
        start_val_time = time.time()
        if epoch % cfg.val_freq == 0 and (epoch != 0 or cfg.val_on_epoch_0):
            iterate_through_dataloader(
                phase=Phase.VAL,
                dataloader=val_loader,
                nerf_to_grasp_success_model=nerf_to_grasp_success_model,
                device=device,
                ce_loss_fn=ce_loss_fn,
                wandb_log_dict=wandb_log_dict,
                gather_predictions=gather_predictions,
                log_confusion_matrix=log_confusion_matrix,
                log_each_batch=True,
            )
        val_time_taken = time.time() - start_val_time

        wandb.log(wandb_log_dict)

        # Set description
        description = " | ".join(
            [
                training_loop_base_description + " (s)",
                f"Save: {save_checkpoint_time_taken:.0f}",
                f"Train: {train_time_taken:.0f}",
                f"Val: {val_time_taken:.0f}",
            ]
        )
        pbar.set_description(description)


# %%
wandb.watch(nerf_to_grasp_success_model, log="gradients", log_freq=100)


# %%
@localscope.mfc
def compute_class_weight_np(
    train_dataset: Subset, input_dataset_full_path: str
) -> np.ndarray:
    try:
        print("Loading grasp success data for class weighting...")
        t1 = time.time()
        with h5py.File(input_dataset_full_path, "r") as hdf5_file:
            grasp_successes_np = np.array(hdf5_file["/grasp_success"][()])
        t2 = time.time()
        print(f"Loaded grasp success data in {t2 - t1:.2f} s")

        print("Extracting training indices...")
        t3 = time.time()
        grasp_successes_np = grasp_successes_np[train_dataset.indices]
        t4 = time.time()
        print(f"Extracted training indices in {t4 - t3:.2f} s")

        print("Computing class weight with this data...")
        t5 = time.time()
        class_weight_np = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(grasp_successes_np),
            y=grasp_successes_np,
        )
        t6 = time.time()
        print(f"Computed class weight in {t6 - t5:.2f} s")

    except Exception as e:
        print(f"Failed to compute class weight: {e}")
        print("Using default class weight")
        class_weight_np = np.array([1.0, 1.0])
    return class_weight_np


class_weight = (
    torch.from_numpy(
        compute_class_weight_np(
            train_dataset=train_dataset, input_dataset_full_path=input_dataset_full_path
        )
    )
    .float()
    .to(device)
)
print(f"Class weight: {class_weight}")
ce_loss_fn = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=cfg.training.label_smoothing)

# %%
run_training_loop(
    cfg=cfg.training,
    train_loader=train_loader,
    val_loader=val_loader,
    nerf_to_grasp_success_model=nerf_to_grasp_success_model,
    device=device,
    ce_loss_fn=ce_loss_fn,
    optimizer=optimizer,
    start_epoch=start_epoch,
    checkpoint_workspace_dir_path=checkpoint_workspace_dir_path,
)

# %% [markdown]
# # Test

# %%
nerf_to_grasp_success_model.eval()
wandb_log_dict = {}
print(f"Running test metrics on epoch {cfg.training.n_epochs}")
wandb_log_dict["epoch"] = cfg.training.n_epochs
iterate_through_dataloader(
    phase=Phase.TEST,
    dataloader=test_loader,
    nerf_to_grasp_success_model=nerf_to_grasp_success_model,
    device=device,
    ce_loss_fn=ce_loss_fn,
    wandb_log_dict=wandb_log_dict,
    gather_predictions=True,
    log_confusion_matrix=True,
    log_each_batch=True,
)

wandb.log(wandb_log_dict)

# %% [markdown]
# # Save Model

# %%
save_checkpoint(
    checkpoint_workspace_dir_path=checkpoint_workspace_dir_path,
    epoch=cfg.training.n_epochs,
    nerf_to_grasp_success_model=nerf_to_grasp_success_model,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
)

# %%
