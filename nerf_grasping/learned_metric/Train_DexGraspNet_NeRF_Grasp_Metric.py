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
# # Imports

# %%
from __future__ import annotations
import trimesh
import time
from collections import defaultdict
import functools
from localscope import localscope
import nerf_grasping
from dataclasses import dataclass
from torchinfo import summary
from torchviz import make_dot
from nerf_grasping.dataset.DexGraspNet_NeRF_Grasps_utils import (
    DIST_BTWN_PTS_MM,
    get_query_points_finger_frame,
    get_contact_candidates_and_target_candidates,
    get_start_and_end_and_up_points,
    get_transform,
    get_scene_dict,
    get_transformed_points,
    get_nerf_densities,
    plot_mesh_and_query_points,
    plot_mesh_and_transforms,
    get_object_code,
    get_object_scale,
    load_nerf,
    NUM_PTS_X,
    NUM_PTS_Y,
    NUM_PTS_Z,
    GRASP_DEPTH_MM,
    FINGER_WIDTH_MM,
    FINGER_HEIGHT_MM,
    NUM_FINGERS,
)
from nerf_grasping.dataset.timers import LoopTimer
from nerf_grasping.learned_metric.Train_DexGraspNet_NeRF_Grasp_Metric_Config import (
    Config,
    TrainingConfig,
)
import os
import h5py
from typing import Optional, Tuple, List
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)


import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Subset,
    Dataset,
    random_split,
)
from sklearn.utils.class_weight import compute_class_weight
import plotly.graph_objects as go
import wandb
from functools import partial
from omegaconf import OmegaConf
from datetime import datetime
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
import sys
from hydra.utils import instantiate
import random
import shutil
from wandb.util import generate_id

from enum import Enum, auto
from nerf_grasping.models.tyler_new_models import get_scheduler


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
NUM_XYZ = 3


class Phase(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


# %%
@functools.lru_cache()
def get_query_points_finger_frame_cached() -> np.ndarray:
    query_points_finger_frame = get_query_points_finger_frame()
    assert query_points_finger_frame.shape == (NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z, NUM_XYZ)
    return query_points_finger_frame

# %%
if is_notebook():
    from tqdm.notebook import tqdm as std_tqdm
else:
    from tqdm import tqdm as std_tqdm

tqdm = partial(std_tqdm, dynamic_ncols=True)

# %% [markdown]
# # Setup Config for Static Type-Checking


# %%
@functools.lru_cache
@localscope.mfc
def datetime_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("datetime_str", datetime_str, replace=True)

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
    with initialize(
        version_base="1.1", config_path="Train_DexGraspNet_NeRF_Grasp_Metric_cfg"
    ):
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
class NeRFGrid_To_GraspSuccess_HDF5_Dataset(Dataset):
    # @localscope.mfc  # ValueError: Cell is empty
    def __init__(
        self,
        input_hdf5_filepath: str,
        max_num_data_points: Optional[int] = None,
        load_nerf_densities_in_ram: bool = False,
        load_grasp_successes_in_ram: bool = True,
        load_grasp_transforms_in_ram: bool = True,
        load_nerf_workspaces_in_ram: bool = True,
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
            assert (
                len(hdf5_file["/grasp_success"].shape) == 1
            ), f"{hdf5_file['/grasp_success'].shape}"
            assert (
                hdf5_file["/nerf_densities"].shape[1:] == (NUM_FINGERS, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z)
            ), f"{hdf5_file['/nerf_densities'].shape}"
            assert hdf5_file["/grasp_transforms"].shape[1:] == (
                NUM_FINGERS,
                4,
                4,
            ), f"{hdf5_file['/grasp_transforms'].shape}"

            # This is usually too big for RAM
            self.nerf_densities = (
                torch.from_numpy(hdf5_file["/nerf_densities"][()]).float()
                if load_nerf_densities_in_ram
                else None
            )

            # This is small enough to fit in RAM
            self.grasp_successes = (
                torch.from_numpy(hdf5_file["/grasp_success"][()]).long()
                if load_grasp_successes_in_ram
                else None
            )

            # This is small enough to fit in RAM
            self.grasp_transforms = (
                torch.from_numpy(hdf5_file["/grasp_transforms"][()]).float()
                if load_grasp_transforms_in_ram
                else None
            )

            # This is small enough to fit in RAM
            self.nerf_workspaces = (
                hdf5_file["/nerf_workspace"][()]
                if load_nerf_workspaces_in_ram
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
            "NUM_FINGERS",
            "NUM_PTS_X",
            "NUM_PTS_Y",
            "NUM_PTS_Z",
        ]
    )
    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        if self.hdf5_file is None:
            # Hope to speed up with rdcc params
            self.hdf5_file = h5py.File(
                self.input_hdf5_filepath,
                "r",
                rdcc_nbytes=1024**2 * 4_000,
                rdcc_w0=0.75,
                rdcc_nslots=4_000,
            )

        nerf_densities = (
            torch.from_numpy(self.hdf5_file["/nerf_densities"][idx]).float()
            if self.nerf_densities is None
            else self.nerf_densities[idx]
        )

        grasp_success = (
            torch.from_numpy(np.array(self.hdf5_file["/grasp_success"][idx])).long()
            if self.grasp_successes is None
            else self.grasp_successes[idx]
        )

        grasp_transforms = (
            torch.from_numpy(np.array(self.hdf5_file["/grasp_transforms"][idx])).float()
            if self.grasp_transforms is None
            else self.grasp_transforms[idx]
        )

        nerf_workspace = (
            self.hdf5_file["/nerf_workspace"][idx]
            if self.nerf_workspaces is None
            else self.nerf_workspaces[idx]
        ).decode("utf-8")

        assert nerf_densities.shape == (NUM_FINGERS, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z)
        assert grasp_success.shape == ()
        assert grasp_transforms.shape == (NUM_FINGERS, 4, 4)

        return nerf_densities, grasp_success, grasp_transforms, nerf_workspace


# %%

input_dataset_full_path = os.path.join(
    nerf_grasping.get_repo_root(),
    cfg.data.input_dataset_root_dir,
    cfg.data.input_dataset_path,
)
full_dataset = NeRFGrid_To_GraspSuccess_HDF5_Dataset(
    input_hdf5_filepath=input_dataset_full_path,
    max_num_data_points=cfg.data.max_num_data_points,
    load_nerf_densities_in_ram=cfg.dataloader.load_nerf_grid_inputs_in_ram,
    load_grasp_successes_in_ram=cfg.dataloader.load_grasp_successes_in_ram,
    load_grasp_transforms_in_ram=cfg.dataloader.load_grasp_transforms_in_ram,
    load_nerf_workspaces_in_ram=cfg.dataloader.load_nerf_workspaces_in_ram,
)

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
@dataclass
class BatchData:
    nerf_densities: torch.Tensor
    grasp_success: torch.Tensor
    grasp_transforms: torch.Tensor
    nerf_workspace: List[str]

    @localscope.mfc
    def to(self, device) -> BatchData:
        self.nerf_densities = self.nerf_densities.to(device)
        self.grasp_success = self.grasp_success.to(device)
        self.grasp_transforms = self.grasp_transforms.to(device)
        return self

    # @localscope.mfc(allowed=["DIST_BTWN_PTS_MM"])
    @property
    def nerf_alphas(self) -> torch.Tensor:
        # alpha = 1 - exp(-delta * sigma)
        #       = probability of collision within this segment starting from beginning of segment
        DELTA = DIST_BTWN_PTS_MM / 1000
        return 1.0 - torch.exp(-DELTA * self.nerf_densities)

    # @localscope.mfc(allowed=["NUM_FINGERS"])
    @property
    def coords(self) -> torch.Tensor:
        # TODO: Change this to not be np and be vectorized
        query_points_finger_frame = get_query_points_finger_frame_cached().reshape(-1, NUM_XYZ)
        all_query_points_object_frame = []
        for i in range(self.batch_size):
            transforms = self.grasp_transforms[i]
            query_points_object_frame = torch.stack(
                [
                    torch.from_numpy(
                        get_transformed_points(
                            points=query_points_finger_frame,
                            transform=transforms[finger_idx].cpu().numpy(),
                        )
                    )
                    for finger_idx in range(NUM_FINGERS)
                ],
                dim=0,
            )
            all_query_points_object_frame.append(query_points_object_frame)
        all_query_points_object_frame = torch.stack(all_query_points_object_frame, dim=0).float().to(self.device)
        assert all_query_points_object_frame.shape == (
            self.batch_size,
            NUM_FINGERS,
            NUM_PTS_X * NUM_PTS_Y * NUM_PTS_Z,
            NUM_XYZ,
        )
        all_query_points_object_frame = all_query_points_object_frame.reshape(
            self.batch_size, NUM_FINGERS, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z, NUM_XYZ
        )
        all_query_points_object_frame = all_query_points_object_frame.permute(
            0, 1, 5, 2, 3, 4
        )
        assert all_query_points_object_frame.shape == (
            self.batch_size,
            NUM_FINGERS,
            NUM_XYZ,
            NUM_PTS_X,
            NUM_PTS_Y,
            NUM_PTS_Z,
        )
        return all_query_points_object_frame

    @property
    def nerf_alphas_with_coords(self) -> torch.Tensor:
        reshaped_nerf_alphas = self.nerf_alphas.reshape(self.batch_size, NUM_FINGERS, 1, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z)
        return_value = torch.cat(
            [
                reshaped_nerf_alphas,
                self.coords,
            ],
            dim=2,
        )
        assert return_value.shape == (
            self.batch_size,
            NUM_FINGERS,
            NUM_XYZ + 1,
            NUM_PTS_X,
            NUM_PTS_Y,
            NUM_PTS_Z,
        )
        return return_value

    @property
    def batch_size(self) -> int:
        return self.grasp_success.shape[0]

    @property
    def device(self) -> torch.device:
        return self.grasp_success.device


@localscope.mfc
def custom_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]
) -> BatchData:
    batch = torch.utils.data.dataloader.default_collate(batch)
    nerf_densities, grasp_successes, grasp_transforms, nerf_workspaces = batch

    return BatchData(
        nerf_densities=nerf_densities,
        grasp_success=grasp_successes,
        grasp_transforms=grasp_transforms,
        nerf_workspace=nerf_workspaces,
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
@localscope.mfc
def print_shapes(batch_data: BatchData) -> None:
    print(f"nerf_alphas.shape: {batch_data.nerf_alphas.shape}")
    print(f"grasp_success.shape: {batch_data.grasp_success.shape}")
    print(f"grasp_transforms.shape: {batch_data.grasp_transforms.shape}")
    print(f"len(nerf_workspace): {len(batch_data.nerf_workspace)}")
    print(f"coords.shape = {batch_data.coords.shape}")
    print(f"nerf_alphas_with_coords.shape = {batch_data.nerf_alphas_with_coords.shape}")


EXAMPLE_BATCH_DATA = next(iter(val_loader))
print_shapes(batch_data=EXAMPLE_BATCH_DATA)

# %% [markdown]
# # Visualize Data


# %%


@localscope.mfc(allowed=["NUM_FINGERS", "NUM_PTS_X", "NUM_PTS_Y", "NUM_PTS_Z", "NUM_XYZ"])
def plot_example(batch_data: BatchData, idx_to_visualize: int = 0) -> go.Figure:
    # Extract data
    grasp_transforms = batch_data.grasp_transforms[idx_to_visualize]
    colors = batch_data.nerf_alphas[idx_to_visualize]
    grasp_success = batch_data.grasp_success[idx_to_visualize].item()

    assert grasp_transforms.shape == (NUM_FINGERS, 4, 4)
    assert colors.shape == (NUM_FINGERS, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z)
    assert grasp_success in [0, 1]

    _, nerf_workspace = os.path.split(batch_data.nerf_workspace[idx_to_visualize])
    object_code = get_object_code(nerf_workspace)
    object_scale = get_object_scale(nerf_workspace)

    # Path to meshes
    DEXGRASPNET_DATA_ROOT = "/juno/u/tylerlum/github_repos/DexGraspNet/data"
    DEXGRASPNET_MESHDATA_ROOT = os.path.join(DEXGRASPNET_DATA_ROOT, "meshdata")
    mesh_path = os.path.join(
        DEXGRASPNET_MESHDATA_ROOT,
        object_code,
        "coacd",
        "decomposed.obj",
    )

    print(f"Loading mesh from {mesh_path}...")
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.apply_transform(trimesh.transformations.scale_matrix(object_scale))

    # Get query points from grasp_transforms
    query_points_list = batch_data.coords[idx_to_visualize]
    assert query_points_list.shape == (
        NUM_FINGERS,
        NUM_XYZ,
        NUM_PTS_X,
        NUM_PTS_Y,
        NUM_PTS_Z,
    )
    query_points_list = query_points_list.permute((0, 2, 3, 4, 1))
    assert query_points_list.shape == (
        NUM_FINGERS,
        NUM_PTS_X,
        NUM_PTS_Y,
        NUM_PTS_Z,
        NUM_XYZ,
    )
    query_points_list = [
        query_points_list[finger_idx].reshape(-1, NUM_XYZ).cpu().numpy()
        for finger_idx in range(NUM_FINGERS)
    ]
    query_point_colors_list = [
        colors[finger_idx].reshape(-1).cpu().numpy() for finger_idx in range(NUM_FINGERS)
    ]
    fig = plot_mesh_and_query_points(
        mesh=mesh,
        query_points_list=query_points_list,
        query_points_colors_list=query_point_colors_list,
        num_fingers=NUM_FINGERS,
    )
    # Set title to grasp_success
    fig.update_layout(title_text=f"grasp_success = {grasp_success}")
    return fig


fig = plot_example(batch_data=EXAMPLE_BATCH_DATA, idx_to_visualize=15)
fig.show()

# Try augment dataset with random rotation 
# HOw do they feed into network?
# Finger boxes (local)
# Finger box location (global)

# %%
EXAMPLE_BATCH_DATA.grasp_success

# %%
fig = plot_example(batch_data=EXAMPLE_BATCH_DATA, idx_to_visualize=14)
fig.show()

# %%
fig = plot_example(batch_data=EXAMPLE_BATCH_DATA, idx_to_visualize=17)
fig.show()

# %%
fig = plot_example(batch_data=EXAMPLE_BATCH_DATA, idx_to_visualize=18)
fig.show()

# %%
fig = plot_example(batch_data=EXAMPLE_BATCH_DATA, idx_to_visualize=19)
fig.show()

# %% [markdown]
# # Create Neural Network Model

# %%
import torch.nn as nn
from nerf_grasping.models.tyler_new_models import (
    ConvOutputTo1D,
    PoolType,
    conv_encoder,
    mlp,
)


class CNN_3D_Classifier(nn.Module):
    # @localscope.mfc
    def __init__(self, input_shape: Tuple[int, int, int, int], n_fingers) -> None:
        # TODO: Make this not hardcoded
        super().__init__()
        self.input_shape = input_shape
        self.n_fingers = n_fingers

        self.conv = conv_encoder(
            input_shape=self.input_shape,
            conv_channels=[32, 64, 128],
            pool_type=PoolType.MAX,
            dropout_prob=0.1,
            conv_output_to_1d=ConvOutputTo1D.AVG_POOL_SPATIAL,
        )

        # Get conv output shape
        example_batch_size = 2
        example_input = torch.zeros(
            example_batch_size, self.n_fingers, *self.input_shape
        )
        example_input = example_input.reshape(
            example_batch_size * self.n_fingers, *self.input_shape
        )
        conv_output = self.conv(example_input)
        self.conv_output_dim = conv_output.shape[-1]
        assert conv_output.shape == (
            example_batch_size * self.n_fingers,
            self.conv_output_dim,
        )

        self.mlp = mlp(
            num_inputs=self.conv_output_dim * self.n_fingers,
            num_outputs=self.n_classes,
            hidden_layers=[256, 256],
        )

    @localscope.mfc
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        assert x.shape == (
            batch_size,
            self.n_fingers,
            *self.input_shape,
        ), f"{x.shape}"

        # Put n_fingers into batch dim
        x = x.reshape(batch_size * self.n_fingers, *self.input_shape)

        x = self.conv(x)
        assert x.shape == (
            batch_size * self.n_fingers,
            self.conv_output_dim,
        ), f"{x.shape}"
        x = x.reshape(batch_size, self.n_fingers * self.conv_output_dim)

        x = self.mlp(x)
        assert x.shape == (batch_size, self.n_classes), f"{x.shape}"
        return x

    @localscope.mfc
    def get_success_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    @localscope.mfc
    def get_success_probability(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.softmax(self.get_success_logits(x), dim=-1)

    @property
    @functools.lru_cache
    def n_classes(self) -> int:
        return 2


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SHAPE = (NUM_XYZ + 1, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z)
nerf_to_grasp_success_model = CNN_3D_Classifier(
    input_shape=INPUT_SHAPE,
    n_fingers=NUM_FINGERS,
).to(device)

# %%
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
    num_training_steps=(len(train_loader) * cfg.training.n_epochs),
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
    start_epoch = checkpoint["epoch"]
    if lr_scheduler is not None and "lr_scheduler" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    print("Done loading checkpoint")

# %% [markdown]
# # Visualize Model

# %%
print(f"nerf_to_grasp_success_model = {nerf_to_grasp_success_model}")
print(f"optimizer = {optimizer}")
print(f"lr_scheduler = {lr_scheduler}")

# %%
summary(
    model=nerf_to_grasp_success_model,
    input_size=(cfg.dataloader.batch_size, NUM_FINGERS, *INPUT_SHAPE),
    device=device,
)

# %%
example_input = (
    torch.zeros((cfg.dataloader.batch_size, NUM_FINGERS, *INPUT_SHAPE))
    .to(device)
    .requires_grad_(True)
)
example_output = nerf_to_grasp_success_model(example_input)
try:
    dot = make_dot(
        example_output,
        params={
            **dict(nerf_to_grasp_success_model.named_parameters()),
            **{"NERF_INPUT": example_input},
            **{"GRASP_SUCCESS": example_output},
        },
    )
    model_graph_filename = "model_graph.png"
    model_graph_filename_no_ext, model_graph_file_ext = model_graph_filename.split(".")
    print(f"Saving to {model_graph_filename}...")
    dot.render(model_graph_filename_no_ext, format=model_graph_file_ext)
    print(f"Done saving to {model_graph_filename}")
except Exception as e:
    print(f"Exception: {e}")
    print("Skipping make_dot")

SHOW_DOT = False
if SHOW_DOT:
    dot

# %% [markdown]
# # Training Setup


# %%
@localscope.mfc
def save_checkpoint(
    checkpoint_workspace_dir_path: str,
    epoch: int,
    nerf_to_grasp_success_model: CNN_3D_Classifier,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
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
            "lr_scheduler": lr_scheduler.state_dict()
            if lr_scheduler is not None
            else None,
        },
        checkpoint_filepath,
    )
    print("Done saving checkpoint")


# %%
@localscope.mfc(allowed=["tqdm"])
def iterate_through_dataloader(
    phase: Phase,
    dataloader: DataLoader,
    nerf_to_grasp_success_model: CNN_3D_Classifier,
    device: torch.device,
    ce_loss_fn: nn.CrossEntropyLoss,
    wandb_log_dict: dict,
    training_cfg: Optional[TrainingConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> None:
    assert phase in [Phase.TRAIN, Phase.VAL, Phase.TEST]
    if phase == Phase.TRAIN:
        nerf_to_grasp_success_model.train()
        assert training_cfg is not None and optimizer is not None
    else:
        nerf_to_grasp_success_model.eval()
        assert training_cfg is None and optimizer is None

    loop_timer = LoopTimer()
    with torch.set_grad_enabled(phase == Phase.TRAIN):
        losses_dict = defaultdict(list)
        all_predictions, all_ground_truths = [], []

        dataload_section_timer = loop_timer.add_section_timer("Data").start()
        for batch_idx, batch_data in (
            pbar := tqdm(enumerate(dataloader), total=len(dataloader))
        ):
            dataload_section_timer.stop()

            batch_idx = int(batch_idx)
            batch_data: BatchData = batch_data.to(device)

            # Forward pass
            with loop_timer.add_section_timer("Fwd"):
                grasp_success_logits = nerf_to_grasp_success_model.get_success_logits(
                    batch_data.nerf_alphas_with_coords
                )
                ce_loss = ce_loss_fn(
                    input=grasp_success_logits, target=batch_data.grasp_success
                )
                total_loss = ce_loss

            # Gradient step
            with loop_timer.add_section_timer("Bwd"):
                if phase == Phase.TRAIN and optimizer is not None:
                    optimizer.zero_grad()
                    total_loss.backward()

                    if (
                        training_cfg is not None
                        and training_cfg.grad_clip_val is not None
                    ):
                        torch.nn.utils.clip_grad_value_(
                            nerf_to_grasp_success_model.parameters(),
                            training_cfg.grad_clip_val,
                        )

                    optimizer.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()

            # Loss logging
            with loop_timer.add_section_timer("Loss"):
                losses_dict["loss"].append(total_loss.item())

            # Gather predictions
            with loop_timer.add_section_timer("Gather"):
                predictions = grasp_success_logits.argmax(dim=-1).tolist()
                ground_truths = batch_data.grasp_success.tolist()
                all_predictions += predictions
                all_ground_truths += ground_truths

            # Set description
            loss_log_str = (
                f"loss: {np.mean(losses_dict['loss']):.5f}"
                if len(losses_dict["loss"]) > 0
                else "loss: N/A"
            )
            description = " | ".join(
                [
                    f"{phase.name.lower()} (ms)",
                    loss_log_str,
                ]
            )
            pbar.set_description(description)

            if batch_idx < len(dataloader) - 1:
                # Avoid starting timer at end of last batch
                dataload_section_timer = loop_timer.add_section_timer("Data").start()

    if optimizer is not None:
        wandb_log_dict[f"{phase.name.lower()}_lr"] = optimizer.param_groups[0]["lr"]

    with loop_timer.add_section_timer("Agg Loss"):
        for loss_name, losses in losses_dict.items():
            wandb_log_dict[f"{phase.name.lower()}_{loss_name}"] = np.mean(losses)

    with loop_timer.add_section_timer("Metrics"):
        if len(all_ground_truths) > 0 and len(all_predictions) > 0:
            for name, function in [
                ("accuracy", accuracy_score),
                ("precision", precision_score),
                ("recall", recall_score),
                ("f1", f1_score),
            ]:
                wandb_log_dict[f"{phase.name.lower()}_{name}"] = function(
                    y_true=all_ground_truths, y_pred=all_predictions
                )
    with loop_timer.add_section_timer("Confusion Matrix"):
        if len(all_ground_truths) > 0 and len(all_predictions) > 0:
            wandb_log_dict[
                f"{phase.name.lower()}_confusion_matrix"
            ] = wandb.plot.confusion_matrix(
                preds=all_predictions,
                y_true=all_ground_truths,
                class_names=["failure", "success"],
                title=f"{phase.name.title()} Confusion Matrix",
            )

    loop_timer.pretty_print_section_times()
    print()
    print()

    return


@localscope.mfc(allowed=["tqdm"])
def run_training_loop(
    training_cfg: TrainingConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    nerf_to_grasp_success_model: CNN_3D_Classifier,
    device: torch.device,
    ce_loss_fn: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    checkpoint_workspace_dir_path: str,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    start_epoch: int = 0,
) -> None:
    training_loop_base_description = "Training Loop"
    for epoch in (
        pbar := tqdm(
            range(start_epoch, training_cfg.n_epochs),
            desc=training_loop_base_description,
        )
    ):
        epoch = int(epoch)
        wandb_log_dict = {}
        wandb_log_dict["epoch"] = epoch

        # Save checkpoint
        start_save_checkpoint_time = time.time()
        if epoch % training_cfg.save_checkpoint_freq == 0 and (
            epoch != 0 or training_cfg.save_checkpoint_on_epoch_0
        ):
            save_checkpoint(
                checkpoint_workspace_dir_path=checkpoint_workspace_dir_path,
                epoch=epoch,
                nerf_to_grasp_success_model=nerf_to_grasp_success_model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )
        save_checkpoint_time_taken = time.time() - start_save_checkpoint_time

        # Train
        start_train_time = time.time()
        iterate_through_dataloader(
            phase=Phase.TRAIN,
            dataloader=train_loader,
            nerf_to_grasp_success_model=nerf_to_grasp_success_model,
            device=device,
            ce_loss_fn=ce_loss_fn,
            wandb_log_dict=wandb_log_dict,
            training_cfg=training_cfg,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        train_time_taken = time.time() - start_train_time

        # Val
        # Can do this before or after training (decided on after since before it was always at -ln(1/N_CLASSES) ~ 0.69)
        start_val_time = time.time()
        if epoch % training_cfg.val_freq == 0 and (
            epoch != 0 or training_cfg.val_on_epoch_0
        ):
            iterate_through_dataloader(
                phase=Phase.VAL,
                dataloader=val_loader,
                nerf_to_grasp_success_model=nerf_to_grasp_success_model,
                device=device,
                ce_loss_fn=ce_loss_fn,
                wandb_log_dict=wandb_log_dict,
            )
        val_time_taken = time.time() - start_val_time

        if wandb.run is not None:
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
ce_loss_fn = nn.CrossEntropyLoss(
    weight=class_weight, label_smoothing=cfg.training.label_smoothing
)

# %%

run_training_loop(
    training_cfg=cfg.training,
    train_loader=train_loader,
    val_loader=val_loader,
    nerf_to_grasp_success_model=nerf_to_grasp_success_model,
    device=device,
    ce_loss_fn=ce_loss_fn,
    optimizer=optimizer,
    checkpoint_workspace_dir_path=checkpoint_workspace_dir_path,
    lr_scheduler=lr_scheduler,
    start_epoch=start_epoch,
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