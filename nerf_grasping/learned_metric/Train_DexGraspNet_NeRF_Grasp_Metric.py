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
import pathlib
import trimesh
import time
from collections import defaultdict
from localscope import localscope
import nerf_grasping
from dataclasses import asdict
from torchinfo import summary
from torchviz import make_dot
from nerf_grasping.learned_metric.DexGraspNet_batch_data import (
    BatchData,
    BatchDataInput,
    BatchDataOutput,
    DepthImageBatchDataInput,
)
from nerf_grasping.dataset.DexGraspNet_NeRF_Grasps_utils import (
    get_object_code,
    get_object_scale,
    plot_mesh_and_query_points,
    plot_mesh_and_transforms,
)
from nerf_grasping.classifier import Classifier, DepthImageClassifier
from nerf_grasping.dataset.timers import LoopTimer
from nerf_grasping.config.classifier_config import (
    UnionClassifierConfig,
    ClassifierConfig,
    ClassifierTrainingConfig,
    TaskType,
)
from nerf_grasping.config.fingertip_config import BaseFingertipConfig
from nerf_grasping.config.nerfdata_config import (
    DepthImageNerfDataConfig,
)
import os
import pypose as pp
import h5py
from typing import Optional, Tuple, List, Dict, Any, Union
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
from plotly.subplots import make_subplots
import wandb
from functools import partial
from datetime import datetime
import sys
import random
import shutil
from wandb.util import generate_id

from enum import Enum, auto
from nerf_grasping.models.tyler_new_models import get_scheduler
from nerf_grasping.config.base import CONFIG_DATETIME_STR

import tyro

os.chdir(nerf_grasping.get_repo_root())


# %%
def assert_equals(a, b):
    assert a == b, f"{a} != {b}"


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
# Make atol and rtol larger than default to avoid errors due to floating point precision.
# Otherwise we get errors about invalid rotation matrices
PP_MATRIX_ATOL, PP_MATRIX_RTOL = 1e-4, 1e-4

NUM_XYZ = 3


class Phase(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


# %%
if is_notebook():
    from tqdm.notebook import tqdm as std_tqdm
else:
    from tqdm import tqdm as std_tqdm

tqdm = partial(std_tqdm, dynamic_ncols=True)

# %% [markdown]
# # Setup Config for Static Type-Checking


# %% [markdown]
# # Load Config

# %%
if is_notebook():
    # arguments = [
    #     "cnn-2d-1d",
    #     "--task-type",
    #     "PASSED_SIMULATION_AND_PENETRATION_THRESHOLD",
    #     "--nerfdata-config.output-filepath",
    #     "data/2023-10-13_13-12-28/learned_metric_dataset/2023-11-13_12-19-14_learned_metric_dataset.h5",
    #     "nerfdata-config.fingertip-config:big-even",
    # ]
    arguments = [
        "depth-cnn-2d",
        "--task-type",
        "PASSED_SIMULATION_AND_PENETRATION_THRESHOLD",
        "--nerfdata-config.output-filepath",
        "data/2023-10-13_13-12-28/learned_metric_dataset/2023-11-13_12-19-14_learned_metric_dataset.h5",
        "nerfdata-config.fingertip-config:big-even",
    ]
else:
    arguments = sys.argv[1:]
    print(f"arguments = {arguments}")

# %%
cfg: ClassifierConfig = tyro.cli(UnionClassifierConfig, args=arguments)
assert cfg.nerfdata_config.fingertip_config is not None

USE_DEPTH_IMAGES = isinstance(cfg.nerfdata_config, DepthImageNerfDataConfig)
if USE_DEPTH_IMAGES:
    DEPTH_IMAGE_N_CHANNELS = 2
    DEPTH_IMAGE_HEIGHT = cfg.nerfdata_config.fingertip_camera_config.H
    DEPTH_IMAGE_WIDTH = cfg.nerfdata_config.fingertip_camera_config.W
else:
    DEPTH_IMAGE_N_CHANNELS = None
    DEPTH_IMAGE_HEIGHT = None
    DEPTH_IMAGE_WIDTH = None

# A relatively dirty hack: create script globals from the config vars.
NUM_FINGERS = cfg.nerfdata_config.fingertip_config.n_fingers
NUM_PTS_X = cfg.nerfdata_config.fingertip_config.num_pts_x
NUM_PTS_Y = cfg.nerfdata_config.fingertip_config.num_pts_y
NUM_PTS_Z = cfg.nerfdata_config.fingertip_config.num_pts_z


# %%
print(f"Config:\n{tyro.extras.to_yaml(cfg)}")

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

run = wandb.init(
    entity=cfg.wandb.entity,
    project=cfg.wandb.project,
    name=cfg.wandb.name_with_date,
    group=cfg.wandb.group,
    job_type=cfg.wandb.job_type,
    config=asdict(cfg),
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
        load_grasp_labels_in_ram: bool = True,
        load_grasp_transforms_in_ram: bool = True,
        load_nerf_configs_in_ram: bool = True,
        load_grasp_configs_in_ram: bool = True,
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
            assert_equals(len(hdf5_file["/passed_simulation"].shape), 1)
            assert_equals(len(hdf5_file["/passed_penetration_threshold"].shape), 1)
            assert_equals(len(hdf5_file["/passed_eval"].shape), 1)
            assert_equals(
                hdf5_file["/nerf_densities"].shape[1:],
                (
                    NUM_FINGERS,
                    NUM_PTS_X,
                    NUM_PTS_Y,
                    NUM_PTS_Z,
                ),
            )
            assert_equals(
                hdf5_file["/grasp_transforms"].shape[1:],
                (
                    NUM_FINGERS,
                    4,
                    4,
                ),
            )
            assert_equals(
                hdf5_file["/grasp_configs"].shape[1:], (NUM_FINGERS, 7 + 16 + 4)
            )

            # This is usually too big for RAM
            self.nerf_densities = (
                torch.from_numpy(hdf5_file["/nerf_densities"][()]).float()
                if load_nerf_densities_in_ram
                else None
            )

            # This is small enough to fit in RAM
            self.passed_simulations = (
                torch.from_numpy(hdf5_file["/passed_simulation"][()]).float()
                if load_grasp_labels_in_ram
                else None
            )
            self.passed_penetration_thresholds = (
                torch.from_numpy(hdf5_file["/passed_penetration_threshold"][()]).float()
                if load_grasp_labels_in_ram
                else None
            )
            self.passed_evals = (
                torch.from_numpy(hdf5_file["/passed_eval"][()]).float()
                if load_grasp_labels_in_ram
                else None
            )

            # This is small enough to fit in RAM
            self.grasp_transforms = (
                pp.from_matrix(
                    torch.from_numpy(hdf5_file["/grasp_transforms"][()]).float(),
                    pp.SE3_type,
                    atol=PP_MATRIX_ATOL,
                    rtol=PP_MATRIX_RTOL,
                )
                if load_grasp_transforms_in_ram
                else None
            )

            # This is small enough to fit in RAM
            self.nerf_configs = (
                hdf5_file["/nerf_config"][()] if load_nerf_configs_in_ram else None
            )

            self.grasp_configs = (
                torch.from_numpy(hdf5_file["/grasp_configs"][()]).float()
                if load_grasp_configs_in_ram
                else None
            )

    @localscope.mfc
    def _set_length(
        self, hdf5_file: h5py.File, max_num_data_points: Optional[int]
    ) -> int:
        length = (
            hdf5_file.attrs["num_data_points"]
            if "num_data_points" in hdf5_file.attrs
            else hdf5_file["/passed_simulation"].shape[0]
        )
        if length != hdf5_file["/passed_simulation"].shape[0]:
            print(
                f"WARNING: num_data_points = {length} != passed_simulation.shape[0] = {hdf5_file['/passed_simulation'].shape[0]}"
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
    def __getitem__(self, idx: int):
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

        passed_simulation = (
            torch.from_numpy(
                np.array(self.hdf5_file["/passed_simulation"][idx])
            ).float()
            if self.passed_simulations is None
            else self.passed_simulations[idx]
        )
        passed_penetration_threshold = (
            torch.from_numpy(
                np.array(self.hdf5_file["/passed_penetration_threshold"][idx])
            ).float()
            if self.passed_penetration_thresholds is None
            else self.passed_penetration_thresholds[idx]
        )
        passed_eval = (
            torch.from_numpy(np.array(self.hdf5_file["/passed_eval"][idx])).float()
            if self.passed_evals is None
            else self.passed_evals[idx]
        )
        assert_equals(passed_simulation.shape, ())
        assert_equals(passed_penetration_threshold.shape, ())
        assert_equals(passed_eval.shape, ())

        # TODO: Consider thresholding passed_X labels to be 0 or 1
        # Convert to float classes (N,) -> (N, 2)
        passed_simulation = torch.stack(
            [1 - passed_simulation, passed_simulation], dim=-1
        )
        passed_penetration_threshold = torch.stack(
            [1 - passed_penetration_threshold, passed_penetration_threshold], dim=-1
        )
        passed_eval = torch.stack([1 - passed_eval, passed_eval], dim=-1)

        grasp_transforms = (
            torch.from_numpy(np.array(self.hdf5_file["/grasp_transforms"][idx])).float()
            if self.grasp_transforms is None
            else self.grasp_transforms[idx]
        )

        nerf_config = (
            self.hdf5_file["/nerf_config"][idx]
            if self.nerf_configs is None
            else self.nerf_configs[idx]
        ).decode("utf-8")

        grasp_configs = (
            torch.from_numpy(np.array(self.hdf5_file["/grasp_configs"][idx])).float()
            if self.grasp_configs is None
            else self.grasp_configs[idx]
        )

        assert_equals(
            nerf_densities.shape, (NUM_FINGERS, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z)
        )
        NUM_CLASSES = 2
        assert_equals(passed_simulation.shape, (NUM_CLASSES,))
        assert_equals(passed_penetration_threshold.shape, (NUM_CLASSES,))
        assert_equals(passed_eval.shape, (NUM_CLASSES,))
        assert_equals(grasp_transforms.shape, (NUM_FINGERS, 4, 4))
        assert_equals(grasp_configs.shape, (NUM_FINGERS, 7 + 16 + 4))

        return (
            nerf_densities,
            passed_simulation,
            passed_penetration_threshold,
            passed_eval,
            grasp_transforms,
            nerf_config,
            grasp_configs,
        )


# %%
class DepthImage_To_GraspSuccess_HDF5_Dataset(Dataset):
    # @localscope.mfc  # ValueError: Cell is empty
    def __init__(
        self,
        input_hdf5_filepath: str,
        max_num_data_points: Optional[int] = None,
        load_depth_images_in_ram: bool = False,
        load_grasp_labels_in_ram: bool = True,
        load_grasp_transforms_in_ram: bool = True,
        load_nerf_configs_in_ram: bool = True,
        load_grasp_configs_in_ram: bool = True,
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
            assert_equals(len(hdf5_file["/passed_simulation"].shape), 1)
            assert_equals(len(hdf5_file["/passed_penetration_threshold"].shape), 1)
            assert_equals(len(hdf5_file["/passed_eval"].shape), 1)
            assert_equals(
                hdf5_file["/depth_images"].shape[1:],
                (
                    NUM_FINGERS,
                    DEPTH_IMAGE_HEIGHT,
                    DEPTH_IMAGE_WIDTH,
                ),
            )
            assert_equals(
                hdf5_file["/uncertainty_images"].shape[1:],
                (
                    NUM_FINGERS,
                    DEPTH_IMAGE_HEIGHT,
                    DEPTH_IMAGE_WIDTH,
                ),
            )
            assert_equals(
                hdf5_file["/grasp_transforms"].shape[1:],
                (
                    NUM_FINGERS,
                    4,
                    4,
                ),
            )
            assert_equals(
                hdf5_file["/grasp_configs"].shape[1:], (NUM_FINGERS, 7 + 16 + 4)
            )

            # This is usually too big for RAM
            self.depth_uncertainty_images = (
                torch.stack(
                    [
                        torch.from_numpy(hdf5_file["/depth_images"][()]).float(),
                        torch.from_numpy(hdf5_file["/uncertainty_images"][()]).float(),
                    ],
                    dim=-3,
                )
                if load_depth_images_in_ram
                else None
            )

            # This is small enough to fit in RAM
            self.passed_simulations = (
                torch.from_numpy(hdf5_file["/passed_simulation"][()]).float()
                if load_grasp_labels_in_ram
                else None
            )
            self.passed_penetration_thresholds = (
                torch.from_numpy(hdf5_file["/passed_penetration_threshold"][()]).float()
                if load_grasp_labels_in_ram
                else None
            )
            self.passed_evals = (
                torch.from_numpy(hdf5_file["/passed_eval"][()]).float()
                if load_grasp_labels_in_ram
                else None
            )

            # This is small enough to fit in RAM
            self.grasp_transforms = (
                pp.from_matrix(
                    torch.from_numpy(hdf5_file["/grasp_transforms"][()]).float(),
                    pp.SE3_type,
                    atol=PP_MATRIX_ATOL,
                    rtol=PP_MATRIX_RTOL,
                )
                if load_grasp_transforms_in_ram
                else None
            )

            # This is small enough to fit in RAM
            self.nerf_configs = (
                hdf5_file["/nerf_config"][()] if load_nerf_configs_in_ram else None
            )

            self.grasp_configs = (
                torch.from_numpy(hdf5_file["/grasp_configs"][()]).float()
                if load_grasp_configs_in_ram
                else None
            )

    @localscope.mfc
    def _set_length(
        self, hdf5_file: h5py.File, max_num_data_points: Optional[int]
    ) -> int:
        length = (
            hdf5_file.attrs["num_data_points"]
            if "num_data_points" in hdf5_file.attrs
            else hdf5_file["/passed_simulation"].shape[0]
        )
        if length != hdf5_file["/passed_simulation"].shape[0]:
            print(
                f"WARNING: num_data_points = {length} != passed_simulation.shape[0] = {hdf5_file['/passed_simulation'].shape[0]}"
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
            "DEPTH_IMAGE_N_CHANNELS",
            "DEPTH_IMAGE_HEIGHT",
            "DEPTH_IMAGE_WIDTH",
        ]
    )
    def __getitem__(self, idx: int):
        if self.hdf5_file is None:
            # Hope to speed up with rdcc params
            self.hdf5_file = h5py.File(
                self.input_hdf5_filepath,
                "r",
                rdcc_nbytes=1024**2 * 4_000,
                rdcc_w0=0.75,
                rdcc_nslots=4_000,
            )

        depth_uncertainty_images = (
            torch.stack(
                [
                    torch.from_numpy(self.hdf5_file["/depth_images"][idx]).float(),
                    torch.from_numpy(
                        self.hdf5_file["/uncertainty_images"][idx]
                    ).float(),
                ],
                dim=-3,
            )
            if self.depth_uncertainty_images is None
            else self.depth_uncertainty_images[idx]
        )

        passed_simulation = (
            torch.from_numpy(
                np.array(self.hdf5_file["/passed_simulation"][idx])
            ).float()
            if self.passed_simulations is None
            else self.passed_simulations[idx]
        )
        passed_penetration_threshold = (
            torch.from_numpy(
                np.array(self.hdf5_file["/passed_penetration_threshold"][idx])
            ).float()
            if self.passed_penetration_thresholds is None
            else self.passed_penetration_thresholds[idx]
        )
        passed_eval = (
            torch.from_numpy(np.array(self.hdf5_file["/passed_eval"][idx])).float()
            if self.passed_evals is None
            else self.passed_evals[idx]
        )
        assert_equals(passed_simulation.shape, ())
        assert_equals(passed_penetration_threshold.shape, ())
        assert_equals(passed_eval.shape, ())

        # TODO: Consider thresholding passed_X labels to be 0 or 1
        # Convert to float classes (N,) -> (N, 2)
        passed_simulation = torch.stack(
            [1 - passed_simulation, passed_simulation], dim=-1
        )
        passed_penetration_threshold = torch.stack(
            [1 - passed_penetration_threshold, passed_penetration_threshold], dim=-1
        )
        passed_eval = torch.stack([1 - passed_eval, passed_eval], dim=-1)

        grasp_transforms = (
            torch.from_numpy(np.array(self.hdf5_file["/grasp_transforms"][idx])).float()
            if self.grasp_transforms is None
            else self.grasp_transforms[idx]
        )

        nerf_config = (
            self.hdf5_file["/nerf_config"][idx]
            if self.nerf_configs is None
            else self.nerf_configs[idx]
        ).decode("utf-8")

        grasp_configs = (
            torch.from_numpy(np.array(self.hdf5_file["/grasp_configs"][idx])).float()
            if self.grasp_configs is None
            else self.grasp_configs[idx]
        )

        assert_equals(
            depth_uncertainty_images.shape,
            (
                NUM_FINGERS,
                DEPTH_IMAGE_N_CHANNELS,
                DEPTH_IMAGE_HEIGHT,
                DEPTH_IMAGE_WIDTH,
            ),
        )
        NUM_CLASSES = 2
        assert_equals(passed_simulation.shape, (NUM_CLASSES,))
        assert_equals(passed_penetration_threshold.shape, (NUM_CLASSES,))
        assert_equals(passed_eval.shape, (NUM_CLASSES,))
        assert_equals(grasp_transforms.shape, (NUM_FINGERS, 4, 4))
        assert_equals(grasp_configs.shape, (NUM_FINGERS, 7 + 16 + 4))

        return (
            depth_uncertainty_images,
            passed_simulation,
            passed_penetration_threshold,
            passed_eval,
            grasp_transforms,
            nerf_config,
            grasp_configs,
        )


# %%

input_dataset_full_path = str(cfg.nerfdata_config.output_filepath)
if USE_DEPTH_IMAGES:
    full_dataset = DepthImage_To_GraspSuccess_HDF5_Dataset(
        input_hdf5_filepath=input_dataset_full_path,
        max_num_data_points=cfg.data.max_num_data_points,
        load_depth_images_in_ram=cfg.dataloader.load_nerf_grid_inputs_in_ram,
        load_grasp_labels_in_ram=cfg.dataloader.load_grasp_labels_in_ram,
        load_grasp_transforms_in_ram=cfg.dataloader.load_grasp_transforms_in_ram,
        load_nerf_configs_in_ram=cfg.dataloader.load_nerf_configs_in_ram,
    )
else:
    full_dataset = NeRFGrid_To_GraspSuccess_HDF5_Dataset(
        input_hdf5_filepath=input_dataset_full_path,
        max_num_data_points=cfg.data.max_num_data_points,
        load_nerf_densities_in_ram=cfg.dataloader.load_nerf_grid_inputs_in_ram,
        load_grasp_labels_in_ram=cfg.dataloader.load_grasp_labels_in_ram,
        load_grasp_transforms_in_ram=cfg.dataloader.load_grasp_transforms_in_ram,
        load_nerf_configs_in_ram=cfg.dataloader.load_nerf_configs_in_ram,
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
assert_equals(
    len(set.intersection(set(train_dataset.indices), set(val_dataset.indices))), 0
)
assert_equals(
    len(set.intersection(set(train_dataset.indices), set(test_dataset.indices))), 0
)
assert_equals(
    len(set.intersection(set(val_dataset.indices), set(test_dataset.indices))), 0
)


# %%


@localscope.mfc(allowed=["PP_MATRIX_ATOL", "PP_MATRIX_RTOL"])
def sample_random_rotate_transforms(N: int) -> pp.LieTensor:
    # Sample big rotations in tangent space of SO(3).
    # Choose 4 * \pi as a heuristic to get pretty evenly spaced rotations.
    # TODO(pculbert): Figure out better uniform sampling on SO(3).
    log_random_rotations = pp.so3(4 * torch.pi * (2 * torch.rand(N, 3) - 1))

    # Return exponentiated rotations.
    random_SO3_rotations = log_random_rotations.Exp()

    # A bit annoying -- need to cast SO(3) -> SE(3).
    random_rotate_transforms = pp.from_matrix(
        random_SO3_rotations.matrix(),
        pp.SE3_type,
        atol=PP_MATRIX_ATOL,
        rtol=PP_MATRIX_RTOL,
    )

    return random_rotate_transforms


@localscope.mfc(allowed=["PP_MATRIX_ATOL", "PP_MATRIX_RTOL"])
def custom_collate_fn(
    batch,
    fingertip_config: BaseFingertipConfig,
    use_random_rotations: bool = True,
    debug_shuffle_labels: bool = False,
    nerf_density_threshold_value: Optional[float] = None,
) -> BatchData:
    batch = torch.utils.data.dataloader.default_collate(batch)
    (
        nerf_densities,
        passed_simulation,
        passed_penetration_threshold,
        passed_eval,
        grasp_transforms,
        nerf_configs,
        grasp_configs,
    ) = batch

    if debug_shuffle_labels:
        shuffle_inds = torch.randperm(passed_simulation.shape[0])
        passed_simulation = passed_simulation[shuffle_inds]
        passed_penetration_threshold = passed_penetration_threshold[shuffle_inds]
        passed_eval = passed_eval[shuffle_inds]

    grasp_transforms = pp.from_matrix(
        grasp_transforms,
        pp.SE3_type,
        atol=PP_MATRIX_ATOL,
        rtol=PP_MATRIX_RTOL,
    )

    batch_size = nerf_densities.shape[0]
    if use_random_rotations:
        random_rotate_transform = sample_random_rotate_transforms(N=batch_size)

        # Apply random rotation to grasp config.
        # NOTE: hardcodes grasp_configs ordering
        wrist_pose = pp.SE3(grasp_configs[..., :7])
        joint_angles = grasp_configs[..., 7:23]
        grasp_orientations = pp.SO3(grasp_configs[..., 23:])

        wrist_pose = random_rotate_transform.unsqueeze(1) @ wrist_pose
        grasp_orientations = (
            random_rotate_transform.rotation().unsqueeze(1) @ grasp_orientations
        )

        grasp_configs = torch.cat(
            (wrist_pose.data, joint_angles, grasp_orientations.data), axis=-1
        )
    else:
        random_rotate_transform = None

    return BatchData(
        input=BatchDataInput(
            nerf_densities=nerf_densities,
            grasp_transforms=grasp_transforms,
            random_rotate_transform=random_rotate_transform,
            fingertip_config=fingertip_config,
            nerf_density_threshold_value=nerf_density_threshold_value,
            grasp_configs=grasp_configs,
        ),
        output=BatchDataOutput(
            passed_simulation=passed_simulation,
            passed_penetration_threshold=passed_penetration_threshold,
            passed_eval=passed_eval,
        ),
        nerf_config=nerf_configs,
    )


@localscope.mfc(allowed=["PP_MATRIX_ATOL", "PP_MATRIX_RTOL"])
def depth_image_custom_collate_fn(
    batch,
    fingertip_config: BaseFingertipConfig,
    use_random_rotations: bool = True,
    debug_shuffle_labels: bool = False,
    nerf_density_threshold_value: Optional[float] = None,
) -> BatchData:
    batch = torch.utils.data.dataloader.default_collate(batch)
    (
        depth_uncertainty_images,
        passed_simulation,
        passed_penetration_threshold,
        passed_eval,
        grasp_transforms,
        nerf_configs,
        grasp_configs,
    ) = batch

    if debug_shuffle_labels:
        shuffle_inds = torch.randperm(passed_simulation.shape[0])
        passed_simulation = passed_simulation[shuffle_inds]
        passed_penetration_threshold = passed_penetration_threshold[shuffle_inds]
        passed_eval = passed_eval[shuffle_inds]

    grasp_transforms = pp.from_matrix(
        grasp_transforms,
        pp.SE3_type,
        atol=PP_MATRIX_ATOL,
        rtol=PP_MATRIX_RTOL,
    )

    batch_size = depth_uncertainty_images.shape[0]
    if use_random_rotations:
        random_rotate_transform = sample_random_rotate_transforms(N=batch_size)

        # Apply random rotation to grasp config.
        # NOTE: hardcodes grasp_configs ordering
        wrist_pose = pp.SE3(grasp_configs[..., :7])
        joint_angles = grasp_configs[..., 7:23]
        grasp_orientations = pp.SO3(grasp_configs[..., 23:])

        wrist_pose = random_rotate_transform.unsqueeze(1) @ wrist_pose
        grasp_orientations = (
            random_rotate_transform.rotation().unsqueeze(1) @ grasp_orientations
        )

        grasp_configs = torch.cat(
            (wrist_pose.data, joint_angles, grasp_orientations.data), axis=-1
        )
    else:
        random_rotate_transform = None

    return BatchData(
        input=DepthImageBatchDataInput(
            depth_uncertainty_images=depth_uncertainty_images,
            grasp_transforms=grasp_transforms,
            random_rotate_transform=random_rotate_transform,
            fingertip_config=fingertip_config,
            nerf_density_threshold_value=nerf_density_threshold_value,
            grasp_configs=grasp_configs,
        ),
        output=BatchDataOutput(
            passed_simulation=passed_simulation,
            passed_penetration_threshold=passed_penetration_threshold,
            passed_eval=passed_eval,
        ),
        nerf_config=nerf_configs,
    )


# %%
if USE_DEPTH_IMAGES:
    train_collate_fn = partial(
        depth_image_custom_collate_fn,
        fingertip_config=cfg.nerfdata_config.fingertip_config,
        use_random_rotations=cfg.data.use_random_rotations,
        debug_shuffle_labels=cfg.data.debug_shuffle_labels,
        nerf_density_threshold_value=cfg.data.nerf_density_threshold_value,
    )
    val_test_collate_fn = partial(
        depth_image_custom_collate_fn,
        fingertip_config=cfg.nerfdata_config.fingertip_config,
        use_random_rotations=True,
        debug_shuffle_labels=cfg.data.debug_shuffle_labels,
        nerf_density_threshold_value=cfg.data.nerf_density_threshold_value,
    )
else:
    train_collate_fn = partial(
        custom_collate_fn,
        fingertip_config=cfg.nerfdata_config.fingertip_config,
        use_random_rotations=cfg.data.use_random_rotations,
        debug_shuffle_labels=cfg.data.debug_shuffle_labels,
        nerf_density_threshold_value=cfg.data.nerf_density_threshold_value,
    )
    val_test_collate_fn = partial(
        custom_collate_fn,
        use_random_rotations=False,
        fingertip_config=cfg.nerfdata_config.fingertip_config,
        nerf_density_threshold_value=cfg.data.nerf_density_threshold_value,
    )  # Run test over actual test transforms.

train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.dataloader.batch_size,
    shuffle=True,
    pin_memory=cfg.dataloader.pin_memory,
    num_workers=cfg.dataloader.num_workers,
    collate_fn=train_collate_fn,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.dataloader.batch_size,
    shuffle=False,
    pin_memory=cfg.dataloader.pin_memory,
    num_workers=cfg.dataloader.num_workers,
    collate_fn=val_test_collate_fn,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=cfg.dataloader.batch_size,
    shuffle=False,
    pin_memory=cfg.dataloader.pin_memory,
    num_workers=cfg.dataloader.num_workers,
    collate_fn=val_test_collate_fn,
)

if cfg.data.use_random_rotations:
    print("Using random rotations for training")
else:
    print("Not using random rotations for training")


# %%
@localscope.mfc
def print_shapes(batch_data: BatchData) -> None:
    if isinstance(batch_data.input, BatchDataInput):
        print(f"nerf_alphas.shape: {batch_data.input.nerf_alphas.shape}")
        print(f"coords.shape = {batch_data.input.coords.shape}")
        print(
            f"nerf_alphas_with_coords.shape = {batch_data.input.nerf_alphas_with_coords.shape}"
        )
    elif isinstance(batch_data.input, DepthImageBatchDataInput):
        print(
            f"depth_uncertainty_images.shape: {batch_data.input.depth_uncertainty_images.shape}"
        )
    else:
        raise ValueError(f"Unknown batch_data.input type: {type(batch_data.input)}")

    print(
        f"augmented_grasp_transforms.shape = {batch_data.input.augmented_grasp_transforms.shape}"
    )
    print(f"grasp_transforms.shape: {batch_data.input.grasp_transforms.shape}")
    print(f"passed_simulation.shape: {batch_data.output.passed_simulation.shape}")
    print(
        f"passed_penetration_threshold.shape: {batch_data.output.passed_penetration_threshold.shape}"
    )
    print(f"passed_eval.shape: {batch_data.output.passed_eval.shape}")
    print(f"len(nerf_config): {len(batch_data.nerf_config)}")


EXAMPLE_BATCH_DATA: BatchData = next(iter(val_loader))
print_shapes(batch_data=EXAMPLE_BATCH_DATA)

# %% [markdown]
# # Visualize Data


# %%
@localscope.mfc(
    allowed=["NUM_FINGERS", "NUM_PTS_X", "NUM_PTS_Y", "NUM_PTS_Z", "NUM_XYZ"]
)
def nerf_densities_plot_example(
    batch_data: BatchData, idx_to_visualize: int = 0, augmented: bool = False
) -> go.Figure:
    assert isinstance(batch_data.input, BatchDataInput)

    if augmented:
        query_points_list = batch_data.input.augmented_coords[idx_to_visualize]
        additional_mesh_transform = (
            batch_data.input.random_rotate_transform[idx_to_visualize]
            .matrix()
            .cpu()
            .numpy()
            if batch_data.input.random_rotate_transform is not None
            else None
        )
    else:
        query_points_list = batch_data.input.coords[idx_to_visualize]
        additional_mesh_transform = None

    # Extract data
    colors = batch_data.input.nerf_alphas[idx_to_visualize]
    passed_simulation = batch_data.output.passed_simulation[idx_to_visualize].tolist()
    passed_penetration_threshold = batch_data.output.passed_penetration_threshold[
        idx_to_visualize
    ].tolist()
    passed_eval = batch_data.output.passed_eval[idx_to_visualize].tolist()

    assert_equals(colors.shape, (NUM_FINGERS, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z))

    nerf_config_path = pathlib.Path(batch_data.nerf_config[idx_to_visualize])
    object_code = get_object_code(nerf_config_path)
    object_scale = get_object_scale(nerf_config_path)

    # Path to meshes
    DEXGRASPNET_DATA_ROOT = str(pathlib.Path(nerf_grasping.get_repo_root()) / "data")
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
    if additional_mesh_transform is not None:
        mesh.apply_transform(additional_mesh_transform)

    # Get query points from grasp_transforms
    assert_equals(
        query_points_list.shape,
        (
            NUM_FINGERS,
            NUM_XYZ,
            NUM_PTS_X,
            NUM_PTS_Y,
            NUM_PTS_Z,
        ),
    )
    query_points_list = query_points_list.permute((0, 2, 3, 4, 1))
    assert_equals(
        query_points_list.shape,
        (
            NUM_FINGERS,
            NUM_PTS_X,
            NUM_PTS_Y,
            NUM_PTS_Z,
            NUM_XYZ,
        ),
    )
    query_points_list = [
        query_points_list[finger_idx].reshape(-1, NUM_XYZ).cpu().numpy()
        for finger_idx in range(NUM_FINGERS)
    ]
    query_point_colors_list = [
        colors[finger_idx].reshape(-1).cpu().numpy()
        for finger_idx in range(NUM_FINGERS)
    ]
    fig = plot_mesh_and_query_points(
        mesh=mesh,
        query_points_list=query_points_list,
        query_points_colors_list=query_point_colors_list,
        num_fingers=NUM_FINGERS,
    )
    # Set title to label
    fig.update_layout(
        title_text=f"passed_simulation = {passed_simulation}, passed_penetration_threshold = {passed_penetration_threshold}, passed_eval = {passed_eval}"
    )
    return fig


@localscope.mfc(
    allowed=[
        "NUM_FINGERS",
        "DEPTH_IMAGE_N_CHANNELS",
        "DEPTH_IMAGE_HEIGHT",
        "DEPTH_IMAGE_WIDTH",
    ]
)
def depth_image_plot_example(
    batch_data: BatchData, idx_to_visualize: int = 0, augmented: bool = False
) -> Tuple[go.Figure, go.Figure]:
    assert isinstance(batch_data.input, DepthImageBatchDataInput)

    if augmented:
        grasp_transforms = [
            batch_data.input.augmented_grasp_transforms[idx_to_visualize][finger_idx]
            .matrix()
            .cpu()
            .numpy()
            for finger_idx in range(NUM_FINGERS)
        ]
        additional_mesh_transform = (
            batch_data.input.random_rotate_transform[idx_to_visualize]
            .matrix()
            .cpu()
            .numpy()
            if batch_data.input.random_rotate_transform is not None
            else None
        )
    else:
        grasp_transforms = [
            batch_data.input.grasp_transforms[idx_to_visualize][finger_idx]
            .matrix()
            .cpu()
            .numpy()
            for finger_idx in range(NUM_FINGERS)
        ]
        additional_mesh_transform = None

    # Extract data
    depth_uncertainty_images = batch_data.input.depth_uncertainty_images[
        idx_to_visualize
    ]
    assert_equals(
        depth_uncertainty_images.shape,
        (NUM_FINGERS, DEPTH_IMAGE_N_CHANNELS, DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH),
    )
    depth_images, uncertainty_images = (
        depth_uncertainty_images[:, 0],
        depth_uncertainty_images[:, 1],
    )
    max_depth, max_uncertainty = (
        depth_images.max().item(),
        uncertainty_images.max().item(),
    )

    passed_simulation = batch_data.output.passed_simulation[idx_to_visualize].tolist()
    passed_penetration_threshold = batch_data.output.passed_penetration_threshold[
        idx_to_visualize
    ].tolist()
    passed_eval = batch_data.output.passed_eval[idx_to_visualize].tolist()

    nerf_config_path = pathlib.Path(batch_data.nerf_config[idx_to_visualize])
    object_code = get_object_code(nerf_config_path)
    object_scale = get_object_scale(nerf_config_path)

    # Path to meshes
    DEXGRASPNET_DATA_ROOT = str(pathlib.Path(nerf_grasping.get_repo_root()) / "data")
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
    if additional_mesh_transform is not None:
        mesh.apply_transform(additional_mesh_transform)

    fig = plot_mesh_and_transforms(
        mesh=mesh,
        transforms=grasp_transforms,
        num_fingers=NUM_FINGERS,
    )
    # Set title to label
    fig.update_layout(
        title_text=f"passed_simulation = {passed_simulation}, passed_penetration_threshold = {passed_penetration_threshold}, passed_eval = {passed_eval}"
    )

    titles = [
        f"Depth {i//2}" if i % 2 == 0 else f"Uncertainty {i//2}"
        for i in range(2 * NUM_FINGERS)
    ]
    fig2 = make_subplots(rows=NUM_FINGERS, cols=2, subplot_titles=titles)
    for finger_idx in range(NUM_FINGERS):
        row = finger_idx + 1
        depth_image = depth_images[finger_idx].cpu().numpy()
        uncertainty_image = uncertainty_images[finger_idx].cpu().numpy()

        # Add 3 channels of this
        N_CHANNELS = 3
        depth_image = np.stack([depth_image] * N_CHANNELS, axis=-1)
        uncertainty_image = np.stack([uncertainty_image] * N_CHANNELS, axis=-1)
        assert_equals(
            depth_image.shape, (DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH, N_CHANNELS)
        )
        assert_equals(
            uncertainty_image.shape, (DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH, N_CHANNELS)
        )

        # Rescale
        depth_image = (depth_image * 255 / max_depth).astype(int)
        uncertainty_image = (uncertainty_image * 255 / max_uncertainty).astype(int)

        fig2.add_trace(go.Image(z=depth_image), row=row, col=1)
        fig2.add_trace(go.Image(z=uncertainty_image), row=row, col=2)

    return fig, fig2


# Add config var to enable / disable plotting.
# %%
PLOT_EXAMPLES = True
if PLOT_EXAMPLES:
    if USE_DEPTH_IMAGES:
        fig, fig2 = depth_image_plot_example(
            batch_data=EXAMPLE_BATCH_DATA, idx_to_visualize=1
        )
        fig.show()
        fig2.show()
    else:
        fig = nerf_densities_plot_example(
            batch_data=EXAMPLE_BATCH_DATA, idx_to_visualize=1
        )
        fig.show()

# %%
if PLOT_EXAMPLES:
    if USE_DEPTH_IMAGES:
        fig, fig2 = depth_image_plot_example(
            batch_data=EXAMPLE_BATCH_DATA, idx_to_visualize=15, augmented=True
        )
        fig.show()
        fig2.show()
    else:
        fig = nerf_densities_plot_example(
            batch_data=EXAMPLE_BATCH_DATA, idx_to_visualize=15, augmented=True
        )
        fig.show()

# %%
if PLOT_EXAMPLES:
    if USE_DEPTH_IMAGES:
        fig, fig2 = depth_image_plot_example(
            batch_data=EXAMPLE_BATCH_DATA, idx_to_visualize=14
        )
        fig.show()
        fig2.show()
    else:
        fig = nerf_densities_plot_example(
            batch_data=EXAMPLE_BATCH_DATA, idx_to_visualize=14
        )
        fig.show()

# %%
print(f"passed_simulation = {EXAMPLE_BATCH_DATA.output.passed_simulation}")
print(
    f"passed_penetration_threshold = {EXAMPLE_BATCH_DATA.output.passed_penetration_threshold}"
)


# %% [markdown]
# # Create Neural Network Model

# %%
import torch.nn as nn

# %%
# TODO(pculbert): double-check the specific instantiate call here is needed.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pull out just the CNN (without wrapping for LieTorch) for training.
assert cfg.model_config is not None
if USE_DEPTH_IMAGES:
    classifier: DepthImageClassifier = (
        cfg.model_config.get_classifier_from_camera_config(
            camera_config=cfg.nerfdata_config.fingertip_camera_config,
            n_tasks=cfg.task_type.n_tasks,
        ).to(device)
    )
else:
    classifier: Classifier = cfg.model_config.get_classifier_from_fingertip_config(
        fingertip_config=cfg.nerfdata_config.fingertip_config,
        n_tasks=cfg.task_type.n_tasks,
    ).to(device)

# %%
start_epoch = 0
optimizer = torch.optim.AdamW(
    params=classifier.parameters(),
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
    classifier.load_state_dict(checkpoint["classifier"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]
    if lr_scheduler is not None and "lr_scheduler" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    print("Done loading checkpoint")

# %% [markdown]
# # Visualize Model

# %%
print(f"classifier = {classifier}")
print(f"optimizer = {optimizer}")
print(f"lr_scheduler = {lr_scheduler}")

# %%
try:
    summary(
        model=classifier,
        input_size=(
            cfg.dataloader.batch_size,
            NUM_FINGERS,
            NUM_PTS_X,
            NUM_PTS_Y,
            NUM_PTS_Z,
        ),
        device=device,
    )
except Exception as e:
    print(f"Exception: {e}")
    print("Skipping summary")

# %%
try:
    example_input = (
        torch.zeros(
            (
                cfg.dataloader.batch_size,
                NUM_FINGERS,
                NUM_PTS_X,
                NUM_PTS_Y,
                NUM_PTS_Z,
            )
        )
        .to(device)
        .requires_grad_(True)
    )
    example_output = classifier(example_input)
    dot = make_dot(
        example_output,
        params={
            **dict(classifier.named_parameters()),
            **{"NERF_INPUT": example_input},
            **{"GRASP_LABELS": example_output},
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
    classifier: Classifier,
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
            "classifier": classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict()
            if lr_scheduler is not None
            else None,
        },
        checkpoint_filepath,
    )
    print("Done saving checkpoint")


# %%
@localscope.mfc(allowed=["tqdm", "USE_DEPTH_IMAGES"])
def _iterate_through_dataloader(
    loop_timer: LoopTimer,
    phase: Phase,
    dataloader: DataLoader,
    classifier: Classifier,
    device: torch.device,
    ce_loss_fns: List[nn.CrossEntropyLoss],
    task_type: TaskType,
    training_cfg: Optional[ClassifierTrainingConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
    losses_dict = defaultdict(list)  # loss name => list of losses (one loss per batch)
    predictions_dict, ground_truths_dict = defaultdict(list), defaultdict(
        list
    )  # task name => list of predictions / ground truths (one per datapoint)

    assert phase in [Phase.TRAIN, Phase.VAL, Phase.TEST]
    if phase == Phase.TRAIN:
        classifier.train()
        assert training_cfg is not None and optimizer is not None
    else:
        classifier.eval()
        assert training_cfg is None and optimizer is None

    assert_equals(len(ce_loss_fns), classifier.n_tasks)
    assert_equals(len(task_type.task_names), classifier.n_tasks)

    with torch.set_grad_enabled(phase == Phase.TRAIN):
        dataload_section_timer = loop_timer.add_section_timer("Data").start()
        for batch_idx, batch_data in (
            pbar := tqdm(enumerate(dataloader), total=len(dataloader))
        ):
            dataload_section_timer.stop()

            batch_idx = int(batch_idx)
            batch_data: BatchData = batch_data.to(device)
            if (
                USE_DEPTH_IMAGES
                and torch.isnan(batch_data.input.depth_uncertainty_images).any()
            ):
                print("!" * 80)
                print(
                    f"Found {torch.isnan(batch_data.input.depth_uncertainty_images).sum()} NANs in batch_data.input.depth_uncertainty_images"
                )
                print("Skipping batch...")
                print("!" * 80)
                print()
                continue

            if (
                not USE_DEPTH_IMAGES
                and torch.isnan(
                    batch_data.input.nerf_alphas_with_augmented_coords
                ).any()
            ):
                print("!" * 80)
                print(
                    f"Found {torch.isnan(batch_data.input.nerf_alphas_with_augmented_coords).sum()} NANs in batch_data.input.nerf_alphas_with_augmented_coords"
                )
                print("Skipping batch...")
                print("!" * 80)
                print()
                continue

            # Forward pass
            with loop_timer.add_section_timer("Fwd"):
                all_logits = classifier.get_all_logits(batch_data.input)
                assert_equals(
                    all_logits.shape,
                    (
                        batch_data.batch_size,
                        classifier.n_tasks,
                        classifier.n_classes,
                    ),
                )

                if task_type == TaskType.PASSED_SIMULATION:
                    task_targets = [batch_data.output.passed_simulation]
                elif task_type == TaskType.PASSED_PENETRATION_THRESHOLD:
                    task_targets = [batch_data.output.passed_penetration_threshold]
                elif task_type == TaskType.PASSED_EVAL:
                    task_targets = [batch_data.output.passed_eval]
                elif task_type == TaskType.PASSED_SIMULATION_AND_PENETRATION_THRESHOLD:
                    task_targets = [
                        batch_data.output.passed_simulation,
                        batch_data.output.passed_penetration_threshold,
                    ]
                else:
                    raise ValueError(f"Unknown task_type: {task_type}")

                assert_equals(len(task_targets), classifier.n_tasks)

                task_losses = []
                for task_i, (ce_loss_fn, task_target, task_name) in enumerate(
                    zip(ce_loss_fns, task_targets, task_type.task_names)
                ):
                    task_logits = all_logits[:, task_i, :]
                    task_loss = ce_loss_fn(
                        input=task_logits,
                        target=task_target,
                    )
                    losses_dict[f"{task_name}_loss"].append(task_loss.item())
                    task_losses.append(task_loss)
                total_loss = torch.sum(
                    torch.stack(task_losses)
                )  # TODO: Consider weighting losses.

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
                            classifier.parameters(),
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
                for task_i, (task_target, task_name) in enumerate(
                    zip(task_targets, task_type.task_names)
                ):
                    task_logits = all_logits[:, task_i, :]
                    predictions = task_logits.argmax(dim=-1).tolist()
                    ground_truths = task_target.argmax(dim=-1).tolist()
                    predictions_dict[f"{task_name}"] += predictions
                    ground_truths_dict[f"{task_name}"] += ground_truths

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

    return losses_dict, predictions_dict, ground_truths_dict


def create_log_dict(
    phase: Phase,
    loop_timer: LoopTimer,
    task_type: TaskType,
    losses_dict: Dict[str, List[float]],
    predictions_dict: Dict[str, List[float]],
    ground_truths_dict: Dict[str, List[float]],
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    temp_log_dict = {}  # Make code cleaner by excluding phase name until the end
    if optimizer is not None:
        temp_log_dict["lr"] = optimizer.param_groups[0]["lr"]

    with loop_timer.add_section_timer("Agg Loss"):
        for loss_name, losses in losses_dict.items():
            temp_log_dict[f"{loss_name}"] = np.mean(losses)

    with loop_timer.add_section_timer("Metrics"):
        assert_equals(set(predictions_dict.keys()), set(ground_truths_dict.keys()))
        assert_equals(set(predictions_dict.keys()), set(task_type.task_names))
        for task_name in task_type.task_names:
            predictions = predictions_dict[task_name]
            ground_truths = ground_truths_dict[task_name]
            for metric_name, function in [
                ("accuracy", accuracy_score),
                ("precision", precision_score),
                ("recall", recall_score),
                ("f1", f1_score),
            ]:
                temp_log_dict[f"{task_name}_{metric_name}"] = function(
                    y_true=ground_truths, y_pred=predictions
                )

    with loop_timer.add_section_timer("Confusion Matrix"):
        assert_equals(set(predictions_dict.keys()), set(ground_truths_dict.keys()))
        assert_equals(set(predictions_dict.keys()), set(task_type.task_names))
        for task_name in task_type.task_names:
            predictions = predictions_dict[task_name]
            ground_truths = ground_truths_dict[task_name]
            temp_log_dict[
                f"{task_name}_confusion_matrix"
            ] = wandb.plot.confusion_matrix(
                preds=predictions,
                y_true=ground_truths,
                class_names=["failure", "success"],
                title=f"{phase.name.title()} {task_name} Confusion Matrix",
            )

    log_dict = {}
    for key, value in temp_log_dict.items():
        log_dict[f"{phase.name.lower()}_{key}"] = value
    return log_dict


@localscope.mfc(allowed=["tqdm"])
def iterate_through_dataloader(
    phase: Phase,
    dataloader: DataLoader,
    classifier: Classifier,
    device: torch.device,
    ce_loss_fns: List[nn.CrossEntropyLoss],
    task_type: TaskType,
    training_cfg: Optional[ClassifierTrainingConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Dict[str, Any]:
    assert_equals(len(ce_loss_fns), classifier.n_tasks)
    assert_equals(len(task_type.task_names), classifier.n_tasks)

    loop_timer = LoopTimer()

    # Iterate through dataloader and get logged results
    losses_dict, predictions_dict, ground_truths_dict = _iterate_through_dataloader(
        loop_timer=loop_timer,
        phase=phase,
        dataloader=dataloader,
        classifier=classifier,
        device=device,
        ce_loss_fns=ce_loss_fns,
        task_type=task_type,
        training_cfg=training_cfg,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    # Log
    log_dict = create_log_dict(
        loop_timer=loop_timer,
        phase=phase,
        task_type=task_type,
        losses_dict=losses_dict,
        predictions_dict=predictions_dict,
        ground_truths_dict=ground_truths_dict,
        optimizer=optimizer,
    )

    loop_timer.pretty_print_section_times()
    print()
    print()

    return log_dict


@localscope.mfc(allowed=["tqdm"])
def run_training_loop(
    training_cfg: ClassifierTrainingConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    classifier: Classifier,
    device: torch.device,
    ce_loss_fns: List[nn.CrossEntropyLoss],
    optimizer: torch.optim.Optimizer,
    checkpoint_workspace_dir_path: str,
    task_type: TaskType,
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
                classifier=classifier,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )
        save_checkpoint_time_taken = time.time() - start_save_checkpoint_time

        # Train
        start_train_time = time.time()
        train_log_dict = iterate_through_dataloader(
            phase=Phase.TRAIN,
            dataloader=train_loader,
            classifier=classifier,
            device=device,
            ce_loss_fns=ce_loss_fns,
            task_type=task_type,
            training_cfg=training_cfg,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        wandb_log_dict.update(train_log_dict)
        train_time_taken = time.time() - start_train_time

        # Val
        # Can do this before or after training (decided on after since before it was always at -ln(1/N_CLASSES) ~ 0.69)
        start_val_time = time.time()
        if epoch % training_cfg.val_freq == 0 and (
            epoch != 0 or training_cfg.val_on_epoch_0
        ):
            classifier.eval()
            val_log_dict = iterate_through_dataloader(
                phase=Phase.VAL,
                dataloader=val_loader,
                classifier=classifier,
                device=device,
                ce_loss_fns=ce_loss_fns,
                task_type=task_type,
            )
            wandb_log_dict.update(val_log_dict)
        val_time_taken = time.time() - start_val_time

        classifier.train()

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
wandb.watch(classifier, log="gradients", log_freq=100)


# %%
@localscope.mfc
def compute_class_weight_np(
    train_dataset: Subset, input_dataset_full_path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        print("Loading grasp success data for class weighting...")
        t1 = time.time()
        with h5py.File(input_dataset_full_path, "r") as hdf5_file:
            passed_simulations_np = np.array(hdf5_file["/passed_simulation"][()])
            passed_penetration_threshold_np = np.array(
                hdf5_file["/passed_penetration_threshold"][()]
            )
            passed_eval_np = np.array(hdf5_file["/passed_eval"][()])
        t2 = time.time()
        print(f"Loaded grasp success data in {t2 - t1:.2f} s")

        print("Extracting training indices...")
        t3 = time.time()
        passed_simulations_np = passed_simulations_np[train_dataset.indices]
        passed_penetration_threshold_np = passed_penetration_threshold_np[
            train_dataset.indices
        ]
        passed_eval_np = passed_eval_np[train_dataset.indices]
        t4 = time.time()
        print(f"Extracted training indices in {t4 - t3:.2f} s")

        print("Computing class weight with this data...")
        t5 = time.time()

        # class_weight threshold required to make binary classes
        CLASS_WEIGHT_THRESHOLD = 0.4
        passed_simulation_class_weight_np = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(passed_simulations_np < CLASS_WEIGHT_THRESHOLD),
            y=passed_simulations_np < CLASS_WEIGHT_THRESHOLD,
        )
        passed_penetration_threshold_class_weight_np = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(passed_penetration_threshold_np < CLASS_WEIGHT_THRESHOLD),
            y=passed_penetration_threshold_np < CLASS_WEIGHT_THRESHOLD,
        )
        passed_eval_class_weight_np = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(passed_eval_np < CLASS_WEIGHT_THRESHOLD),
            y=passed_eval_np < CLASS_WEIGHT_THRESHOLD,
        )
        t6 = time.time()
        print(f"Computed class weight in {t6 - t5:.2f} s")

    except Exception as e:
        print(f"Failed to compute class weight: {e}")
        print("Using default class weight")
        passed_simulation_class_weight_np = np.array([1.0, 1.0])
        passed_penetration_threshold_class_weight_np = np.array([1.0, 1.0])
        passed_eval_class_weight_np = np.array([1.0, 1.0])
    return (
        passed_simulation_class_weight_np,
        passed_penetration_threshold_class_weight_np,
        passed_eval_class_weight_np,
    )


(
    passed_simulation_class_weight,
    passed_penetration_threshold_class_weight,
    passed_eval_class_weight,
) = compute_class_weight_np(
    train_dataset=train_dataset, input_dataset_full_path=input_dataset_full_path
)
passed_simulation_class_weight = (
    torch.from_numpy(passed_simulation_class_weight).float().to(device)
)
passed_penetration_threshold_class_weight = (
    torch.from_numpy(passed_penetration_threshold_class_weight).float().to(device)
)
passed_eval_class_weight = torch.from_numpy(passed_eval_class_weight).float().to(device)
print(f"passed_simulation_class_weight = {passed_simulation_class_weight}")
print(
    f"passed_penetration_threshold_class_weight = {passed_penetration_threshold_class_weight}"
)
print(f"passed_eval_class_weight = {passed_eval_class_weight}")

if cfg.training.extra_punish_false_positive_factor != 0.0:
    print(
        f"cfg.training.extra_punish_false_positive_factor = {cfg.training.extra_punish_false_positive_factor}"
    )
    passed_simulation_class_weight[1] *= (
        1 + cfg.training.extra_punish_false_positive_factor
    )
    passed_penetration_threshold_class_weight[1] *= (
        1 + cfg.training.extra_punish_false_positive_factor
    )
    passed_eval_class_weight[1] *= 1 + cfg.training.extra_punish_false_positive_factor
    print(
        f"After adjustment, passed_simulation_class_weight: {passed_simulation_class_weight}"
    )
    print(
        f"After adjustment, passed_simulation_class_weight: {passed_penetration_threshold_class_weight}"
    )
    print(
        f"After adjustment, passed_simulation_class_weight: {passed_eval_class_weight}"
    )

passed_simulation_ce_loss_fn = nn.CrossEntropyLoss(
    weight=passed_simulation_class_weight, label_smoothing=cfg.training.label_smoothing
)
passed_penetration_threshold_ce_loss_fn = nn.CrossEntropyLoss(
    weight=passed_penetration_threshold_class_weight,
    label_smoothing=cfg.training.label_smoothing,
)
passed_eval_ce_loss_fn = nn.CrossEntropyLoss(
    weight=passed_eval_class_weight, label_smoothing=cfg.training.label_smoothing
)

if cfg.task_type == TaskType.PASSED_SIMULATION:
    ce_loss_fns = [passed_simulation_ce_loss_fn]
elif cfg.task_type == TaskType.PASSED_PENETRATION_THRESHOLD:
    ce_loss_fns = [passed_penetration_threshold_ce_loss_fn]
elif cfg.task_type == TaskType.PASSED_EVAL:
    ce_loss_fns = [passed_eval_ce_loss_fn]
elif cfg.task_type == TaskType.PASSED_SIMULATION_AND_PENETRATION_THRESHOLD:
    ce_loss_fns = [
        passed_simulation_ce_loss_fn,
        passed_penetration_threshold_ce_loss_fn,
    ]
else:
    raise ValueError(f"Unknown task_type: {cfg.task_type}")

# Save out config to file if we haven't yet.
cfg_path = pathlib.Path(checkpoint_workspace_dir_path) / "config.yaml"
if not cfg_path.exists():
    cfg_yaml = tyro.extras.to_yaml(cfg)
    with open(cfg_path, "w") as f:
        f.write(cfg_yaml)

print(cfg)
if cfg.data.debug_shuffle_labels:
    print(
        "WARNING: Shuffle labels is turned on! Random labels are being passed. Press 'c' to continue"
    )
    breakpoint()

# %%

run_training_loop(
    training_cfg=cfg.training,
    train_loader=train_loader,
    val_loader=val_loader,
    classifier=classifier,
    device=device,
    ce_loss_fns=ce_loss_fns,
    optimizer=optimizer,
    checkpoint_workspace_dir_path=checkpoint_workspace_dir_path,
    task_type=cfg.task_type,
    lr_scheduler=lr_scheduler,
    start_epoch=start_epoch,
)

# %% [markdown]
# # Test

# %%
classifier.eval()
wandb_log_dict = {}
print(f"Running test metrics on epoch {cfg.training.n_epochs}")
wandb_log_dict["epoch"] = cfg.training.n_epochs
test_log_dict = iterate_through_dataloader(
    phase=Phase.TEST,
    dataloader=test_loader,
    classifier=classifier,
    device=device,
    ce_loss_fns=ce_loss_fns,
    task_type=cfg.task_type,
)
wandb_log_dict.update(test_log_dict)
wandb.log(wandb_log_dict)

# %% [markdown]
# # Save Model

# %%
save_checkpoint(
    checkpoint_workspace_dir_path=checkpoint_workspace_dir_path,
    epoch=cfg.training.n_epochs,
    classifier=classifier,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
)

# %%
wandb.finish()
