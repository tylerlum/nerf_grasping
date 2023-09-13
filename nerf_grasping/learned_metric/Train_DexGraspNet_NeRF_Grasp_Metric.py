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
import functools
from localscope import localscope
import nerf_grasping
from dataclasses import asdict
from torchinfo import summary
from torchviz import make_dot
from nerf_grasping.learned_metric.DexGraspNet_batch_data import (
    BatchData,
    BatchDataInput,
    BatchDataOutput,
)
from nerf_grasping.dataset.DexGraspNet_NeRF_Grasps_utils import (
    get_object_code,
    get_object_scale,
    plot_mesh_and_query_points,
)
from nerf_grasping.classifier import Classifier
from nerf_grasping.dataset.timers import LoopTimer
from nerf_grasping.config.classifier_config import (
    UnionClassifierConfig,
)
from nerf_grasping.config.fingertip_config import BaseFingertipConfig
from nerf_grasping.config.nerfdata_config import GraspConditionedGridDataConfig
import os
import pypose as pp
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
from datetime import datetime
import sys
import random
import shutil
from wandb.util import generate_id

from enum import Enum, auto
from nerf_grasping.models.tyler_new_models import get_scheduler

import tyro


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
    arguments = []
else:
    arguments = sys.argv[1:]
    print(f"arguments = {arguments}")

# %%
cfg: UnionClassifierConfig = tyro.cli(UnionClassifierConfig, args=arguments)

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
    name=cfg.wandb.name,
    group=cfg.wandb.group if len(cfg.wandb.group) > 0 else None,
    job_type=cfg.wandb.job_type if len(cfg.wandb.job_type) > 0 else None,
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
        load_grasp_successes_in_ram: bool = True,
        load_grasp_transforms_in_ram: bool = True,
        load_nerf_configs_in_ram: bool = True,
        use_conditioning_var: bool = False,
        load_conditioning_var_in_ram: bool = True,
    ) -> None:
        super().__init__()
        self.input_hdf5_filepath = input_hdf5_filepath

        # Recommended in https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/13
        self.hdf5_file = None
        self.use_conditioning_var = use_conditioning_var

        with h5py.File(self.input_hdf5_filepath, "r") as hdf5_file:
            self.len = self._set_length(
                hdf5_file=hdf5_file, max_num_data_points=max_num_data_points
            )

            # Check that the data is in the expected format
            assert (
                len(hdf5_file["/passed_simulation"].shape) == 1
            ), f"{hdf5_file['/passed_simulation'].shape}"
            assert (
                len(hdf5_file["/passed_penetration_threshold"].shape) == 1
            ), f"{hdf5_file['/passed_penetration_threshold'].shape}"
            assert (
                len(hdf5_file["/passed_self_penetration_threshold"].shape) == 1
            ), f"{hdf5_file['/passed_self_penetration_threshold'].shape}"
            assert hdf5_file["/nerf_densities"].shape[1:] == (
                NUM_FINGERS,
                NUM_PTS_X,
                NUM_PTS_Y,
                NUM_PTS_Z,
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
            self.passed_simulations = (
                torch.from_numpy(hdf5_file["/passed_simulation"][()]).long()
                if load_grasp_successes_in_ram
                else None
            )
            self.passed_penetration_thresholds = (
                torch.from_numpy(hdf5_file["/passed_penetration_threshold"][()]).long()
                if load_grasp_successes_in_ram
                else None
            )
            self.passed_self_penetration_thresholds = (
                torch.from_numpy(
                    hdf5_file["/passed_self_penetration_threshold"][()]
                ).long()
                if load_grasp_successes_in_ram
                else None
            )

            # This is small enough to fit in RAM
            self.grasp_transforms = (
                pp.from_matrix(
                    torch.from_numpy(hdf5_file["/grasp_transforms"][()]).float(),
                    pp.SE3_type,
                )
                if load_grasp_transforms_in_ram
                else None
            )

            # This is small enough to fit in RAM
            self.nerf_configs = (
                hdf5_file["/nerf_config"][()] if load_nerf_configs_in_ram else None
            )

            if use_conditioning_var:
                assert (
                    len(hdf5_file["/conditioning_var"].shape) == 2
                ), f"{hdf5_file['/conditioning_var'].shape}"
                self.conditioning_var = (
                    hdf5_file["/conditioning_var"][()]
                    if load_conditioning_var_in_ram
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

        passed_simulation = (
            torch.from_numpy(np.array(self.hdf5_file["/passed_simulation"][idx])).long()
            if self.passed_simulations is None
            else self.passed_simulations[idx]
        )
        passed_penetration_threshold = (
            torch.from_numpy(
                np.array(self.hdf5_file["/passed_penetration_threshold"][idx])
            ).long()
            if self.passed_penetration_thresholds is None
            else self.passed_penetration_thresholds[idx]
        )
        passed_self_penetration_threshold = (
            torch.from_numpy(
                np.array(self.hdf5_file["/passed_self_penetration_threshold"][idx])
            ).long()
            if self.passed_self_penetration_thresholds is None
            else self.passed_self_penetration_thresholds[idx]
        )

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

        assert nerf_densities.shape == (NUM_FINGERS, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z)
        assert passed_simulation.shape == ()
        assert passed_penetration_threshold.shape == ()
        assert passed_self_penetration_threshold.shape == ()
        assert grasp_transforms.shape == (NUM_FINGERS, 4, 4)

        if self.use_conditioning_var:
            conditioning_var = (
                self.hdf5_file["/conditioning_var"][idx]
                if self.conditioning_var is None
                else self.conditioning_var[idx]
            )
        else:
            conditioning_var = None

        return (
            nerf_densities,
            passed_simulation,
            passed_penetration_threshold,
            passed_self_penetration_threshold,
            grasp_transforms,
            nerf_config,
            conditioning_var,
        )


# %%

input_dataset_full_path = str(cfg.nerfdata_config.output_filepath)
full_dataset = NeRFGrid_To_GraspSuccess_HDF5_Dataset(
    input_hdf5_filepath=input_dataset_full_path,
    max_num_data_points=cfg.data.max_num_data_points,
    load_nerf_densities_in_ram=cfg.dataloader.load_nerf_grid_inputs_in_ram,
    load_grasp_successes_in_ram=cfg.dataloader.load_grasp_successes_in_ram,
    load_grasp_transforms_in_ram=cfg.dataloader.load_grasp_transforms_in_ram,
    load_nerf_configs_in_ram=cfg.dataloader.load_nerf_configs_in_ram,
    use_conditioning_var=isinstance(
        cfg.nerfdata_config, GraspConditionedGridDataConfig
    ),
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


def sample_random_rotate_transforms(N: int) -> pp.LieTensor:
    # Sample big rotations in tangent space of SO(3).
    # Choose 4 * \pi as a heuristic to get pretty evenly spaced rotations.
    # TODO(pculbert): Figure out better uniform sampling on SO(3).
    log_random_rotations = pp.so3(4 * torch.pi * (2 * torch.rand(N, 3) - 1))

    # Return exponentiated rotations.
    random_SO3_rotations = log_random_rotations.Exp()

    # A bit annoying -- need to cast SO(3) -> SE(3).
    random_rotate_transforms = pp.from_matrix(
        random_SO3_rotations.matrix(), pp.SE3_type
    )

    return random_rotate_transforms


@localscope.mfc
def custom_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]],
    fingertip_config: BaseFingertipConfig,
    use_random_rotations: bool = True,
    debug_shuffle_labels: bool = False,
    use_conditioning_var: bool = False,
) -> BatchData:
    batch = torch.utils.data.dataloader.default_collate(batch)
    (
        nerf_densities,
        passed_simulation,
        passed_penetration_threshold,
        passed_self_penetration_threshold,
        grasp_transforms,
        nerf_configs,
        conditioning_var,
    ) = batch

    if use_conditioning_var:
        assert conditioning_var is not None
    else:
        assert conditioning_var is None

    if debug_shuffle_labels:
        shuffle_inds = torch.randperm(passed_simulation.shape[0])
        passed_simulation = passed_simulation[shuffle_inds]
        passed_penetration_threshold = passed_penetration_threshold[shuffle_inds]
        passed_self_penetration_threshold = passed_self_penetration_threshold[
            shuffle_inds
        ]

    grasp_transforms = pp.from_matrix(grasp_transforms, pp.SE3_type)

    batch_size = nerf_densities.shape[0]
    if use_random_rotations:
        random_rotate_transform = sample_random_rotate_transforms(N=batch_size)

        if use_conditioning_var:
            # Apply random rotation to grasp config.
            # NOTE: hardcodes that conditioning is a grasp conditioning.
            wrist_pose = pp.SE3(conditioning_var[..., :7])
            joint_angles = conditioning_var[..., 7:23]
            grasp_orientations = pp.SO3(conditioning_var[..., 23:])

            wrist_pose = random_rotate_transform.unsqueeze(1) @ wrist_pose
            grasp_orientations = (
                random_rotate_transform.rotation().unsqueeze(1) @ grasp_orientations
            )

            conditioning_var = torch.cat(
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
            conditioning_var=conditioning_var,
        ),
        output=BatchDataOutput(
            passed_simulation=passed_simulation,
            passed_penetration_threshold=passed_penetration_threshold,
            passed_self_penetration_threshold=passed_self_penetration_threshold,
        )
        nerf_config=nerf_configs,
    )


# %%
train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.dataloader.batch_size,
    shuffle=True,
    pin_memory=cfg.dataloader.pin_memory,
    num_workers=cfg.dataloader.num_workers,
    collate_fn=partial(
        custom_collate_fn,
        fingertip_config=cfg.nerfdata_config.fingertip_config,
        use_random_rotations=cfg.data.use_random_rotations,
        debug_shuffle_labels=cfg.data.debug_shuffle_labels,
        use_conditioning_var=isinstance(
            cfg.nerfdata_config, GraspConditionedGridDataConfig
        ),
    ),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.dataloader.batch_size,
    shuffle=False,
    pin_memory=cfg.dataloader.pin_memory,
    num_workers=cfg.dataloader.num_workers,
    collate_fn=partial(
        custom_collate_fn,
        fingertip_config=cfg.nerfdata_config.fingertip_config,
        use_random_rotations=False,
        use_conditioning_var=isinstance(
            cfg.nerfdata_config, GraspConditionedGridDataConfig
        ),
    ),  # Run val over actual grasp transforms (no random rotations)
)
test_loader = DataLoader(
    test_dataset,
    batch_size=cfg.dataloader.batch_size,
    shuffle=False,
    pin_memory=cfg.dataloader.pin_memory,
    num_workers=cfg.dataloader.num_workers,
    collate_fn=partial(
        custom_collate_fn,
        use_random_rotations=False,
        fingertip_config=cfg.nerfdata_config.fingertip_config,
        use_conditioning_var=isinstance(
            cfg.nerfdata_config, GraspConditionedGridDataConfig
        ),
    ),  # Run test over actual test transforms.
)

if cfg.data.use_random_rotations:
    print("Using random rotations for training")
else:
    print("Not using random rotations for training")


# %%
@localscope.mfc
def print_shapes(batch_data: BatchData) -> None:
    print(f"nerf_alphas.shape: {batch_data.input.nerf_alphas.shape}")
    print(f"grasp_success.shape: {batch_data.output.grasp_success.shape}")
    print(f"passed_simulation.shape: {batch_data.output.passed_simulation.shape}")
    print(
        f"passed_penetration_threshold.shape: {batch_data.output.passed_penetration_threshold.shape}"
    )
    print(
        f"passed_self_penetration_threshold.shape: {batch_data.output.passed_self_penetration_threshold.shape}"
    )
    print(f"grasp_transforms.shape: {batch_data.input.grasp_transforms.shape}")
    print(f"len(nerf_config): {len(batch_data.nerf_config)}")
    print(f"coords.shape = {batch_data.input.coords.shape}")
    print(
        f"nerf_alphas_with_coords.shape = {batch_data.input.nerf_alphas_with_coords.shape}"
    )
    print(
        f"augmented_grasp_transforms.shape = {batch_data.input.augmented_grasp_transforms.shape}"
    )


EXAMPLE_BATCH_DATA = next(iter(val_loader))
print_shapes(batch_data=EXAMPLE_BATCH_DATA)

# %% [markdown]
# # Visualize Data


# %%


@localscope.mfc(
    allowed=["NUM_FINGERS", "NUM_PTS_X", "NUM_PTS_Y", "NUM_PTS_Z", "NUM_XYZ"]
)
def plot_example(
    batch_data: BatchData, idx_to_visualize: int = 0, augmented: bool = False
) -> go.Figure:
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
    grasp_success = batch_data.output.grasp_success[idx_to_visualize].item()
    passed_simulation = batch_data.output.passed_simulation[idx_to_visualize].item()
    passed_penetration_threshold = batch_data.output.passed_penetration_threshold[
        idx_to_visualize
    ].item()
    passed_self_penetration_threshold = batch_data.output.passed_self_penetration_threshold[
        idx_to_visualize
    ].item()

    assert colors.shape == (NUM_FINGERS, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z)
    assert grasp_success in [0, 1]
    assert passed_simulation in [0, 1]
    assert passed_penetration_threshold in [0, 1]
    assert passed_self_penetration_threshold in [0, 1]

    nerf_config_path = pathlib.Path(batch_data.nerf_config[idx_to_visualize])
    object_code = get_object_code(nerf_config_path)
    object_scale = get_object_scale(nerf_config_path)

    # Path to meshes
    DEXGRASPNET_DATA_ROOT = str(pathlib.Path(nerf_grasping.get_repo_root()) / "data")
    # TODO: add to cfg.
    DEXGRASPNET_MESHDATA_ROOT = os.path.join(DEXGRASPNET_DATA_ROOT, "meshdata_trial")
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
        title_text=f"grasp_success = {grasp_success}, passed_simulation = {passed_simulation}, passed_penetration_threshold = {passed_penetration_threshold}, passed_self_penetration_threshold = {passed_self_penetration_threshold}"
    )
    return fig


# Add config var to enable / disable plotting.
# %%
# fig = plot_example(batch_data=EXAMPLE_BATCH_DATA, idx_to_visualize=15)
# fig.show()

# %%
# fig = plot_example(batch_data=EXAMPLE_BATCH_DATA, idx_to_visualize=15, augmented=True)
# fig.show()

# %%
EXAMPLE_BATCH_DATA.output.grasp_success, EXAMPLE_BATCH_DATA.output.passed_simulation, EXAMPLE_BATCH_DATA.output.passed_penetration_threshold, EXAMPLE_BATCH_DATA.output.passed_self_penetration_threshold

# # %%
# fig = plot_example(batch_data=EXAMPLE_BATCH_DATA, idx_to_visualize=14)
# fig.show()
#
# # %%
# fig = plot_example(batch_data=EXAMPLE_BATCH_DATA, idx_to_visualize=14, augmented=True)
# fig.show()
#
# # %%
# fig = plot_example(batch_data=EXAMPLE_BATCH_DATA, idx_to_visualize=17)
# fig.show()
#
# # %%
# fig = plot_example(batch_data=EXAMPLE_BATCH_DATA, idx_to_visualize=17, augmented=True)
# fig.show()
#
# # %%
# fig = plot_example(batch_data=EXAMPLE_BATCH_DATA, idx_to_visualize=18)
# fig.show()
#
# # %%
# fig = plot_example(batch_data=EXAMPLE_BATCH_DATA, idx_to_visualize=18, augmented=True)
# fig.show()
#
# # %%
# fig = plot_example(batch_data=EXAMPLE_BATCH_DATA, idx_to_visualize=19)
# fig.show()
#
# # %%
# fig = plot_example(batch_data=EXAMPLE_BATCH_DATA, idx_to_visualize=19, augmented=True)
# fig.show()

# %% [markdown]
# # Create Neural Network Model

# %%
import torch.nn as nn

# %%
# TODO(pculbert): double-check the specific instantiate call here is needed.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pull out just the CNN (without wrapping for LieTorch) for training.
assert cfg.model_config is not None
nerf_to_grasp_success_model: Classifier = (
    cfg.model_config.get_classifier_from_fingertip_config(
        fingertip_config=cfg.nerfdata_config.fingertip_config,
        n_tasks=cfg.task_type.n_tasks,
    ).to(device)
)

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
try:
    summary(
        model=nerf_to_grasp_success_model,
        input_size=(
            cfg.dataloader.batch_size,
            cfg.model_config.n_fingers,
            *cfg.model_config.input_shape,
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
                cfg.model_config.n_fingers,
                *cfg.model_config.input_shape,
            )
        )
        .to(device)
        .requires_grad_(True)
    )
    example_output = nerf_to_grasp_success_model(example_input)
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
    nerf_to_grasp_success_model: Classifier,
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
    nerf_to_grasp_success_model: Classifier,
    device: torch.device,
    passed_simulation_ce_loss_fn: nn.CrossEntropyLoss,
    passed_penetration_threshold_ce_loss_fn: nn.CrossEntropyLoss,
    passed_self_penetration_threshold_ce_loss_fn: nn.CrossEntropyLoss,
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
            if torch.isnan(batch_data.input.nerf_alphas_with_augmented_coords).any():
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
                (
                    passed_simulation_logits,
                    passed_penetration_threshold_logits,
                    passed_self_penetration_threshold_logits,
                ) = nerf_to_grasp_success_model.get_all_logits(batch_data.input)
                passed_simulation_ce_loss = passed_simulation_ce_loss_fn(
                    input=passed_simulation_logits, target=batch_data.passed_simulation
                )
                passed_penetration_threshold_ce_loss = (
                    passed_penetration_threshold_ce_loss_fn(
                        input=passed_penetration_threshold_logits,
                        target=batch_data.passed_penetration_threshold,
                    )
                )
                passed_self_penetration_threshold_ce_loss = (
                    passed_self_penetration_threshold_ce_loss_fn(
                        input=passed_self_penetration_threshold_logits,
                        target=batch_data.passed_self_penetration_threshold,
                    )
                )
                total_loss = (
                    passed_simulation_ce_loss
                    + passed_penetration_threshold_ce_loss
                    + passed_self_penetration_threshold_ce_loss
                )

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
                passed_simulation_predictions = passed_simulation_logits.argmax(dim=-1).tolist()
                passed_simulation_ground_truths = batch_data.passed_simulation.tolist()
                all_predictions += passed_simulation_predictions
                all_ground_truths += passed_simulation_ground_truths

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
        if (
            len(all_ground_truths) > 0
            and len(all_predictions) > 0
            and wandb.run is not None
        ):
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
    nerf_to_grasp_success_model: Classifier,
    device: torch.device,
    passed_simulation_ce_loss_fn: nn.CrossEntropyLoss,
    passed_penetration_threshold_ce_loss_fn: nn.CrossEntropyLoss,
    passed_self_penetration_threshold_ce_loss_fn: nn.CrossEntropyLoss,
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
            passed_simulation_ce_loss_fn=passed_simulation_ce_loss_fn,
            passed_penetration_threshold_ce_loss_fn=passed_penetration_threshold_ce_loss_fn,
            passed_self_penetration_threshold_ce_loss_fn=passed_self_penetration_threshold_ce_loss_fn,
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
            nerf_to_grasp_success_model.eval()
            iterate_through_dataloader(
                phase=Phase.VAL,
                dataloader=val_loader,
                nerf_to_grasp_success_model=nerf_to_grasp_success_model,
                device=device,
                ce_loss_fn=ce_loss_fn,
                wandb_log_dict=wandb_log_dict,
            )
        val_time_taken = time.time() - start_val_time

        nerf_to_grasp_success_model.train()

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        print("Loading grasp success data for class weighting...")
        t1 = time.time()
        with h5py.File(input_dataset_full_path, "r") as hdf5_file:
            passed_simulations_np = np.array(hdf5_file["/passed_simulation"][()])
            passed_penetration_threshold_np = np.array(
                hdf5_file["/passed_penetration_threshold"][()]
            )
            passed_self_penetration_threshold_np = np.array(
                hdf5_file["/passed_self_penetration_threshold"][()]
            )
        t2 = time.time()
        print(f"Loaded grasp success data in {t2 - t1:.2f} s")

        print("Extracting training indices...")
        t3 = time.time()
        passed_simulations_np = passed_simulations_np[train_dataset.indices]
        passed_penetration_threshold_np = passed_penetration_threshold_np[
            train_dataset.indices
        ]
        passed_self_penetration_threshold_np = passed_self_penetration_threshold_np[
            train_dataset.indices
        ]
        t4 = time.time()
        print(f"Extracted training indices in {t4 - t3:.2f} s")

        print("Computing class weight with this data...")
        t5 = time.time()
        passed_simulation_class_weight_np = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(passed_simulations_np),
            y=passed_simulations_np,
        )
        passed_penetration_threshold_class_weight_np = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(passed_penetration_threshold_np),
            y=passed_penetration_threshold_np,
        )
        passed_self_penetration_threshold_class_weight_np = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(passed_self_penetration_threshold_np),
            y=passed_self_penetration_threshold_np,
        )
        t6 = time.time()
        print(f"Computed class weight in {t6 - t5:.2f} s")

    except Exception as e:
        print(f"Failed to compute class weight: {e}")
        print("Using default class weight")
        passed_simulation_class_weight_np = np.array([1.0, 1.0])
        passed_penetration_threshold_class_weight_np = np.array([1.0, 1.0])
        passed_self_penetration_threshold_class_weight_np = np.array([1.0, 1.0])
    return passed_simulation_class_weight_np, passed_penetration_threshold_class_weight_np, passed_self_penetration_threshold_class_weight_np

passed_simulation_class_weight, passed_penetration_threshold_class_weight, passed_self_penetration_threshold_class_weight= (
    compute_class_weight_np(
        train_dataset=train_dataset, input_dataset_full_path=input_dataset_full_path
    )
)
passed_simulation_class_weight = (
    torch.from_numpy(passed_simulation_class_weight).float().to(device)
)
passed_penetration_threshold_class_weight = (
    torch.from_numpy(passed_penetration_threshold_class_weight).float().to(device)
)
passed_self_penetration_threshold_class_weight = (
    torch.from_numpy(passed_self_penetration_threshold_class_weight).float().to(device)
)
print(f"passed_simulation_class_weight = {passed_simulation_class_weight}")
print(f"passed_penetration_threshold_class_weight = {passed_penetration_threshold_class_weight}")
print(f"passed_self_penetration_threshold_class_weight = {passed_self_penetration_threshold_class_weight}")

PUNISH_FALSE_POSITIVE_FACTOR = 1.0
if PUNISH_FALSE_POSITIVE_FACTOR != 1.0:
    print(f"HACK: PUNISH_FALSE_POSITIVE_FACTOR = {PUNISH_FALSE_POSITIVE_FACTOR}")
    passed_simulation_class_weight[1] *= PUNISH_FALSE_POSITIVE_FACTOR
    passed_penetration_threshold_class_weight[1] *= PUNISH_FALSE_POSITIVE_FACTOR
    passed_self_penetration_threshold_class_weight[1] *= PUNISH_FALSE_POSITIVE_FACTOR
    print(f"After hack, passed_simulation_class_weight: {passed_simulation_class_weight}")
    print(f"After hack, passed_penetration_threshold_class_weight: {passed_penetration_threshold_class_weight}")
    print(f"After hack, passed_self_penetration_threshold_class_weight: {passed_self_penetration_threshold_class_weight}")

passed_simulation_ce_loss_fn = nn.CrossEntropyLoss(
    weight=passed_simulation_class_weight, label_smoothing=cfg.training.label_smoothing
)
passed_penetration_threshold_ce_loss_fn = nn.CrossEntropyLoss(
    weight=passed_penetration_threshold_class_weight, label_smoothing=cfg.training.label_smoothing
)
passed_self_penetration_threshold_ce_loss_fn = nn.CrossEntropyLoss(
    weight=passed_self_penetration_threshold_class_weight, label_smoothing=cfg.training.label_smoothing
)


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

# %%
wandb.finish()
