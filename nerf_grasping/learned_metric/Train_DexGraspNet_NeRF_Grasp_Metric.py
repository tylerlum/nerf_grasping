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
from localscope import localscope
import nerf_grasping
from dataclasses import dataclass
from nerf_grasping.dataset.DexGraspNet_NeRF_Grasps_utils import (
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
    validate_nerf_checkpoints_path,
    load_nerf,
    NUM_PTS_X,
    NUM_PTS_Y,
    NUM_PTS_Z,
    GRASP_DEPTH_MM,
    FINGER_WIDTH_MM,
    FINGER_HEIGHT_MM,
    NUM_FINGERS,
)
import os
import h5py
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    random_split,
)
import plotly.graph_objects as go

# %% [markdown]
# # Read in Data

# %%


# %%
INPUT_EXAMPLE_SHAPE = (NUM_FINGERS, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z)


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
                hdf5_file["/nerf_densities"].shape[1:] == INPUT_EXAMPLE_SHAPE
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
            "INPUT_EXAMPLE_SHAPE",
            "NUM_FINGERS",
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

        assert nerf_densities.shape == INPUT_EXAMPLE_SHAPE
        assert grasp_success.shape == ()
        assert grasp_transforms.shape == (NUM_FINGERS, 4, 4)

        return nerf_densities, grasp_success, grasp_transforms, nerf_workspace


# %%

input_dataset = os.path.join(
    nerf_grasping.get_repo_root(),
    "2023-07-01_dataset_DESIRED_DIST_TOWARDS_OBJECT_SURFACE_MULTIPLE_STEPS_v2_learned_metric_dataset",
    "2023-07-27_01-25-58_learned_metric_dataset.h5",
)
full_dataset = NeRFGrid_To_GraspSuccess_HDF5_Dataset(input_dataset)

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [0.8, 0.1, 0.1],
    generator=torch.Generator().manual_seed(42),
)

# %%
@dataclass
class BatchData:
    nerf_densities: torch.Tensor
    grasp_success: torch.Tensor
    grasp_transforms: torch.Tensor
    nerf_workspace: List[str]

    @localscope.mfc
    def to(self, device):
        self.nerf_densities = self.nerf_densities.to(device)
        self.grasp_success = self.grasp_success.to(device)
        self.grasp_transforms = self.grasp_transforms.to(device)
        return self

def custom_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]) -> BatchData:
    batch = torch.utils.data.dataloader.default_collate(batch)
    nerf_densities, grasp_successes, grasp_transforms, nerf_workspaces = batch

    return BatchData(
        nerf_densities=nerf_densities,
        grasp_success=grasp_successes,
        grasp_transforms=grasp_transforms,
        nerf_workspace=nerf_workspaces,
    )

# %%
BATCH_SIZE = 32
PIN_MEMORY = True
NUM_WORKERS = 8
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=PIN_MEMORY,
    num_workers=NUM_WORKERS,
    collate_fn=custom_collate_fn,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=PIN_MEMORY,
    num_workers=NUM_WORKERS,
    collate_fn=custom_collate_fn,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=PIN_MEMORY,
    num_workers=NUM_WORKERS,
    collate_fn=custom_collate_fn,
)

# %%
for batch_data in train_loader:
    print(batch_data.nerf_densities.shape)
    print(batch_data.grasp_success.shape)
    print(batch_data.grasp_transforms.shape)
    print(len(batch_data.nerf_workspace))
    break

# %%
