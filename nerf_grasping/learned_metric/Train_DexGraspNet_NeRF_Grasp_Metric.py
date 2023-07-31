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
import functools
from localscope import localscope
import nerf_grasping
from dataclasses import dataclass
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
from nerf_grasping.dataset.timers import LoopTimer
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

    @property
    @localscope.mfc(allowed=["DIST_BTWN_PTS_MM"])
    def nerf_alphas(self) -> torch.Tensor:
        # alpha = 1 - exp(-delta * sigma)
        #       = probability of collision within this segment starting from beginning of segment
        DELTA = DIST_BTWN_PTS_MM / 1000
        return 1.0 - torch.exp(-DELTA * self.nerf_densities)


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
    batch_data: BatchData = batch_data
    print(f"nerf_alphas.shape: {batch_data.nerf_alphas.shape}")
    print(batch_data.nerf_densities.shape)
    print(batch_data.grasp_success.shape)
    print(batch_data.grasp_transforms.shape)
    print(len(batch_data.nerf_workspace))
    break

# %%
import torch.nn as nn
from nerf_grasping.models.tyler_new_models import (
    ConvOutputTo1D,
    PoolType,
    conv_encoder,
    mlp,
)


class CNN_3D_Classifier(nn.Module):
    def __init__(
        self, input_example_shape: Tuple[int, int, int], n_fingers: int
    ) -> None:
        # TODO: Make this not hardcoded
        super().__init__()
        self.input_example_shape = input_example_shape
        self.n_fingers = n_fingers

        assert len(input_example_shape) == 3
        self.input_shape = (1, *input_example_shape)

        self.conv = conv_encoder(
            input_shape=self.input_shape,
            conv_channels=[32, 64, 128, 256],
            pool_type=PoolType.MAX,
            dropout_prob=0.1,
            conv_output_to_1d=ConvOutputTo1D.AVG_POOL_SPATIAL,
        )

        # Get conv output shape
        example_batch_size = 2
        example_input = torch.zeros(example_batch_size, self.n_fingers, *self.input_shape)
        example_input = example_input.reshape(
            example_batch_size * self.n_fingers, 1, *self.input_example_shape
        )
        conv_output = self.conv(example_input)
        self.conv_output_dim = conv_output.shape[-1]
        assert conv_output.shape == (example_batch_size * self.n_fingers, self.conv_output_dim)

        self.mlp = mlp(
            num_inputs=self.conv_output_dim * self.n_fingers,
            num_outputs=self.n_classes,
            hidden_layers=[256, 256, 256],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        assert x.shape == (batch_size, self.n_fingers, *self.input_example_shape), f"{x.shape}"

        # Put n_fingers into batch dim
        x = x.reshape(batch_size * self.n_fingers, 1, *self.input_example_shape)
        assert x.shape == (batch_size * self.n_fingers, *self.input_shape), f"{x.shape} != {(batch_size, *self.input_shape)}"

        x = self.conv(x)
        assert x.shape == (batch_size * self.n_fingers, self.conv_output_dim), f"{x.shape}"
        x = x.reshape(batch_size, self.n_fingers * self.conv_output_dim)

        x = self.mlp(x)
        assert x.shape == (batch_size, self.n_classes), f"{x.shape}"
        return x

    def get_success_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def get_success_probability(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.softmax(self.get_success_logits(x), dim=-1)

    @property
    @functools.lru_cache
    def n_classes(self) -> int:
        return 2


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nerf_to_grasp_success_model = CNN_3D_Classifier(
    input_example_shape=(NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z), n_fingers=NUM_FINGERS
).to(
    device
)

# %%
start_epoch = 0
optimizer = torch.optim.AdamW(
    params=nerf_to_grasp_success_model.parameters(),
    lr=3e-4,
    # betas=cfg.training.betas,
    # weight_decay=cfg.training.weight_decay,
)

# %%

from enum import Enum, auto
class Phase(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()

import wandb
from tqdm import tqdm
import time
from collections import defaultdict
@localscope.mfc
def iterate_through_dataloader(
    phase: Phase,
    dataloader: DataLoader,
    nerf_to_grasp_success_model: CNN_3D_Classifier,
    device: torch.device,
    wandb_log_dict: dict,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    ce_loss_fn = nn.CrossEntropyLoss(
        # weight=class_weight, label_smoothing=cfg.training.label_smoothing
    )
    assert phase in [Phase.TRAIN, Phase.VAL, Phase.TEST]
    if phase == Phase.TRAIN:
        nerf_to_grasp_success_model.train()
        assert optimizer is not None

    else:
        nerf_to_grasp_success_model.eval()
        assert optimizer is None

    loop_timer = LoopTimer()
    with torch.set_grad_enabled(phase == Phase.TRAIN):
        losses_dict = defaultdict(list)

        dataload_section_timer = loop_timer.add_section_timer("Data").start()
        for batch_idx, batch_data in (
            pbar := tqdm(enumerate(dataloader), total=len(dataloader))
        ):
            dataload_section_timer.stop()

            batch_idx = int(batch_idx)

            # Forward pass
            with loop_timer.add_section_timer("Fwd"):
                batch_data = batch_data.to(device)

                grasp_success_logits = nerf_to_grasp_success_model.get_success_logits(
                    batch_data.nerf_alphas
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

                    if True:
                        torch.nn.utils.clip_grad_value_(
                            nerf_to_grasp_success_model.parameters(),
                            1.0,
                        )

                    optimizer.step()

            # Loss logging
            with loop_timer.add_section_timer("Loss"):
                losses_dict[f"{phase.name.lower()}_loss"].append(total_loss.item())

            # Set description
            loss_log_str = (
                f"loss: {np.mean(losses_dict[f'{phase.name.lower()}_loss']):.5f}"
                if len(losses_dict[f"{phase.name.lower()}_loss"]) > 0
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

    loop_timer.pretty_print_section_times()
    print()
    print()

    if optimizer is not None:
        wandb_log_dict[f"{phase.name.lower()}_lr"] = optimizer.param_groups[0]["lr"]

    for loss_name, losses in losses_dict.items():
        wandb_log_dict[loss_name] = np.mean(losses)

    return


@localscope.mfc
def run_training_loop(
    nerf_to_grasp_success_model: CNN_3D_Classifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    start_epoch: int,
) -> None:
    training_loop_base_description = "Training Loop"
    for epoch in (
        pbar := tqdm(
            range(start_epoch, 100), desc=training_loop_base_description
        )
    ):
        epoch = int(epoch)
        wandb_log_dict = {}
        wandb_log_dict["epoch"] = epoch

        # Save checkpoint
        start_save_checkpoint_time = time.time()
        if epoch % 100 == 0 and (
            epoch != 0 or True
        ):
            pass
            # save_checkpoint(
            #     checkpoint_workspace_dir_path=checkpoint_workspace_dir_path,
            #     epoch=epoch,
            #     nerf_to_grasp_success_model=nerf_to_grasp_success_model,
            #     optimizer=optimizer,
            #     lr_scheduler=lr_scheduler,
            # )
        save_checkpoint_time_taken = time.time() - start_save_checkpoint_time

        # Train
        start_train_time = time.time()
        iterate_through_dataloader(
            phase=Phase.TRAIN,
            dataloader=train_loader,
            nerf_to_grasp_success_model=nerf_to_grasp_success_model,
            device=device,
            wandb_log_dict=wandb_log_dict,
            optimizer=optimizer,
        )
        train_time_taken = time.time() - start_train_time

        # Val
        # Can do this before or after training (decided on after since before it was always at -ln(1/N_CLASSES) ~ 0.69)
        start_val_time = time.time()
        if epoch % 5 == 0 and (epoch != 0 or True):
            iterate_through_dataloader(
                phase=Phase.VAL,
                dataloader=val_loader,
                nerf_to_grasp_success_model=nerf_to_grasp_success_model,
                device=device,
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


run_training_loop(
    nerf_to_grasp_success_model,
    train_loader,
    val_loader,
    device,
    optimizer,
    start_epoch,
)
# %%
