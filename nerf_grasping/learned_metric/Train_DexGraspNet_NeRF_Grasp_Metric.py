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

# %%
nerf_densities, grasp_success, grasp_transforms, nerf_workspace = full_dataset[0]
nerf_densities.shape, grasp_success.shape, grasp_transforms.shape, nerf_workspace

# %%
_, workspace_name = os.path.split(nerf_workspace)
object_code, object_scale = get_object_code(workspace_name), get_object_scale(
    workspace_name
)
object_code, object_scale

# %%
DEXGRASPNET_DATA_ROOT = "/juno/u/tylerlum/github_repos/DexGraspNet/data"
DEXGRASPNET_MESHDATA_ROOT = os.path.join(DEXGRASPNET_DATA_ROOT, "meshdata")
mesh_path = os.path.join(
    DEXGRASPNET_MESHDATA_ROOT,
    object_code,
    "coacd",
    "decomposed.obj",
)

import trimesh

mesh = trimesh.load(mesh_path, force="mesh")
mesh.apply_transform(trimesh.transformations.scale_matrix(object_scale))

# %%
fig = plot_mesh_and_transforms(
    mesh=mesh, transforms=grasp_transforms, num_fingers=NUM_FINGERS
)
fig.show()

# %%
query_points_finger_frame = get_query_points_finger_frame(
    num_pts_x=NUM_PTS_X,
    num_pts_y=NUM_PTS_Y,
    num_pts_z=NUM_PTS_Z,
    grasp_depth_mm=GRASP_DEPTH_MM,
    finger_width_mm=FINGER_WIDTH_MM,
    finger_height_mm=FINGER_HEIGHT_MM,
)
query_points_object_frame_list = [
    get_transformed_points(query_points_finger_frame.reshape(-1, 3), transform)
    for transform in grasp_transforms
]
colors_list = [nerf_densities[i].reshape(-1, 3) for i in range(NUM_FINGERS)]
fig = plot_mesh_and_query_points(
    mesh=mesh,
    query_points_list=query_points_object_frame_list,
    query_points_colors_list=colors_list,
    num_fingers=NUM_FINGERS,
)
fig.show()

# %%


def plot_query_points(
    query_points_list: List[np.ndarray],
    query_points_colors_list: List[np.ndarray],
    num_fingers: int,
) -> go.Figure:
    assert (
        len(query_points_list) == len(query_points_colors_list) == num_fingers
    ), f"{len(query_points_list)} != {num_fingers}"

    # Create the layout
    layout = go.Layout(
        scene=get_scene_dict(),
        showlegend=True,
        title="Mesh",
    )

    # Create the figure
    fig = go.Figure(layout=layout)

    for finger_idx in range(num_fingers):
        query_points = query_points_list[finger_idx]
        query_points_colors = query_points_colors_list[finger_idx]
        print(f"query_points_colors = {query_points_colors}")
        query_point_plot = go.Scatter3d(
            x=query_points[:, 0],
            y=query_points[:, 1],
            z=query_points[:, 2],
            mode="markers",
            marker=dict(
                size=4,
                color=query_points_colors,
                colorscale="viridis",
                colorbar=dict(title="Density Scale") if finger_idx == 0 else {},
            ),
            name=f"Query Point Densities Finger {finger_idx}",
        )
        fig.add_trace(query_point_plot)

    fig.update_layout(legend_orientation="h")  # Avoid overlapping legend
    return fig


# %%

fig = plot_query_points(
    query_points_list=query_points_object_frame_list,
    query_points_colors_list=colors_list,
    num_fingers=NUM_FINGERS,
)
fig.show()

# %%
