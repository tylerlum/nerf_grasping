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
# # Create DexGraspNet NeRF Grasps
#
# ## Summary (Jul 26, 2023)
#
# The purpose of this script is to iterate through each NeRF object and labeled grasp, sample densities in the grasp trajectory, and storing the data

# %%
import h5py
import math
import nerf_grasping
from nerf_grasping.grasp_utils import ig_to_nerf
import os
import trimesh
import numpy as np
from tqdm import tqdm
from datetime import datetime
import time
from nerf_grasping.dataset.DexGraspNet_NeRF_Grasps_utils import (
    get_query_points_finger_frame,
    get_contact_candidates_and_target_candidates,
    get_start_and_end_and_up_points,
    get_transform,
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

datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# %%
# PARAMS
DEXGRASPNET_DATA_ROOT = "/juno/u/tylerlum/github_repos/DexGraspNet/data"
GRASP_DATASET_FOLDER = (
    "2023-07-01_dataset_DESIRED_DIST_TOWARDS_OBJECT_SURFACE_MULTIPLE_STEPS_v2"
)
NERF_CHECKPOINTS_FOLDER = "2023-07-25_nerf_checkpoints"
OUTPUT_FOLDER = f"{GRASP_DATASET_FOLDER}_learned_metric_dataset"
OUTPUT_FILENAME = f"{datetime_str}_learned_metric_dataset.h5"
PLOT_ONLY_ONE = False
SAVE_DATASET = True
PRINT_TIMING = False
LIMIT_NUM_WORKSPACES = 2  # None for no limit

# %%
DEXGRASPNET_MESHDATA_ROOT = os.path.join(DEXGRASPNET_DATA_ROOT, "meshdata")
DEXGRASPNET_DATASET_ROOT = os.path.join(
    DEXGRASPNET_DATA_ROOT,
    GRASP_DATASET_FOLDER,
)
NERF_CHECKPOINTS_PATH = os.path.join(
    nerf_grasping.get_repo_root(), NERF_CHECKPOINTS_FOLDER
)
OUTPUT_FOLDER_PATH = os.path.join(
    nerf_grasping.get_repo_root(),
    OUTPUT_FOLDER,
)
OUTPUT_FILE_PATH = os.path.join(
    OUTPUT_FOLDER_PATH,
    OUTPUT_FILENAME,
)
TORCH_NGP_BOUND = 2.0  # Copied from nerf collection script
TORCH_NGP_SCALE = 1.0  # Copied from nerf collection script

# %%


class Timer:
    def __init__(self, name_of_timed_commands, get_current_time_fn=time.perf_counter):
        self.name = name_of_timed_commands
        self.get_current_time_fn = get_current_time_fn

    def __enter__(self):
        if PRINT_TIMING:
            self.start = self.get_current_time_fn()
        return self

    def __exit__(self, type, value, traceback):
        if PRINT_TIMING:
            print(
                f"Time elapsed for '{self.name}' is {(self.get_current_time_fn() - self.start) * 1000} ms"
            )
        return


# %%
if not os.path.exists(OUTPUT_FOLDER_PATH):
    print(f"Creating output folder {OUTPUT_FOLDER_PATH}")
    os.makedirs(OUTPUT_FOLDER_PATH)
else:
    print(f"Output folder {OUTPUT_FOLDER_PATH} already exists")

# %%
if os.path.exists(OUTPUT_FILE_PATH):
    print(f"Output file {OUTPUT_FILE_PATH} already exists")
    assert False, "Output file already exists"


# %%


# %%


validate_nerf_checkpoints_path(
    nerf_checkpoints_path=NERF_CHECKPOINTS_PATH,
)


# %%
# Get contact candidates and target contact candidates


query_points_finger_frame = get_query_points_finger_frame(
    num_pts_x=NUM_PTS_X,
    num_pts_y=NUM_PTS_Y,
    num_pts_z=NUM_PTS_Z,
    grasp_depth_mm=GRASP_DEPTH_MM,
    finger_width_mm=FINGER_WIDTH_MM,
    finger_height_mm=FINGER_HEIGHT_MM,
)


# %%



# %%
workspaces = os.listdir(NERF_CHECKPOINTS_PATH)
if LIMIT_NUM_WORKSPACES is not None:
    workspaces = workspaces[:LIMIT_NUM_WORKSPACES]

NUM_DATA_POINTS_PER_OBJECT = 500
NUM_SCALES = 5
APPROX_NUM_DATA_POINTS_PER_OBJECT_PER_SCALE = NUM_DATA_POINTS_PER_OBJECT // NUM_SCALES
BUFFER_SCALING = 2
MAX_NUM_DATA_POINTS = (
    len(workspaces) * APPROX_NUM_DATA_POINTS_PER_OBJECT_PER_SCALE * BUFFER_SCALING
)
print(f"MAX_NUM_DATA_POINTS: {MAX_NUM_DATA_POINTS}")

with h5py.File(OUTPUT_FILE_PATH, "w") as hdf5_file:
    current_idx = 0

    # TODO: Figure out what needs to be stored
    nerf_densities_dataset = hdf5_file.create_dataset(
        "/nerf_densities",
        shape=(MAX_NUM_DATA_POINTS, NUM_FINGERS, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z),
        dtype="f",
        chunks=(1, NUM_FINGERS, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z),
    )
    grasp_success_dataset = hdf5_file.create_dataset(
        "/grasp_success", shape=(MAX_NUM_DATA_POINTS,), dtype="i"
    )
    nerf_workspace_dataset = hdf5_file.create_dataset(
        "/nerf_workspace", shape=(MAX_NUM_DATA_POINTS,), dtype=h5py.string_dtype()
    )
    object_code_dataset = hdf5_file.create_dataset(
        "/object_code", shape=(MAX_NUM_DATA_POINTS,), dtype=h5py.string_dtype()
    )
    object_scale_dataset = hdf5_file.create_dataset(
        "/object_scale", shape=(MAX_NUM_DATA_POINTS,), dtype="f"
    )
    grasp_idx_dataset = hdf5_file.create_dataset(
        "/grasp_idx", shape=(MAX_NUM_DATA_POINTS,), dtype="i"
    )
    grasp_transforms_dataset = hdf5_file.create_dataset(
        "/grasp_transforms", shape=(MAX_NUM_DATA_POINTS, NUM_FINGERS, 4, 4), dtype="f"
    )

    # Iterate through all
    for workspace in tqdm(workspaces, desc="nerf workspaces", dynamic_ncols=True):
        with Timer("prepare to read in data"):
            # Prepare to read in data
            workspace_path = os.path.join(NERF_CHECKPOINTS_PATH, workspace)
            object_code = get_object_code(workspace)
            object_scale = get_object_scale(workspace)
            mesh_path = os.path.join(
                DEXGRASPNET_MESHDATA_ROOT,
                object_code,
                "coacd",
                "decomposed.obj",
            )
            grasp_dataset_path = os.path.join(
                DEXGRASPNET_DATASET_ROOT,
                f"{object_code}.npy",
            )

            # Check that mesh and grasp dataset exist
            assert os.path.exists(mesh_path), f"mesh_path {mesh_path} does not exist"
            assert os.path.exists(
                grasp_dataset_path
            ), f"grasp_dataset_path {grasp_dataset_path} does not exist"

        # Read in data
        with Timer("load_nerf"):
            nerf_model = load_nerf(
                path_to_workspace=workspace_path,
                bound=TORCH_NGP_BOUND,
                scale=TORCH_NGP_SCALE,
            )

        with Timer("load mesh"):
            mesh = trimesh.load(mesh_path, force="mesh")
            mesh.apply_transform(trimesh.transformations.scale_matrix(object_scale))

        with Timer("load grasp data"):
            full_grasp_data_list = np.load(grasp_dataset_path, allow_pickle=True)
            correct_scale_grasp_data_list = [
                grasp_data
                for grasp_data in full_grasp_data_list
                if math.isclose(grasp_data["scale"], object_scale, rel_tol=1e-3)
            ]

        # Store query points in finger frame (before transform)
        with Timer("get_query_points_finger_frame"):
            query_points_finger_frame = get_query_points_finger_frame(
                num_pts_x=NUM_PTS_X,
                num_pts_y=NUM_PTS_Y,
                num_pts_z=NUM_PTS_Z,
                grasp_depth_mm=GRASP_DEPTH_MM,
                finger_width_mm=FINGER_WIDTH_MM,
                finger_height_mm=FINGER_HEIGHT_MM,
            )

        for grasp_idx, grasp_data in (pbar := tqdm(
            enumerate(correct_scale_grasp_data_list),
            total=len(correct_scale_grasp_data_list),
            dynamic_ncols=True,
        )):
            pbar.set_description(f"grasp data, current_idx: {current_idx}")
            # Go from contact candidates to transforms
            with Timer("get_contact_candidates_and_target_candidates"):
                (
                    contact_candidates,
                    target_contact_candidates,
                ) = get_contact_candidates_and_target_candidates(grasp_data)
            with Timer("get_start_and_end_and_up_points"):
                start_points, end_points, up_points = get_start_and_end_and_up_points(
                    contact_candidates=contact_candidates,
                    target_contact_candidates=target_contact_candidates,
                    num_fingers=NUM_FINGERS,
                )
            with Timer("get_transforms"):
                transforms = [
                    get_transform(start_points[i], end_points[i], up_points[i])
                    for i in range(NUM_FINGERS)
                ]

            # Transform query points
            with Timer("get_transformed_points"):
                query_points_object_frame_list = [
                    get_transformed_points(
                        query_points_finger_frame.reshape(-1, 3), transform
                    )
                    for transform in transforms
                ]
            with Timer("ig_to_nerf"):
                query_points_isaac_frame_list = [
                    np.copy(query_points_object_frame)
                    for query_points_object_frame in query_points_object_frame_list
                ]
                query_points_nerf_frame_list = [
                    ig_to_nerf(query_points_isaac_frame, return_tensor=True)
                    for query_points_isaac_frame in query_points_isaac_frame_list
                ]

            # Get densities
            with Timer("get_nerf_densities"):
                nerf_densities = [
                    get_nerf_densities(
                        nerf_model, query_points_nerf_frame.float().cuda()
                    )
                    .reshape(-1)
                    .detach()
                    .cpu()
                    .numpy()
                    for query_points_nerf_frame in query_points_nerf_frame_list
                ]

            # Plot
            if PLOT_ONLY_ONE:
                fig = plot_mesh_and_query_points(
                    mesh=mesh,
                    query_points_list=query_points_object_frame_list,
                    query_points_colors_list=nerf_densities,
                    num_fingers=NUM_FINGERS,
                )
                fig.show()
                fig2 = plot_mesh_and_transforms(
                    mesh=mesh,
                    transforms=transforms,
                    num_fingers=NUM_FINGERS,
                )
                fig2.show()
                assert False, "PLOT_ONLY_ONE is True"
            # Save values
            if SAVE_DATASET:
                # Ensure no nans (most likely come from nerf densities)
                if not np.isnan(nerf_densities).any():
                    with Timer("save values"):
                        nerf_densities = np.stack(nerf_densities, axis=0).reshape(
                            NUM_FINGERS, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z
                        )
                        nerf_densities_dataset[current_idx] = nerf_densities
                        grasp_success_dataset[current_idx] = grasp_data["valid"]
                        nerf_workspace_dataset[current_idx] = workspace_path
                        object_code_dataset[current_idx] = object_code
                        object_scale_dataset[current_idx] = object_scale
                        grasp_idx_dataset[current_idx] = grasp_idx
                        grasp_transforms_dataset[current_idx] = np.stack(
                            transforms, axis=0
                        )

                        current_idx += 1

                        # May not be max_num_data_points if nan grasps
                        hdf5_file.attrs["num_data_points"] = current_idx
                else:
                    print()
                    print("-" * 80)
                    print(
                        f"WARNING: Found {np.isnan(nerf_densities).sum()} nans in grasp {grasp_idx} of {workspace_path}"
                    )
                    print("-" * 80)
                    print()
            print()

# %%
