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
    get_validated_nerf_workspaces,
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
from functools import partial

datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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
if is_notebook():
    from tqdm.notebook import tqdm as std_tqdm
else:
    from tqdm import tqdm as std_tqdm

tqdm = partial(std_tqdm, dynamic_ncols=True)

# %%
# PARAMS
DEXGRASPNET_DATA_ROOT = "/juno/u/tylerlum/github_repos/DexGraspNet/data"
GRASP_DATASET_FOLDER = (
    "2023-07-01_dataset_DESIRED_DIST_TOWARDS_OBJECT_SURFACE_MULTIPLE_STEPS_v2"
)
NERF_CHECKPOINTS_FOLDER = "nerfcheckpoints"
OUTPUT_FOLDER = f"{GRASP_DATASET_FOLDER}_learned_metric_dataset"
OUTPUT_FILENAME = f"{datetime_str}_learned_metric_dataset.h5"
PLOT_ONLY_ONE = False
SAVE_DATASET = True
PRINT_TIMING = True
LIMIT_NUM_WORKSPACES = None  # None for no limit

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
validated_nerf_workspaces = get_validated_nerf_workspaces(
    nerf_checkpoints_path=NERF_CHECKPOINTS_PATH,
)


# %%
query_points_finger_frame = get_query_points_finger_frame()



# %%
if LIMIT_NUM_WORKSPACES is not None:
    print(f"Limiting number of workspaces to {LIMIT_NUM_WORKSPACES}")
    validated_nerf_workspaces = validated_nerf_workspaces[:LIMIT_NUM_WORKSPACES]

NUM_DATA_POINTS_PER_OBJECT = 500
NUM_SCALES = 5
APPROX_NUM_DATA_POINTS_PER_OBJECT_PER_SCALE = NUM_DATA_POINTS_PER_OBJECT // NUM_SCALES
BUFFER_SCALING = 2
MAX_NUM_DATA_POINTS = (
    len(validated_nerf_workspaces) * APPROX_NUM_DATA_POINTS_PER_OBJECT_PER_SCALE * BUFFER_SCALING
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
    loop_timer = LoopTimer()
    for workspace in tqdm(validated_nerf_workspaces, desc="nerf workspaces", dynamic_ncols=True):
        with loop_timer.add_section_timer("prepare to read in data"):
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
        with loop_timer.add_section_timer("load_nerf"):
            nerf_model = load_nerf(
                path_to_workspace=workspace_path,
                bound=TORCH_NGP_BOUND,
                scale=TORCH_NGP_SCALE,
            )

        with loop_timer.add_section_timer("load mesh"):
            mesh = trimesh.load(mesh_path, force="mesh")
            mesh.apply_transform(trimesh.transformations.scale_matrix(object_scale))

        with loop_timer.add_section_timer("load grasp data"):
            full_grasp_data_list = np.load(grasp_dataset_path, allow_pickle=True)
            correct_scale_grasp_data_list = [
                grasp_data
                for grasp_data in full_grasp_data_list
                if math.isclose(grasp_data["scale"], object_scale, rel_tol=1e-3)
            ]

        for grasp_idx, grasp_data in (pbar := tqdm(
            enumerate(correct_scale_grasp_data_list),
            total=len(correct_scale_grasp_data_list),
            dynamic_ncols=True,
        )):
            pbar.set_description(f"grasp data, current_idx: {current_idx}")
            # Go from contact candidates to transforms
            with loop_timer.add_section_timer("get_contact_candidates_and_target_candidates"):
                (
                    contact_candidates,
                    target_contact_candidates,
                ) = get_contact_candidates_and_target_candidates(grasp_data)
            with loop_timer.add_section_timer("get_start_and_end_and_up_points"):
                start_points, end_points, up_points = get_start_and_end_and_up_points(
                    contact_candidates=contact_candidates,
                    target_contact_candidates=target_contact_candidates,
                    num_fingers=NUM_FINGERS,
                )
            with loop_timer.add_section_timer("get_transforms"):
                transforms = [
                    get_transform(start_points[i], end_points[i], up_points[i])
                    for i in range(NUM_FINGERS)
                ]

            # Transform query points
            with loop_timer.add_section_timer("get_transformed_points"):
                query_points_object_frame_list = [
                    get_transformed_points(
                        query_points_finger_frame.reshape(-1, 3), transform
                    )
                    for transform in transforms
                ]
            with loop_timer.add_section_timer("ig_to_nerf"):
                query_points_isaac_frame_list = [
                    np.copy(query_points_object_frame)
                    for query_points_object_frame in query_points_object_frame_list
                ]
                query_points_nerf_frame_list = [
                    ig_to_nerf(query_points_isaac_frame, return_tensor=True)
                    for query_points_isaac_frame in query_points_isaac_frame_list
                ]

            # Get densities
            with loop_timer.add_section_timer("get_nerf_densities"):
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
                    with loop_timer.add_section_timer("save values"):
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
        if PRINT_TIMING:
            loop_timer.pretty_print_section_times()
        print()

# %%
