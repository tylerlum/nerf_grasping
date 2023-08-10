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
import os
import trimesh
import numpy as np
from tqdm import tqdm
from datetime import datetime
import time
from nerf_grasping.dataset.DexGraspNet_NeRF_Grasps_utils import (
    get_contact_candidates_and_target_candidates,
    get_start_and_end_and_up_points,
    get_transform,
    plot_mesh_and_query_points,
    plot_mesh_and_transforms,
    get_object_code,
    get_object_scale,
    get_nerf_configs,
    plot_nerf_densities,
    load_nerf,
)
from nerf_grasping.dataset.timers import LoopTimer
from nerf_grasping.grasp_utils import (
    NUM_PTS_X,
    NUM_PTS_Y,
    NUM_PTS_Z,
    GRASP_DEPTH_MM,
    NUM_FINGERS,
    get_ray_samples,
    get_ray_origins_finger_frame,
)
from functools import partial

datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# %%
def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractireakeShell":
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

if not os.getcwd().split("/")[-1] == "nerf_grasping":
    os.chdir("../..")

tqdm = partial(std_tqdm, dynamic_ncols=True)

# %%
# PARAMS
DEXGRASPNET_DATA_ROOT = "."
GRASP_DATASET_FOLDER = "graspdata"
NERF_CHECKPOINTS_FOLDER = "nerfcheckpoints"
OUTPUT_FOLDER = f"{GRASP_DATASET_FOLDER}_learned_metric_dataset"
OUTPUT_FILENAME = f"{datetime_str}_learned_metric_dataset.h5"
PLOT_ONLY_ONE = True
SAVE_DATASET = True
PRINT_TIMING = True
LIMIT_NUM_CONFIGS = None  # None for no limit

# %%
DEXGRASPNET_MESHDATA_ROOT = os.path.join(DEXGRASPNET_DATA_ROOT, "meshdata")
DEXGRASPNET_DATASET_ROOT = os.path.join(
    DEXGRASPNET_DATA_ROOT,
    GRASP_DATASET_FOLDER,
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
nerf_configs = get_nerf_configs(
    nerf_checkpoints_path=NERF_CHECKPOINTS_FOLDER,
)


# %%
ray_origins_finger_frame = get_ray_origins_finger_frame()

# %%
if LIMIT_NUM_CONFIGS is not None:
    print(f"Limiting number of configs to {LIMIT_NUM_CONFIGS}")
    nerf_configs = nerf_configs[:LIMIT_NUM_CONFIGS]

NUM_DATA_POINTS_PER_OBJECT = 500
NUM_SCALES = 5
APPROX_NUM_DATA_POINTS_PER_OBJECT_PER_SCALE = NUM_DATA_POINTS_PER_OBJECT // NUM_SCALES
BUFFER_SCALING = 2
MAX_NUM_DATA_POINTS = (
    len(nerf_configs) * APPROX_NUM_DATA_POINTS_PER_OBJECT_PER_SCALE * BUFFER_SCALING
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
    nerf_config_dataset = hdf5_file.create_dataset(
        "/nerf_config", shape=(MAX_NUM_DATA_POINTS,), dtype=h5py.string_dtype()
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
    for config in tqdm(nerf_configs, desc="nerf configs", dynamic_ncols=True):
        with loop_timer.add_section_timer("prepare to read in data"):
            # Prepare to read in data

            object_code = get_object_code(config)
            object_scale = get_object_scale(config)
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
            nerf_model = load_nerf(config)

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

        for grasp_idx, grasp_data in (
            pbar := tqdm(
                enumerate(correct_scale_grasp_data_list),
                total=len(correct_scale_grasp_data_list),
                dynamic_ncols=True,
            )
        ):
            pbar.set_description(f"grasp data, current_idx: {current_idx}")
            # Go from contact candidates to transforms
            with loop_timer.add_section_timer(
                "get_contact_candidates_and_target_candidates"
            ):
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
                ray_samples_list = [
                    get_ray_samples(ray_origins_finger_frame, transform)
                    for transform in transforms
                ]

            # TODO(pculbert): Check we can actually get rid of IG transform.
            with loop_timer.add_section_timer("ig_to_nerf"):
                query_points_list = [
                    np.copy(rr.frustums.get_positions().cpu().numpy().reshape(-1, 3))
                    for rr in ray_samples_list
                ]

            # Get densities
            with loop_timer.add_section_timer("get_nerf_densities"):
                nerf_densities = [
                    nerf_model.get_density(ray_samples.to("cuda"))[0]
                    .reshape(-1)
                    .detach()
                    .cpu()
                    .numpy()
                    for ray_samples in ray_samples_list
                ]

            # Plot
            if PLOT_ONLY_ONE:
                delta = GRASP_DEPTH_MM / NUM_PTS_Z
                nerf_alphas = [1 - np.exp(-delta * dd) for dd in nerf_densities]
                fig = plot_mesh_and_query_points(
                    mesh=mesh,
                    query_points_list=query_points_list,
                    query_points_colors_list=nerf_alphas,
                    num_fingers=NUM_FINGERS,
                )
                fig.show()
                fig2 = plot_mesh_and_transforms(
                    mesh=mesh,
                    transforms=[tt.matrix().numpy() for tt in transforms],
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
                        nerf_config_dataset[current_idx] = str(config)
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
                        f"WARNING: Found {np.isnan(nerf_densities).sum()} nans in grasp {grasp_idx} of {config}"
                    )
                    print("-" * 80)
                    print()
        if PRINT_TIMING:
            loop_timer.pretty_print_section_times()
        print()

# %%
