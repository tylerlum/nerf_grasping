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
# ## Summary (Aug 25, 2023)
#
# The purpose of this script is to iterate through each NeRF object and evaled grasp config, sample densities in the grasp trajectory, and store the data

# %%
import pathlib
import h5py
import math
import torch
import pypose as pp
import nerf_grasping
import os
import trimesh
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Tuple, List, Dict, Any
from nerf_grasping.dataset.DexGraspNet_NeRF_Grasps_utils import (
    plot_mesh_and_query_points,
    plot_mesh_and_transforms,
    plot_mesh_and_high_density_points,
    get_ray_samples_in_mesh_region,
)
from nerf_grasping.dataset.timers import LoopTimer
from nerf_grasping.optimizer_utils import AllegroHandConfig
from nerf_grasping.grasp_utils import (
    NUM_PTS_X,
    NUM_PTS_Y,
    NUM_PTS_Z,
    GRASP_DEPTH_MM,
    NUM_FINGERS,
    DIST_BTWN_PTS_MM,
    get_ray_samples,
    get_ray_origins_finger_frame,
    get_nerf_configs,
    load_nerf,
)
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

if not os.getcwd().split("/")[-1] == "nerf_grasping":
    os.chdir("../..")

tqdm = partial(std_tqdm, dynamic_ncols=True)


# %%
def parse_object_code_and_scale(object_code_and_scale_str: str) -> Tuple[str, float]:
    keyword = "_0_"
    idx = object_code_and_scale_str.rfind(keyword)
    object_code = object_code_and_scale_str[:idx]

    idx_offset_for_scale = keyword.index("0")
    object_scale = float(
        object_code_and_scale_str[idx + idx_offset_for_scale :].replace("_", ".")
    )
    return object_code, object_scale


def parse_nerf_config(nerf_config: pathlib.Path) -> str:
    # Eg. PosixPath('2023-08-25_nerfcheckpoints/sem-Gun-4745991e7c0c7966a93f1ea6ebdeec6f_0_10/nerfacto/2023-08-25_132225')
    # Return sem-Gun-4745991e7c0c7966a93f1ea6ebdeec6f_0_10
    parts = nerf_config.parts
    object_code_and_scale_str = parts[-4]
    return object_code_and_scale_str


# %%
# PARAMS
DEXGRASPNET_DATA_ROOT = pathlib.Path(".")
EVALED_GRASP_CONFIG_DICTS_FOLDER = "2023-08-25_evaled_grasp_config_dicts"
NERF_CHECKPOINTS_FOLDER = "2023-08-25_nerfcheckpoints"
OUTPUT_FOLDER = f"{EVALED_GRASP_CONFIG_DICTS_FOLDER}_learned_metric_dataset"
OUTPUT_FILENAME = f"{datetime_str}_learned_metric_dataset.h5"
PLOT_ONLY_ONE = True
SAVE_DATASET = True
PRINT_TIMING = True
LIMIT_NUM_CONFIGS = None  # None for no limit

# %%
DEXGRASPNET_MESHDATA_ROOT_PATH = DEXGRASPNET_DATA_ROOT / "meshdata"
EVALED_GRASP_CONFIG_DICTS_PATH = (
    DEXGRASPNET_DATA_ROOT / EVALED_GRASP_CONFIG_DICTS_FOLDER
)

OUTPUT_FOLDER_PATH = pathlib.Path(nerf_grasping.get_repo_root()) / OUTPUT_FOLDER
OUTPUT_FILE_PATH = OUTPUT_FOLDER_PATH / OUTPUT_FILENAME

# %%
if not OUTPUT_FOLDER_PATH.exists():
    print(f"Creating output folder {OUTPUT_FOLDER_PATH}")
    OUTPUT_FOLDER_PATH.mkdir(parents=True)
else:
    print(f"Output folder {OUTPUT_FOLDER_PATH} already exists")

# %%
if OUTPUT_FILE_PATH.exists():
    print(f"Output file {OUTPUT_FILE_PATH} already exists")
    assert False, "Output file already exists"


# %%
nerf_configs = get_nerf_configs(
    nerf_checkpoints_path=str(NERF_CHECKPOINTS_FOLDER),
)


# %%
ray_origins_finger_frame = get_ray_origins_finger_frame()

# %%
if LIMIT_NUM_CONFIGS is not None:
    print(f"Limiting number of configs to {LIMIT_NUM_CONFIGS}")
    nerf_configs = nerf_configs[:LIMIT_NUM_CONFIGS]

NUM_DATA_POINTS_PER_FILE = 500
BUFFER_SCALING = 2  # Not sure if need this, but just in case
MAX_NUM_DATA_POINTS = len(nerf_configs) * NUM_DATA_POINTS_PER_FILE * BUFFER_SCALING
print(f"MAX_NUM_DATA_POINTS: {MAX_NUM_DATA_POINTS}")

with h5py.File(OUTPUT_FILE_PATH, "w") as hdf5_file:
    current_idx = 0

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
            object_code_and_scale_str = parse_nerf_config(config)
            object_code, object_scale = parse_object_code_and_scale(
                object_code_and_scale_str
            )

            # Prepare to read in data
            mesh_path = (
                DEXGRASPNET_MESHDATA_ROOT_PATH
                / object_code
                / "coacd"
                / "decomposed.obj"
            )
            evaled_grasp_config_dicts_filepath = (
                EVALED_GRASP_CONFIG_DICTS_PATH / f"{object_code_and_scale_str}.npy"
            )

            # Check that mesh and grasp dataset exist
            assert os.path.exists(mesh_path), f"mesh_path {mesh_path} does not exist"
            assert os.path.exists(
                evaled_grasp_config_dicts_filepath
            ), f"evaled_grasp_config_dicts_filepath {evaled_grasp_config_dicts_filepath} does not exist"

        # Read in data
        with loop_timer.add_section_timer("load_nerf"):
            nerf_model = load_nerf(config)

        with loop_timer.add_section_timer("load mesh"):
            mesh = trimesh.load(mesh_path, force="mesh")
            mesh.apply_transform(trimesh.transformations.scale_matrix(object_scale))

        with loop_timer.add_section_timer("load grasp data"):
            evaled_grasp_config_dicts: List[Dict[str, Any]] = np.load(
                evaled_grasp_config_dicts_filepath, allow_pickle=True
            )

        for grasp_idx, evaled_grasp_config_dict in (
            pbar := tqdm(
                enumerate(evaled_grasp_config_dicts),
                total=len(evaled_grasp_config_dicts),
                dynamic_ncols=True,
            )
        ):
            pbar.set_description(f"grasp data, current_idx: {current_idx}")
            with loop_timer.add_section_timer("get_transforms"):
                try:
                    # TODO: Potentially clean this up using AllegroGraspConfig.from_grasp_config_dicts
                    hand_config = AllegroHandConfig.from_hand_config_dicts(
                        evaled_grasp_config_dicts[grasp_idx : grasp_idx + 1],
                    )
                    fingertip_positions = (
                        hand_config.get_fingertip_transforms().translation().squeeze(dim=0)
                    )
                    assert fingertip_positions.shape == (NUM_FINGERS, 3)

                    grasp_orientations = torch.tensor(
                        evaled_grasp_config_dicts[grasp_idx]["grasp_orientations"],
                        dtype=fingertip_positions.dtype,
                        device=fingertip_positions.device,
                    )
                    assert grasp_orientations.shape == (NUM_FINGERS, 3, 3)
                    grasp_orientations = pp.from_matrix(grasp_orientations, pp.SO3_type)

                    transforms = pp.SE3(
                        torch.cat(
                            [
                                fingertip_positions,
                                grasp_orientations,
                            ],
                            dim=-1,
                        )
                    )
                    assert transforms.lshape == (NUM_FINGERS,)
                    transforms = [
                        transforms[i].detach().clone() for i in range(NUM_FINGERS)
                    ]
                except ValueError as e:
                    print("+" * 80)
                    print(f"ValueError: {e}")
                    print(f"Skipping grasp_idx: {grasp_idx} for config: {config}")
                    print("+" * 80)
                    print()
                    continue

            # Transform query points
            with loop_timer.add_section_timer("get_transformed_points"):
                ray_samples_list = [
                    get_ray_samples(ray_origins_finger_frame, transform)
                    for transform in transforms
                ]

            with loop_timer.add_section_timer("get_query_points"):
                query_points_list = [
                    np.copy(
                        rr.frustums.get_positions().cpu().numpy()
                    )  # Shape [n_x, n_y, n_z, 3]
                    for rr in ray_samples_list
                ]

                assert query_points_list[0].shape == (
                    NUM_PTS_X,
                    NUM_PTS_Y,
                    NUM_PTS_Z,
                    3,
                ), f"query_points_list[0].shape: {query_points_list[0].shape}"

            # Get densities
            with loop_timer.add_section_timer("get_nerf_densities"):
                nerf_densities = [
                    nerf_model.get_density(ray_samples.to("cuda"))[0]
                    .detach()
                    .cpu()
                    .numpy()
                    for ray_samples in ray_samples_list  # Shape [n_x, n_y, n_z].
                ]

                assert nerf_densities[0].shape == (
                    NUM_PTS_X,
                    NUM_PTS_Y,
                    NUM_PTS_Z,
                    1,
                ), f"nerf_densities[0].shape: {nerf_densities[0].shape}"

            # Plot
            if PLOT_ONLY_ONE:
                delta = DIST_BTWN_PTS_MM / 1000
                other_delta = GRASP_DEPTH_MM / 1000 / (NUM_PTS_Z - 1)
                assert np.isclose(delta, other_delta)

                nerf_alphas = [1 - np.exp(-delta * dd) for dd in nerf_densities]
                fig = plot_mesh_and_query_points(
                    mesh=mesh,
                    query_points_list=[qq.reshape(-1, 3) for qq in query_points_list],
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

                PLOT_ALL_HIGH_DENSITY_POINTS = True
                if PLOT_ALL_HIGH_DENSITY_POINTS:
                    ray_samples_in_mesh_region = get_ray_samples_in_mesh_region(
                        mesh=mesh,
                        num_pts_x=10,
                        num_pts_y=10,
                        num_pts_z=10,
                    )
                    query_points_in_mesh_region_isaac_frame = np.copy(
                        ray_samples_in_mesh_region.frustums.get_positions()
                        .cpu()
                        .numpy()
                        .reshape(-1, 3)
                    )
                    nerf_densities_in_mesh_region = (
                        nerf_model.get_density(ray_samples_in_mesh_region.to("cuda"))[0]
                        .reshape(-1)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    nerf_alphas_in_mesh_region = 1 - np.exp(
                        -delta * nerf_densities_in_mesh_region
                    )

                    fig3 = plot_mesh_and_high_density_points(
                        mesh=mesh,
                        query_points=query_points_in_mesh_region_isaac_frame,
                        query_points_colors=nerf_alphas_in_mesh_region,
                        density_threshold=0.01,
                    )
                    fig3.show()

                PLOT_ALPHAS_EACH_FINGER_1D = True
                if PLOT_ALPHAS_EACH_FINGER_1D:
                    import matplotlib.pyplot as plt

                    nrows, ncols = NUM_FINGERS, 1
                    fig4, axes = plt.subplots(
                        nrows=nrows, ncols=ncols, figsize=(10, 10)
                    )
                    axes = axes.flatten()
                    for i in range(NUM_FINGERS):
                        ax = axes[i]
                        finger_alphas = nerf_alphas[i].reshape(
                            NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z
                        )
                        finger_alphas_maxes = np.max(finger_alphas, axis=(0, 1))
                        finger_alphas_means = np.mean(finger_alphas, axis=(0, 1))
                        ax.plot(finger_alphas_maxes, label="max")
                        ax.plot(finger_alphas_means, label="mean")
                        ax.legend()
                        ax.set_xlabel("z")
                        ax.set_ylabel("alpha")
                        ax.set_title(f"finger {i}")
                        ax.set_ylim([0, 1])
                    fig4.tight_layout()
                    fig4.show()

                PLOT_ALPHA_IMAGES_EACH_FINGER = True
                if PLOT_ALPHA_IMAGES_EACH_FINGER:
                    num_images = 5
                    nrows, ncols = NUM_FINGERS, num_images
                    fig5, axes = plt.subplots(
                        nrows=nrows, ncols=ncols, figsize=(10, 10)
                    )
                    alpha_images = [
                        x.reshape(NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z) for x in nerf_alphas
                    ]

                    for finger_i in range(NUM_FINGERS):
                        for image_i in range(num_images):
                            ax = axes[finger_i, image_i]
                            image = alpha_images[finger_i][
                                :, :, int(image_i * NUM_PTS_Z / num_images)
                            ]
                            ax.imshow(image, vmin=image.min(), vmax=image.max())
                            ax.set_title(f"finger {finger_i}, image {image_i}")
                    fig5.tight_layout()
                    fig5.show()

                    assert False, "PLOT_ONLY_ONE is True"

            # Save values
            if SAVE_DATASET:
                with loop_timer.add_section_timer("save values"):
                    nerf_densities = np.stack(nerf_densities, axis=0).reshape(
                        NUM_FINGERS, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z
                    )
                    transforms = np.stack(
                        [tt.matrix().cpu().numpy() for tt in transforms], axis=0
                    )

                    # Ensure no nans (most likely come from nerf densities)
                    if np.isnan(nerf_densities).any() or np.isnan(transforms).any():
                        print()
                        print("-" * 80)
                        print(
                            f"WARNING: Found {np.isnan(nerf_densities).sum()} nerf density nans and {np.isnan(transforms).sum()} transform nans in grasp {grasp_idx} of {config}"
                        )
                        print("Skipping this one...")
                        print("-" * 80)
                        print()
                        continue

                    nerf_densities_dataset[current_idx] = nerf_densities
                    grasp_success_dataset[current_idx] = evaled_grasp_config_dict[
                        "passed_eval"
                    ]
                    nerf_config_dataset[current_idx] = str(config)
                    object_code_dataset[current_idx] = object_code
                    object_scale_dataset[current_idx] = object_scale
                    grasp_idx_dataset[current_idx] = grasp_idx
                    grasp_transforms_dataset[current_idx] = transforms

                    current_idx += 1

                    # May not be max_num_data_points if nan grasps
                    hdf5_file.attrs["num_data_points"] = current_idx
        if PRINT_TIMING:
            loop_timer.pretty_print_section_times()
        print()

# %%
