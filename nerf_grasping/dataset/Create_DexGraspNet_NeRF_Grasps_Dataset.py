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
import nerfstudio
import sys
import pathlib
import h5py
import math
import matplotlib.pyplot as plt
import torch
import pypose as pp
import nerf_grasping
import os
import trimesh
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional
from nerf_grasping.dataset.DexGraspNet_NeRF_Grasps_utils import (
    plot_mesh_and_query_points,
    plot_mesh_and_transforms,
    plot_mesh_and_high_density_points,
    get_ray_samples_in_mesh_region,
    parse_object_code_and_scale,
)
from nerf_grasping.dataset.timers import LoopTimer
from nerf_grasping.optimizer_utils import AllegroHandConfig, AllegroGraspConfig
from nerf_grasping.grasp_utils import (
    get_ray_samples,
    get_ray_origins_finger_frame,
    get_nerf_configs,
    load_nerf,
)
from functools import partial
from nerf_grasping.config.nerfdata_config import (
    UnionNerfDataConfig,
    DepthImageNerfDataConfig,
    GridNerfDataConfig,
    GraspConditionedGridDataConfig,
)
import tyro
from localscope import localscope

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

os.chdir(nerf_grasping.get_repo_root())

tqdm = partial(std_tqdm, dynamic_ncols=True)


# %%
@localscope.mfc
def parse_nerf_config(nerf_config: pathlib.Path) -> str:
    # Input: PosixPath('2023-08-25_nerfcheckpoints/sem-Gun-4745991e7c0c7966a93f1ea6ebdeec6f_0_10/nerfacto/2023-08-25_132225/config.yml')
    # Return sem-Gun-4745991e7c0c7966a93f1ea6ebdeec6f_0_10
    parts = nerf_config.parts
    object_code_and_scale_str = parts[-4]
    return object_code_and_scale_str


@localscope.mfc(allowed=["tqdm"])
def count_total_num_grasps(
    nerf_configs: List[pathlib.Path], evaled_grasp_config_dicts_path: pathlib.Path
) -> int:
    ACTUALLY_COUNT_ALL = False
    total_num_grasps = 0

    for config in tqdm(nerf_configs, desc="counting num grasps", dynamic_ncols=True):
        # Read in grasp data
        object_code_and_scale_str = parse_nerf_config(config)
        object_code, object_scale = parse_object_code_and_scale(
            object_code_and_scale_str
        )
        evaled_grasp_config_dict_filepath = (
            evaled_grasp_config_dicts_path / f"{object_code_and_scale_str}.npy"
        )
        assert (
            evaled_grasp_config_dict_filepath.exists()
        ), f"evaled_grasp_config_dict_filepath {evaled_grasp_config_dict_filepath} does not exist"
        evaled_grasp_config_dict: Dict[str, Any] = np.load(
            evaled_grasp_config_dict_filepath, allow_pickle=True
        ).item()

        num_grasps = evaled_grasp_config_dict["trans"].shape[0]
        assert evaled_grasp_config_dict["trans"].shape == (
            num_grasps,
            3,
        )  # Sanity check

        # Count num_grasps
        if not ACTUALLY_COUNT_ALL:
            print(
                f"assuming all {len(nerf_configs)} evaled grasp config dicts have {num_grasps} grasps"
            )
            return num_grasps * len(nerf_configs)

        total_num_grasps += num_grasps
    return total_num_grasps


# WEIRD HACK SO YOU CAN STILL RUN VSC JUPYTER CELLS.
# %%
if __name__ == "__main__" and "get_ipython" not in dir():
    cfg: UnionNerfDataConfig = tyro.cli(UnionNerfDataConfig)
else:
    cfg: UnionNerfDataConfig = tyro.cli(UnionNerfDataConfig, args=[])

if isinstance(cfg, DepthImageNerfDataConfig):
    raise NotImplementedError("DepthImageNerfDataConfig not implemented yet")

# %%
if not cfg.output_filepath.parent.exists():
    print(f"Creating output folder {cfg.output_filepath.parent}")
    cfg.output_filepath.parent.mkdir(parents=True)
else:
    print(f"Output folder {cfg.output_filepath.parent} already exists")

# %%
if cfg.output_filepath.exists():
    print(f"Output file {cfg.output_filepath} already exists")
    assert False, "Output file already exists"


# %%
assert cfg.nerf_checkpoints_path.exists(), f"{cfg.nerf_checkpoints_path} does not exist"
nerf_configs = get_nerf_configs(
    nerf_checkpoints_path=str(cfg.nerf_checkpoints_path),
)
assert (
    len(nerf_configs) > 0
), f"Did not find any nerf configs in {cfg.nerf_checkpoints_path}"
print(f"Found {len(nerf_configs)} nerf configs")


# %%
ray_origins_finger_frame = get_ray_origins_finger_frame(cfg.fingertip_config)

# %%
if cfg.limit_num_configs is not None:
    print(f"Limiting number of configs to {cfg.limit_num_configs}")
nerf_configs = nerf_configs[: cfg.limit_num_configs]

if cfg.max_num_data_points_per_file is not None:
    max_num_datapoints = len(nerf_configs) * cfg.max_num_data_points_per_file
else:
    max_num_datapoints = count_total_num_grasps(
        nerf_configs=nerf_configs,
        evaled_grasp_config_dicts_path=cfg.evaled_grasp_config_dicts_path,
    )

print(f"max num datapoints: {max_num_datapoints}")

# %%
if cfg.save_dataset:
    print(f"Saving config to {cfg.config_filepath}")
    cfg_yaml = tyro.extras.to_yaml(cfg)
    with open(cfg.config_filepath, "w") as f:
        f.write(cfg_yaml)

print(cfg)

# %% [markdown]
# ## Define dataset creation functions and run.


@localscope.mfc
def create_grid_dataset(cfg: GridNerfDataConfig, hdf5_file: h5py.File, max_num_datapoints: int):
    nerf_densities_dataset = hdf5_file.create_dataset(
        "/nerf_densities",
        shape=(
            max_num_datapoints,
            cfg.fingertip_config.n_fingers,
            cfg.fingertip_config.num_pts_x,
            cfg.fingertip_config.num_pts_y,
            cfg.fingertip_config.num_pts_z,
        ),
        dtype="f",
        chunks=(
            1,
            cfg.fingertip_config.n_fingers,
            cfg.fingertip_config.num_pts_x,
            cfg.fingertip_config.num_pts_y,
            cfg.fingertip_config.num_pts_z,
        ),
    )
    grasp_success_dataset = hdf5_file.create_dataset(
        "/grasp_success", shape=(max_num_datapoints,), dtype="i"
    )
    nerf_config_dataset = hdf5_file.create_dataset(
        "/nerf_config", shape=(max_num_datapoints,), dtype=h5py.string_dtype()
    )
    object_code_dataset = hdf5_file.create_dataset(
        "/object_code", shape=(max_num_datapoints,), dtype=h5py.string_dtype()
    )
    object_scale_dataset = hdf5_file.create_dataset(
        "/object_scale", shape=(max_num_datapoints,), dtype="f"
    )
    grasp_idx_dataset = hdf5_file.create_dataset(
        "/grasp_idx", shape=(max_num_datapoints,), dtype="i"
    )
    grasp_transforms_dataset = hdf5_file.create_dataset(
        "/grasp_transforms",
        shape=(max_num_datapoints, cfg.fingertip_config.n_fingers, 4, 4),
        dtype="f",
    )

    return (
        nerf_densities_dataset,
        grasp_success_dataset,
        nerf_config_dataset,
        object_code_dataset,
        object_scale_dataset,
        grasp_idx_dataset,
        grasp_transforms_dataset,
    )


@localscope.mfc
def create_depth_image_dataset(cfg: DepthImageNerfDataConfig, hdf5_file: h5py.File, max_num_datapoints: int):
    nerf_densities_dataset = hdf5_file.create_dataset(
        "/depth_images",
        shape=(
            max_num_datapoints,
            cfg.fingertip_config.n_fingers,
            cfg.fingertip_camera_config.H,
            cfg.fingertip_camera_config.W,
        ),
        dtype="f",
        chunks=(
            1,
            cfg.fingertip_config.n_fingers,
            cfg.fingertip_camera_config.H,
            cfg.fingertip_camera_config.W,
        ),
    )
    grasp_success_dataset = hdf5_file.create_dataset(
        "/grasp_success", shape=(max_num_datapoints,), dtype="i"
    )
    nerf_config_dataset = hdf5_file.create_dataset(
        "/nerf_config", shape=(max_num_datapoints,), dtype=h5py.string_dtype()
    )
    object_code_dataset = hdf5_file.create_dataset(
        "/object_code", shape=(max_num_datapoints,), dtype=h5py.string_dtype()
    )
    object_scale_dataset = hdf5_file.create_dataset(
        "/object_scale", shape=(max_num_datapoints,), dtype="f"
    )
    grasp_idx_dataset = hdf5_file.create_dataset(
        "/grasp_idx", shape=(max_num_datapoints,), dtype="i"
    )
    grasp_transforms_dataset = hdf5_file.create_dataset(
        "/grasp_transforms",
        shape=(max_num_datapoints, cfg.fingertip_config.n_fingers, 4, 4),
        dtype="f",
    )

    return (
        nerf_densities_dataset,
        grasp_success_dataset,
        nerf_config_dataset,
        object_code_dataset,
        object_scale_dataset,
        grasp_idx_dataset,
        grasp_transforms_dataset,
    )


@localscope.mfc
def get_nerf_densities(
    loop_timer: LoopTimer,
    cfg: UnionNerfDataConfig,
    grasp_frame_transforms: pp.LieTensor,
    ray_origins_finger_frame: torch.Tensor,
    nerf_model: nerfstudio.models.base_model.Model,
) -> Tuple[np.ndarray, np.ndarray]:
    assert grasp_frame_transforms.lshape == (cfg.fingertip_config.n_fingers,)

    transform_list = [
        grasp_frame_transforms[i].detach().clone()
        for i in range(cfg.fingertip_config.n_fingers)
    ]

    # Create density grid for grid dataset.
    if isinstance(cfg, GridNerfDataConfig) or isinstance(
        cfg, GraspConditionedGridDataConfig
    ):
        # Transform query points
        with loop_timer.add_section_timer("get_ray_samples"):
            ray_samples_list = [
                get_ray_samples(
                    ray_origins_finger_frame, transform, cfg.fingertip_config
                )
                for transform in transform_list
            ]

        with loop_timer.add_section_timer("frustums.get_positions"):
            query_points_list = [
                np.copy(
                    rr.frustums.get_positions()
                    .cpu()
                    .numpy()
                    .reshape(
                        cfg.fingertip_config.num_pts_x,
                        cfg.fingertip_config.num_pts_y,
                        cfg.fingertip_config.num_pts_z,
                        3,
                    )
                )  # Shape [n_x, n_y, n_z, 3]
                for rr in ray_samples_list
            ]

        # Get densities
        with loop_timer.add_section_timer("get_density"):
            nerf_densities = [
                nerf_model.get_density(ray_samples.to("cuda"))[0]
                .detach()
                .cpu()
                .numpy()
                .reshape(
                    cfg.fingertip_config.num_pts_x,
                    cfg.fingertip_config.num_pts_y,
                    cfg.fingertip_config.num_pts_z,
                )
                for ray_samples in ray_samples_list  # Shape [n_x, n_y, n_z].
            ]

        query_points_list = np.stack(query_points_list, axis=0).reshape(
            cfg.fingertip_config.n_fingers,
            cfg.fingertip_config.num_pts_x,
            cfg.fingertip_config.num_pts_y,
            cfg.fingertip_config.num_pts_z,
            3,
        )
        nerf_densities = np.stack(nerf_densities, axis=0).reshape(
            cfg.fingertip_config.n_fingers,
            cfg.fingertip_config.num_pts_x,
            cfg.fingertip_config.num_pts_y,
            cfg.fingertip_config.num_pts_z,
        )

        return nerf_densities, query_points_list


with h5py.File(cfg.output_filepath, "w") as hdf5_file:
    current_idx = 0

    if isinstance(cfg, GraspConditionedGridDataConfig):
        # Create dataset with extra field for full grasp config.
        (
            nerf_densities_dataset,
            grasp_success_dataset,
            nerf_config_dataset,
            object_code_dataset,
            object_scale_dataset,
            grasp_idx_dataset,
            grasp_transforms_dataset,
        ) = create_grid_dataset(cfg, hdf5_file, max_num_datapoints)
        conditioning_var_dataset = hdf5_file.create_dataset(
            "/conditioning_var",
            shape=(
                max_num_datapoints,
                cfg.fingertip_config.n_fingers,
                7 + 16 + 4,
            ),  # 7 for pose, 16 for rotation matrix, 4 for grasp orientation, for each finger
            dtype="f",
        )
    elif isinstance(cfg, GridNerfDataConfig):
        (
            nerf_densities_dataset,
            grasp_success_dataset,
            nerf_config_dataset,
            object_code_dataset,
            object_scale_dataset,
            grasp_idx_dataset,
            grasp_transforms_dataset,
        ) = create_grid_dataset(cfg, hdf5_file, max_num_datapoints)
    elif isinstance(cfg, DepthImageNerfDataConfig):
        (
            nerf_densities_dataset,
            grasp_success_dataset,
            nerf_config_dataset,
            object_code_dataset,
            object_scale_dataset,
            grasp_idx_dataset,
            grasp_transforms_dataset,
        ) = create_depth_image_dataset(cfg, hdf5_file, max_num_datapoints)
    else:
        raise NotImplementedError(f"Unknown config type {cfg}")

    # Slice out the grasp index we want to visualize if plot_only_one is True.
    if cfg.plot_only_one:
        assert cfg.nerf_visualize_index is not None
        assert cfg.nerf_visualize_index < len(
            nerf_configs
        ), f"Visualize index out of bounds"
        nerf_configs = nerf_configs[
            cfg.nerf_visualize_index : cfg.nerf_visualize_index + 1
        ]

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
                cfg.dexgraspnet_meshdata_root / object_code / "coacd" / "decomposed.obj"
            )
            evaled_grasp_config_dict_filepath = (
                cfg.evaled_grasp_config_dicts_path / f"{object_code_and_scale_str}.npy"
            )

            # Check that mesh and grasp dataset exist
            assert mesh_path.exists(), f"mesh_path {mesh_path} does not exist"
            assert os.path.exists(
                evaled_grasp_config_dict_filepath
            ), f"evaled_grasp_config_dict_filepath {evaled_grasp_config_dict_filepath} does not exist"

        # Read in data
        with loop_timer.add_section_timer("load_nerf"):
            nerf_model = load_nerf(config)

        with loop_timer.add_section_timer("load mesh"):
            mesh = trimesh.load(mesh_path, force="mesh")
            mesh.apply_transform(trimesh.transformations.scale_matrix(object_scale))

        with loop_timer.add_section_timer("load grasp data"):
            evaled_grasp_config_dict: Dict[str, Any] = np.load(
                evaled_grasp_config_dict_filepath, allow_pickle=True
            ).item()

        # Extract useful parts of grasp data
        grasp_configs = AllegroGraspConfig.from_grasp_config_dict(
            evaled_grasp_config_dict
        )
        grasp_successes = evaled_grasp_config_dict["passed_eval"]

        # If plot_only_one is True, slice out the grasp index we want to visualize.
        if cfg.plot_only_one:
            assert cfg.grasp_visualize_index is not None
            assert (
                cfg.grasp_visualize_index < grasp_configs.batch_size
            ), f"{cfg.grasp_visualize_index} out of bounds for batch size {grasp_configs.batch_size}"
            grasp_configs = grasp_configs[
                cfg.grasp_visualize_index : cfg.grasp_visualize_index + 1
            ]
            grasp_successes = grasp_successes[
                cfg.grasp_visualize_index : cfg.grasp_visualize_index + 1
            ]

        if (
            cfg.max_num_data_points_per_file is not None
            and grasp_configs.batch_size > cfg.max_num_data_points_per_file
        ):
            print(
                "WARNING: Too many grasp configs, dropping some datapoints from NeRF dataset."
            )
            print(
                f"batch_size = {grasp_configs.batch_size}, cfg.max_num_data_points_per_file = {cfg.max_num_data_points_per_file}"
            )

        grasp_configs = grasp_configs[:max_num_datapoints]
        grasp_successes = grasp_successes[:max_num_datapoints]
        grasp_frame_transforms_arr = grasp_configs.grasp_frame_transforms

        assert grasp_successes.shape == (grasp_configs.batch_size,)
        assert grasp_frame_transforms_arr.lshape == (
            grasp_configs.batch_size,
            cfg.fingertip_config.n_fingers,
        )

        if isinstance(cfg, GraspConditionedGridDataConfig):
            grasp_config_tensors = grasp_configs.as_tensor()

        for grasp_idx, (grasp_success, grasp_frame_transforms) in (
            pbar := tqdm(
                enumerate(
                    zip(
                        grasp_successes,
                        grasp_frame_transforms_arr,
                    )
                ),
                total=grasp_configs.batch_size,
                dynamic_ncols=True,
            )
        ):
            pbar.set_description(f"grasp data, current_idx: {current_idx}")
            nerf_densities, query_points_list = get_nerf_densities(
                loop_timer=loop_timer,
                cfg=cfg,
                grasp_frame_transforms=grasp_frame_transforms,
                ray_origins_finger_frame=ray_origins_finger_frame,
                nerf_model=nerf_model,
            )
            if cfg.plot_only_one:
                break

            # Ensure no nans (most likely come from weird grasp transforms)
            if (
                np.isnan(nerf_densities).any()
                or torch.isnan(grasp_frame_transforms).any()
            ):
                print("\n" + "-" * 80)
                print(
                    f"WARNING: Found {np.isnan(nerf_densities).sum()} nerf density nans and {torch.isnan(grasp_frame_transforms).sum()} transform nans in grasp {grasp_idx} of {config}"
                )
                print("Skipping this one...")
                print("-" * 80 + "\n")
                continue

            # Save values
            if not cfg.save_dataset:
                continue
            with loop_timer.add_section_timer("save values"):
                nerf_densities_dataset[current_idx] = nerf_densities
                grasp_success_dataset[current_idx] = grasp_success
                nerf_config_dataset[current_idx] = str(config)
                object_code_dataset[current_idx] = object_code
                object_scale_dataset[current_idx] = object_scale
                grasp_idx_dataset[current_idx] = grasp_idx
                grasp_transforms_dataset[current_idx] = (
                    grasp_frame_transforms.matrix().cpu().detach().numpy()
                )

                if isinstance(cfg, GraspConditionedGridDataConfig):
                    grasp_config_tensor = (
                        grasp_config_tensors[grasp_idx].detach().cpu().numpy()
                    )
                    assert grasp_config_tensor.shape == (
                        cfg.fingertip_config.n_fingers,
                        7 + 16,
                    )
                    conditioning_var_dataset[current_idx] = grasp_config_tensor

                current_idx += 1

                # May not be max_num_data_points if nan grasps
                hdf5_file.attrs["num_data_points"] = current_idx

        if cfg.print_timing:
            loop_timer.pretty_print_section_times()
        print()

# %%
if not cfg.plot_only_one:
    print("Done!")
    sys.exit()

grasp_frame_transforms = grasp_frame_transforms.matrix().cpu().detach().numpy()
# Plot
delta = (
    cfg.fingertip_config.grasp_depth_mm / 1000 / (cfg.fingertip_config.num_pts_z - 1)
)

nerf_alphas = [1 - np.exp(-delta * dd) for dd in nerf_densities]
fig = plot_mesh_and_query_points(
    mesh=mesh,
    query_points_list=[qq.reshape(-1, 3) for qq in query_points_list],
    query_points_colors_list=[x.reshape(-1) for x in nerf_alphas],
    num_fingers=cfg.fingertip_config.n_fingers,
    title=f"Mesh and Query Points, Success: {grasp_success}",
)
fig.show()
fig2 = plot_mesh_and_transforms(
    mesh=mesh,
    transforms=[
        grasp_frame_transforms[i]
        for i in range(cfg.fingertip_config.n_fingers)
    ],
    num_fingers=cfg.fingertip_config.n_fingers,
    title=f"Mesh and Transforms, Success: {grasp_success}",
)
fig2.show()

if cfg.plot_all_high_density_points:
    PLOT_NUM_PTS_X, PLOT_NUM_PTS_Y, PLOT_NUM_PTS_Z = 100, 100, 100
    ray_samples_in_mesh_region = get_ray_samples_in_mesh_region(
        mesh=mesh,
        num_pts_x=PLOT_NUM_PTS_X,
        num_pts_y=PLOT_NUM_PTS_Y,
        num_pts_z=PLOT_NUM_PTS_Z,
    )
    query_points_in_mesh_region_isaac_frame = np.copy(
        ray_samples_in_mesh_region.frustums.get_positions()
        .cpu()
        .numpy()
        .reshape(
            PLOT_NUM_PTS_X,
            PLOT_NUM_PTS_Y,
            PLOT_NUM_PTS_Z,
            3,
        )
    )
    nerf_densities_in_mesh_region = (
        nerf_model.get_density(ray_samples_in_mesh_region.to("cuda"))[0]
        .detach()
        .cpu()
        .numpy()
        .reshape(
            PLOT_NUM_PTS_X,
            PLOT_NUM_PTS_Y,
            PLOT_NUM_PTS_Z,
        )
    )

    nerf_alphas_in_mesh_region = 1 - np.exp(-delta * nerf_densities_in_mesh_region)

    fig3 = plot_mesh_and_high_density_points(
        mesh=mesh,
        query_points=query_points_in_mesh_region_isaac_frame.reshape(-1, 3),
        query_points_colors=nerf_alphas_in_mesh_region.reshape(-1),
        density_threshold=0.01,
    )
    fig3.show()

if cfg.plot_alphas_each_finger_1D:
    nrows, ncols = cfg.fingertip_config.n_fingers, 1
    fig4, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    axes = axes.flatten()
    for i in range(cfg.fingertip_config.n_fingers):
        ax = axes[i]
        finger_alphas = nerf_alphas[i].reshape(
            cfg.fingertip_config.num_pts_x,
            cfg.fingertip_config.num_pts_y,
            cfg.fingertip_config.num_pts_z,
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

if cfg.plot_alpha_images_each_finger:
    num_images = 5
    nrows, ncols = cfg.fingertip_config.n_fingers, num_images
    fig5, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    alpha_images = [
        x.reshape(
            cfg.fingertip_config.num_pts_x,
            cfg.fingertip_config.num_pts_y,
            cfg.fingertip_config.num_pts_z,
        )
        for x in nerf_alphas
    ]

    for finger_i in range(cfg.fingertip_config.n_fingers):
        for image_i in range(num_images):
            ax = axes[finger_i, image_i]
            image = alpha_images[finger_i][
                :,
                :,
                int(image_i * cfg.fingertip_config.num_pts_z / num_images),
            ]
            ax.imshow(
                image,
                vmin=nerf_alphas[i].min(),
                vmax=nerf_alphas[i].max(),
            )
            ax.set_title(f"finger {finger_i}, image {image_i}")
    fig5.tight_layout()
    fig5.show()
    plt.show(block=True)

assert False, "cfg.plot_only_one is True"
