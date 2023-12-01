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
import sys
import pathlib
import h5py
import matplotlib.pyplot as plt
import torch
import pypose as pp
import nerf_grasping
import os
import trimesh
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Dict, Any
from nerf_grasping.dataset.DexGraspNet_NeRF_Grasps_utils import (
    plot_mesh_and_query_points,
    plot_mesh_and_transforms,
    plot_mesh_and_high_density_points,
    get_ray_samples_in_mesh_region,
    parse_object_code_and_scale,
)
from nerf_grasping.dataset.timers import LoopTimer
from nerf_grasping.optimizer_utils import AllegroGraspConfig
from nerf_grasping.grasp_utils import (
    get_ray_samples,
    get_ray_origins_finger_frame,
    get_nerf_configs,
    load_nerf_pipeline,
)
from nerf_grasping.nerf_utils import (
    get_cameras,
    render,
)
from nerf_grasping.config.base import CONFIG_DATETIME_STR
from functools import partial
from nerf_grasping.config.nerfdata_config import (
    UnionNerfDataConfig,
    DepthImageNerfDataConfig,
    GridNerfDataConfig,
    BaseNerfDataConfig,
)
import tyro
from localscope import localscope
from nerfstudio.fields.base_field import Field
from nerfstudio.models.base_model import Model


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
if is_notebook():
    from tqdm.notebook import tqdm as std_tqdm
else:
    from tqdm import tqdm as std_tqdm

os.chdir(nerf_grasping.get_repo_root())

tqdm = partial(std_tqdm, dynamic_ncols=True)


# %%
@localscope.mfc
def nerf_config_to_object_code_and_scale_str(nerf_config: pathlib.Path) -> str:
    # Input: PosixPath('2023-08-25_nerfcheckpoints/sem-Gun-4745991e7c0c7966a93f1ea6ebdeec6f_0_10/nerfacto/2023-08-25_132225/config.yml')
    # Return sem-Gun-4745991e7c0c7966a93f1ea6ebdeec6f_0_10
    parts = nerf_config.parts
    object_code_and_scale_str = parts[-4]
    return object_code_and_scale_str


@localscope.mfc
def get_closest_matching_nerf_config(
    target_object_code_and_scale_str: str, nerf_configs: List[pathlib.Path]
) -> pathlib.Path:
    # Parse target object code and scale
    target_object_code, target_scale = parse_object_code_and_scale(
        target_object_code_and_scale_str
    )

    # Prepare data for comparisons
    nerf_object_code_and_scale_strs = [
        nerf_config_to_object_code_and_scale_str(config) for config in nerf_configs
    ]
    nerf_object_codes = [
        parse_object_code_and_scale(object_code_and_scale_str)[0]
        for object_code_and_scale_str in nerf_object_code_and_scale_strs
    ]
    nerf_object_scales = [
        parse_object_code_and_scale(object_code_and_scale_str)[1]
        for object_code_and_scale_str in nerf_object_code_and_scale_strs
    ]

    # Check for exact match
    exact_matches = [
        nerf_config
        for object_code_and_scale_str, nerf_config in zip(
            nerf_object_code_and_scale_strs, nerf_configs
        )
        if target_object_code_and_scale_str == object_code_and_scale_str
    ]
    if exact_matches:
        if len(exact_matches) > 1:
            print(
                f"Multiple exact matches found for {target_object_code_and_scale_str}, {exact_matches}"
            )
        return exact_matches[0]

    print(
        f"No exact matches found for {target_object_code_and_scale_str}. Searching for closest matches..."
    )

    # Check for closest scale match
    same_code_configs = [
        config
        for code, config in zip(nerf_object_codes, nerf_configs)
        if code == target_object_code
    ]
    if same_code_configs:
        scale_diffs = [
            abs(scale - target_scale)
            for code, scale in zip(nerf_object_codes, nerf_object_scales)
            if code == target_object_code
        ]
        closest_scale_config = same_code_configs[np.argmin(scale_diffs)]
        return closest_scale_config

    print(f"No configs found with code {target_object_code}.")

    # Check for closest object code match
    import Levenshtein

    levenshtein_distances = [
        Levenshtein.distance(
            target_object_code_and_scale_str, object_code_and_scale_str
        )
        for object_code_and_scale_str in nerf_object_code_and_scale_strs
    ]
    min_distance_index = np.argmin(levenshtein_distances)
    if levenshtein_distances[min_distance_index] < 10:
        return nerf_configs[min_distance_index]

    raise ValueError(
        f"No suitable NeRF config found for {target_object_code_and_scale_str}, min dist = {levenshtein_distances[min_distance_index]}"
    )


@localscope.mfc(allowed=["tqdm"])
def count_total_num_grasps(
    evaled_grasp_config_dict_filepaths: List[pathlib.Path],
) -> int:
    ACTUALLY_COUNT_ALL = False
    total_num_grasps = 0

    for evaled_grasp_config_dict_filepath in tqdm(
        evaled_grasp_config_dict_filepaths,
        desc="counting num grasps",
        dynamic_ncols=True,
    ):
        # Read in grasp data
        assert (
            evaled_grasp_config_dict_filepath.exists()
        ), f"evaled_grasp_config_dict_filepath {evaled_grasp_config_dict_filepath} does not exist"
        evaled_grasp_config_dict: Dict[str, Any] = np.load(
            evaled_grasp_config_dict_filepath, allow_pickle=True
        ).item()

        num_grasps = evaled_grasp_config_dict["trans"].shape[0]
        assert_equals(
            evaled_grasp_config_dict["trans"].shape,
            (
                num_grasps,
                3,
            ),
        )  # Sanity check

        # Count num_grasps
        if not ACTUALLY_COUNT_ALL:
            print(
                f"assuming all {len(evaled_grasp_config_dict_filepaths)} evaled grasp config dicts have {num_grasps} grasps"
            )
            return num_grasps * len(evaled_grasp_config_dict_filepaths)

        total_num_grasps += num_grasps
    return total_num_grasps


# WEIRD HACK SO YOU CAN STILL RUN VSC JUPYTER CELLS.
# %%
if __name__ == "__main__" and "get_ipython" not in dir():
    cfg: BaseNerfDataConfig = tyro.cli(UnionNerfDataConfig)
else:
    cfg: BaseNerfDataConfig = tyro.cli(UnionNerfDataConfig, args=["depth-image"])

print(f"Config:\n{tyro.extras.to_yaml(cfg)}")
assert cfg.fingertip_config is not None

# %%
if cfg.output_filepath is None:
    cfg.output_filepath = (
        cfg.evaled_grasp_config_dicts_path.parent
        / "learned_metric_dataset"
        / f"{CONFIG_DATETIME_STR}_learned_metric_dataset.h5"
    )
assert cfg.output_filepath is not None
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
assert (
    cfg.evaled_grasp_config_dicts_path.exists()
), f"{cfg.evaled_grasp_config_dicts_path} does not exist"
evaled_grasp_config_dict_filepaths = list(
    cfg.evaled_grasp_config_dicts_path.glob("*.npy")
)

# from glob import glob
# evaled_grasp_config_dict_filepaths = [
#     pathlib.Path(x)
#     for x in glob(
#         str(
#             cfg.evaled_grasp_config_dicts_path
#             / "**"
#             / "evaled_grasp_config_dicts"
#             / "*.npy"
#         ),
#         recursive=True,
#     )
# ]
assert (
    len(evaled_grasp_config_dict_filepaths) > 0
), f"Did not find any evaled grasp config dicts in {cfg.evaled_grasp_config_dicts_path}"
print(f"Found {len(evaled_grasp_config_dict_filepaths)} evaled grasp config dicts")

# %%
ray_origins_finger_frame = get_ray_origins_finger_frame(cfg.fingertip_config)

# %%
if cfg.limit_num_configs is not None:
    print(f"Limiting number of configs to {cfg.limit_num_configs}")
evaled_grasp_config_dict_filepaths = evaled_grasp_config_dict_filepaths[
    : cfg.limit_num_configs
]

if cfg.max_num_data_points_per_file is not None:
    max_num_datapoints = (
        len(evaled_grasp_config_dict_filepaths) * cfg.max_num_data_points_per_file
    )
else:
    max_num_datapoints = count_total_num_grasps(
        evaled_grasp_config_dict_filepaths=evaled_grasp_config_dict_filepaths,
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
def create_grid_dataset(
    cfg: GridNerfDataConfig, hdf5_file: h5py.File, max_num_datapoints: int
) -> Tuple[h5py.Dataset, ...]:
    assert cfg.fingertip_config is not None

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
    passed_eval_dataset = hdf5_file.create_dataset(
        "/passed_eval", shape=(max_num_datapoints,), dtype="f"
    )
    passed_simulation_dataset = hdf5_file.create_dataset(
        "/passed_simulation",
        shape=(max_num_datapoints,),
        dtype="f",
    )
    passed_penetration_threshold_dataset = hdf5_file.create_dataset(
        "/passed_penetration_threshold",
        shape=(max_num_datapoints,),
        dtype="f",
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
    grasp_configs_dataset = hdf5_file.create_dataset(
        "/grasp_configs",
        shape=(
            max_num_datapoints,
            cfg.fingertip_config.n_fingers,
            7 + 16 + 4,
        ),  # 7 for pose, 16 for joint angles, 4 for grasp orientation, for each finger
        dtype="f",
    )

    return (
        nerf_densities_dataset,
        passed_eval_dataset,
        passed_simulation_dataset,
        passed_penetration_threshold_dataset,
        nerf_config_dataset,
        object_code_dataset,
        object_scale_dataset,
        grasp_idx_dataset,
        grasp_transforms_dataset,
        grasp_configs_dataset,
    )


@localscope.mfc
def create_depth_image_dataset(
    cfg: DepthImageNerfDataConfig, hdf5_file: h5py.File, max_num_datapoints: int
) -> Tuple[h5py.Dataset, ...]:
    assert cfg.fingertip_config is not None

    depth_images_dataset = hdf5_file.create_dataset(
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
    uncertainty_images_dataset = hdf5_file.create_dataset(
        "/uncertainty_images",
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
    passed_eval_dataset = hdf5_file.create_dataset(
        "/passed_eval", shape=(max_num_datapoints,), dtype="f"
    )
    passed_simulation_dataset = hdf5_file.create_dataset(
        "/passed_simulation",
        shape=(max_num_datapoints,),
        dtype="f",
    )
    passed_penetration_threshold_dataset = hdf5_file.create_dataset(
        "/passed_penetration_threshold",
        shape=(max_num_datapoints,),
        dtype="f",
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
    grasp_configs_dataset = hdf5_file.create_dataset(
        "/grasp_configs",
        shape=(
            max_num_datapoints,
            cfg.fingertip_config.n_fingers,
            7 + 16 + 4,
        ),  # 7 for wrist pose, 16 for joint angles, 4 for grasp orientation, for each finger
        dtype="f",
    )

    return (
        depth_images_dataset,
        uncertainty_images_dataset,
        passed_eval_dataset,
        passed_simulation_dataset,
        passed_penetration_threshold_dataset,
        nerf_config_dataset,
        object_code_dataset,
        object_scale_dataset,
        grasp_idx_dataset,
        grasp_transforms_dataset,
        grasp_configs_dataset,
    )


@torch.no_grad()
@localscope.mfc(allowed=["tqdm"])
def get_depth_and_uncertainty_images(
    loop_timer: LoopTimer,
    cfg: BaseNerfDataConfig,
    grasp_frame_transforms: pp.LieTensor,
    nerf_model: Model,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert cfg.fingertip_config is not None
    assert isinstance(cfg, DepthImageNerfDataConfig)

    with loop_timer.add_section_timer("get_cameras"):
        cameras = get_cameras(grasp_frame_transforms, cfg.fingertip_camera_config)

    batch_size = cameras.shape[0]
    with loop_timer.add_section_timer("render"):
        # Split cameras into chunks so everything fits on the gpu
        split_inds = torch.arange(0, batch_size, cfg.cameras_samples_chunk_size)
        split_inds = torch.cat(
            [split_inds, torch.tensor([batch_size]).to(split_inds.device)]
        )
        depths, uncertainties = [], []
        for curr_ind, next_ind in tqdm(
            zip(split_inds[:-1], split_inds[1:]),
            total=len(split_inds) - 1,
            desc="render",
            dynamic_ncols=True,
        ):
            curr_cameras = cameras[curr_ind:next_ind].to("cuda")
            curr_depth, curr_uncertainty = render(
                curr_cameras, nerf_model, "median", far_plane=0.15
            )
            assert (
                curr_depth.shape
                == curr_uncertainty.shape
                == (
                    cfg.fingertip_camera_config.H,
                    cfg.fingertip_camera_config.W,
                    curr_cameras.shape[0] * cfg.fingertip_config.n_fingers,
                )
            )
            depths.append(curr_depth.cpu())
            uncertainties.append(curr_uncertainty.cpu())
            curr_cameras.to("cpu")

        depths = torch.cat(depths, dim=-1)
        uncertainties = torch.cat(uncertainties, dim=-1)
        assert (
            depths.shape
            == uncertainties.shape
            == (
                cfg.fingertip_camera_config.H,
                cfg.fingertip_camera_config.W,
                batch_size * cfg.fingertip_config.n_fingers,
            )
        )

    return (
        depths.permute(2, 0, 1).reshape(
            batch_size,
            cfg.fingertip_config.n_fingers,
            cfg.fingertip_camera_config.H,
            cfg.fingertip_camera_config.W,
        ),
        uncertainties.permute(2, 0, 1).reshape(
            batch_size,
            cfg.fingertip_config.n_fingers,
            cfg.fingertip_camera_config.H,
            cfg.fingertip_camera_config.W,
        ),
    )


@torch.no_grad()
@localscope.mfc(allowed=["tqdm"])
def get_nerf_densities(
    loop_timer: LoopTimer,
    cfg: BaseNerfDataConfig,
    grasp_frame_transforms: pp.LieTensor,
    ray_origins_finger_frame: torch.Tensor,
    nerf_field: Field,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert cfg.fingertip_config is not None

    # Shape check grasp_frame_transforms
    batch_size = grasp_frame_transforms.shape[0]
    assert_equals(
        grasp_frame_transforms.lshape,
        (
            batch_size,
            cfg.fingertip_config.n_fingers,
        ),
    )

    # Create density grid for grid dataset.
    assert isinstance(cfg, GridNerfDataConfig)

    # Transform query points
    with loop_timer.add_section_timer("get_ray_samples"):
        ray_samples = get_ray_samples(
            ray_origins_finger_frame,
            grasp_frame_transforms,
            cfg.fingertip_config,
        )

    with loop_timer.add_section_timer("frustums.get_positions"):
        query_points = ray_samples.frustums.get_positions().reshape(
            batch_size,
            cfg.fingertip_config.n_fingers,
            cfg.fingertip_config.num_pts_x,
            cfg.fingertip_config.num_pts_y,
            cfg.fingertip_config.num_pts_z,
            3,
        )

    with loop_timer.add_section_timer("get_density"):
        # Split ray_samples into chunks so everything fits on the gpu
        split_inds = torch.arange(0, batch_size, cfg.ray_samples_chunk_size)
        split_inds = torch.cat(
            [split_inds, torch.tensor([batch_size]).to(split_inds.device)]
        )
        nerf_density_list = []
        for curr_ind, next_ind in tqdm(
            zip(split_inds[:-1], split_inds[1:]),
            total=len(split_inds) - 1,
            desc="get_density",
            dynamic_ncols=True,
        ):
            curr_ray_samples = ray_samples[curr_ind:next_ind].to("cuda")
            nerf_density_list.append(
                nerf_field.get_density(curr_ray_samples)[0]
                .reshape(
                    -1,
                    cfg.fingertip_config.n_fingers,
                    cfg.fingertip_config.num_pts_x,
                    cfg.fingertip_config.num_pts_y,
                    cfg.fingertip_config.num_pts_z,
                )
                .cpu()
            )
            curr_ray_samples.to("cpu")
        nerf_densities = torch.cat(nerf_density_list, dim=0)

    return nerf_densities, query_points


with h5py.File(cfg.output_filepath, "w") as hdf5_file:
    current_idx = 0

    if isinstance(cfg, GridNerfDataConfig):
        (
            nerf_densities_dataset,
            passed_eval_dataset,
            passed_simulation_dataset,
            passed_penetration_threshold_dataset,
            nerf_config_dataset,
            object_code_dataset,
            object_scale_dataset,
            grasp_idx_dataset,
            grasp_transforms_dataset,
            grasp_configs_dataset,
        ) = create_grid_dataset(cfg, hdf5_file, max_num_datapoints)
    elif isinstance(cfg, DepthImageNerfDataConfig):
        (
            depth_images_dataset,
            uncertainty_images_dataset,
            passed_eval_dataset,
            passed_simulation_dataset,
            passed_penetration_threshold_dataset,
            nerf_config_dataset,
            object_code_dataset,
            object_scale_dataset,
            grasp_idx_dataset,
            grasp_transforms_dataset,
            grasp_configs_dataset,
        ) = create_depth_image_dataset(cfg, hdf5_file, max_num_datapoints)
    else:
        raise NotImplementedError(f"Unknown config type {cfg}")

    # Slice out the grasp index we want to visualize if plot_only_one is True.
    if cfg.plot_only_one:
        assert cfg.config_dict_visualize_index is not None
        assert cfg.config_dict_visualize_index < len(
            evaled_grasp_config_dict_filepaths
        ), f"Visualize index out of bounds"
        evaled_grasp_config_dict_filepaths = evaled_grasp_config_dict_filepaths[
            cfg.config_dict_visualize_index : cfg.config_dict_visualize_index + 1
        ]

    # Iterate through all
    loop_timer = LoopTimer()
    pbar = tqdm(
        evaled_grasp_config_dict_filepaths,
        dynamic_ncols=True,
    )
    for evaled_grasp_config_dict_filepath in pbar:
        pbar.set_description(f"Processing {evaled_grasp_config_dict_filepath}")
        try:
            with loop_timer.add_section_timer("prepare to read in data"):
                object_code_and_scale_str = evaled_grasp_config_dict_filepath.stem
                object_code, object_scale = parse_object_code_and_scale(
                    object_code_and_scale_str
                )

                # Get nerf config
                nerf_config = get_closest_matching_nerf_config(
                    target_object_code_and_scale_str=object_code_and_scale_str,
                    nerf_configs=nerf_configs,
                )

                # Check that mesh and grasp dataset exist
                assert os.path.exists(
                    evaled_grasp_config_dict_filepath
                ), f"evaled_grasp_config_dict_filepath {evaled_grasp_config_dict_filepath} does not exist"

            # Read in data
            with loop_timer.add_section_timer("load_nerf"):
                # Load nerf pipeline
                # Note: for some reason this helps avoid GPU memory leak
                #       whereas loading nerf_model or nerf_field directly causes GPU memory leak
                nerf_pipeline = load_nerf_pipeline(nerf_config)

            with loop_timer.add_section_timer("load grasp data"):
                evaled_grasp_config_dict: Dict[str, Any] = np.load(
                    evaled_grasp_config_dict_filepath, allow_pickle=True
                ).item()

            # Extract useful parts of grasp data
            grasp_configs = AllegroGraspConfig.from_grasp_config_dict(
                evaled_grasp_config_dict
            )
            passed_evals = evaled_grasp_config_dict["passed_eval"]
            passed_simulations = evaled_grasp_config_dict["passed_simulation"]
            passed_penetration_thresholds = evaled_grasp_config_dict[
                "passed_penetration_threshold"
            ]

            # If plot_only_one is True, slice out the grasp index we want to visualize.
            if cfg.plot_only_one:
                assert cfg.grasp_visualize_index is not None
                assert (
                    cfg.grasp_visualize_index < grasp_configs.batch_size
                ), f"{cfg.grasp_visualize_index} out of bounds for batch size {grasp_configs.batch_size}"
                grasp_configs = grasp_configs[
                    cfg.grasp_visualize_index : cfg.grasp_visualize_index + 1
                ]
                passed_evals = passed_evals[
                    cfg.grasp_visualize_index : cfg.grasp_visualize_index + 1
                ]

                passed_simulations = passed_simulations[
                    cfg.grasp_visualize_index : cfg.grasp_visualize_index + 1
                ]
                passed_penetration_thresholds = passed_penetration_thresholds[
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

            passed_evals = passed_evals[:max_num_datapoints]
            passed_simulations = passed_simulations[:max_num_datapoints]
            passed_penetration_thresholds = passed_penetration_thresholds[
                :max_num_datapoints
            ]
            grasp_frame_transforms = grasp_configs.grasp_frame_transforms

            assert_equals(passed_evals.shape, (grasp_configs.batch_size,))
            assert_equals(
                grasp_frame_transforms.lshape,
                (
                    grasp_configs.batch_size,
                    cfg.fingertip_config.n_fingers,
                ),
            )

            if isinstance(cfg, GridNerfDataConfig):
                # Process batch of grasp data.
                nerf_densities, query_points = get_nerf_densities(
                    loop_timer=loop_timer,
                    cfg=cfg,
                    grasp_frame_transforms=grasp_frame_transforms,
                    ray_origins_finger_frame=ray_origins_finger_frame,
                    nerf_field=nerf_pipeline.model.field,
                )
                if nerf_densities.isnan().any():
                    print("\n" + "-" * 80)
                    print(
                        f"WARNING: Found {nerf_densities.isnan().sum()} nerf density nans in {nerf_config}"
                    )
                    print("Skipping this one...")
                    print("-" * 80 + "\n")
                    continue

            elif isinstance(cfg, DepthImageNerfDataConfig):
                depth_images, uncertainty_images = get_depth_and_uncertainty_images(
                    loop_timer=loop_timer,
                    cfg=cfg,
                    grasp_frame_transforms=grasp_frame_transforms,
                    nerf_model=nerf_pipeline.model,
                )
                if depth_images.isnan().any():
                    print("\n" + "-" * 80)
                    print(
                        f"WARNING: Found {depth_images.isnan().sum()} depth image nans in {nerf_config}"
                    )
                    print("Skipping this one...")
                    print("-" * 80 + "\n")
                    continue

            if cfg.plot_only_one:
                break

            # Ensure no nans (most likely come from weird grasp transforms)
            if grasp_frame_transforms.isnan().any():
                print("\n" + "-" * 80)
                print(
                    f"WARNING: Found {grasp_frame_transforms.isnan().sum()} transform nans in {evaled_grasp_config_dict_filepath}"
                )
                print("Skipping this one...")
                print("-" * 80 + "\n")
                continue

            # Save values
            if not cfg.save_dataset:
                continue

            with loop_timer.add_section_timer("save values"):
                prev_idx = current_idx
                current_idx += grasp_configs.batch_size
                passed_eval_dataset[prev_idx:current_idx] = passed_evals
                passed_simulation_dataset[prev_idx:current_idx] = passed_simulations
                passed_penetration_threshold_dataset[
                    prev_idx:current_idx
                ] = passed_penetration_thresholds
                nerf_config_dataset[prev_idx:current_idx] = [str(nerf_config)] * (
                    current_idx - prev_idx
                )
                object_code_dataset[prev_idx:current_idx] = [object_code] * (
                    current_idx - prev_idx
                )
                object_scale_dataset[prev_idx:current_idx] = object_scale
                grasp_idx_dataset[prev_idx:current_idx] = np.arange(
                    prev_idx, current_idx
                )
                grasp_transforms_dataset[prev_idx:current_idx] = (
                    grasp_frame_transforms.matrix().cpu().detach().numpy()
                )

                if isinstance(cfg, GridNerfDataConfig):
                    nerf_densities_dataset[prev_idx:current_idx] = (
                        nerf_densities.detach().cpu().numpy()
                    )

                if isinstance(cfg, DepthImageNerfDataConfig):
                    depth_images_dataset[prev_idx:current_idx] = (
                        depth_images.detach().cpu().numpy()
                    )
                    uncertainty_images_dataset[prev_idx:current_idx] = (
                        uncertainty_images.detach().cpu().numpy()
                    )

                grasp_config_tensors = grasp_configs.as_tensor().detach().cpu().numpy()
                assert_equals(
                    grasp_config_tensors.shape,
                    (
                        grasp_configs.batch_size,
                        cfg.fingertip_config.n_fingers,
                        7
                        + 16
                        + 4,  # wrist pose, joint angles, grasp orientations (as quats)
                    ),
                )
                grasp_configs_dataset[prev_idx:current_idx] = grasp_config_tensors

                # May not be max_num_data_points if nan grasps
                hdf5_file.attrs["num_data_points"] = current_idx
            del nerf_pipeline
        except Exception as e:
            print("\n" + "-" * 80)
            print(f"WARNING: Failed to process {evaled_grasp_config_dict_filepath}")
            print(f"Exception: {e}")
            print("Skipping this one...")
            print("-" * 80 + "\n")
            continue

        if cfg.print_timing:
            loop_timer.pretty_print_section_times()
        print()

# %%
if not cfg.plot_only_one:
    print("Done!")
    sys.exit()

grasp_frame_transforms = (
    grasp_frame_transforms.matrix().cpu().detach().numpy()[cfg.grasp_visualize_index]
)
# Plot
delta = (
    cfg.fingertip_config.grasp_depth_mm / 1000 / (cfg.fingertip_config.num_pts_z - 1)
)

mesh_path = cfg.dexgraspnet_meshdata_root / object_code / "coacd" / "decomposed.obj"
assert mesh_path.exists(), f"mesh_path {mesh_path} does not exist"
mesh = trimesh.load(mesh_path, force="mesh")
mesh.apply_transform(trimesh.transformations.scale_matrix(object_scale))

if "nerf_densities" in globals():
    nerf_alphas = [
        1 - np.exp(-delta * dd) for dd in nerf_densities[cfg.grasp_visualize_index]
    ]
    fig = plot_mesh_and_query_points(
        mesh=mesh,
        query_points_list=[
            qq.reshape(-1, 3) for qq in query_points[cfg.grasp_visualize_index]
        ],
        query_points_colors_list=[x.reshape(-1) for x in nerf_alphas],
        num_fingers=cfg.fingertip_config.n_fingers,
        title=f"Mesh and Query Points, Success: {passed_evals[cfg.grasp_visualize_index]}",
    )
    fig.show()

fig2 = plot_mesh_and_transforms(
    mesh=mesh,
    transforms=[
        grasp_frame_transforms[i] for i in range(cfg.fingertip_config.n_fingers)
    ],
    num_fingers=cfg.fingertip_config.n_fingers,
    title=f"Mesh and Transforms, Success: {passed_evals[cfg.grasp_visualize_index]}",
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
        nerf_pipeline.model.field.get_density(ray_samples_in_mesh_region.to("cuda"))[0]
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

if cfg.plot_alphas_each_finger_1D and "nerf_densities" in globals():
    nerf_alphas = [
        1 - np.exp(-delta * dd) for dd in nerf_densities[cfg.grasp_visualize_index]
    ]
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

if cfg.plot_alpha_images_each_finger and "nerf_densities" in globals():
    nerf_alphas = [
        1 - np.exp(-delta * dd) for dd in nerf_densities[cfg.grasp_visualize_index]
    ]

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

if "depth_images" in globals():
    # plot depth and uncertainty side-by-side
    min_depth, max_depth = (
        torch.min(depth_images).item(),
        torch.max(depth_images).item(),
    )
    min_uncertainty, max_uncertainty = (
        torch.min(uncertainty_images).item(),
        torch.max(uncertainty_images).item(),
    )
    plt.figure(figsize=(20, 10))
    for finger_idx in range(cfg.fingertip_config.n_fingers):
        plot_idx = 2 * finger_idx + 1
        plt.subplot(cfg.fingertip_config.n_fingers, 2, plot_idx)
        plt.imshow(
            depth_images[cfg.grasp_visualize_index, finger_idx].detach().cpu(),
            vmin=min_depth,
            vmax=max_depth,
        )
        plt.title(f"Depth {finger_idx}")
        plt.colorbar()
        plt.subplot(cfg.fingertip_config.n_fingers, 2, plot_idx + 1)
        plt.imshow(
            uncertainty_images[cfg.grasp_visualize_index, finger_idx].detach().cpu(),
            vmin=min_uncertainty,
            vmax=max_uncertainty,
        )
        plt.title(f"Uncertainty {finger_idx}")
        plt.colorbar()
    plt.tight_layout()
    print(f"depth_images.min(): {depth_images.min()}")
    print(f"depth_images.max(): {depth_images.max()}")
    plt.show(block=True)

assert False, "cfg.plot_only_one is True"
