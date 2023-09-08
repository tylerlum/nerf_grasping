from dataclasses import dataclass, field
from nerf_grasping.config.fingertip_config import (
    UnionFingertipConfig,
    EvenlySpacedFingertipConfig,
)
from nerf_grasping.config.camera_config import CameraConfig
import pathlib
from typing import Optional, Union
from datetime import datetime
import tyro
import nerf_grasping
from nerf_grasping.config.base import CONFIG_DATETIME_STR
from enum import Enum, auto


@dataclass
class BaseNerfDataConfig:
    """Top-level config for NeRF data generation."""

    dexgraspnet_data_root: pathlib.Path = (
        pathlib.Path(nerf_grasping.get_repo_root()) / "data"
    )
    dexgraspnet_meshdata_root: pathlib.Path = (
        dexgraspnet_data_root / "2023-09-01_meshdata_trial"
    )
    evaled_grasp_config_dicts_path: pathlib.Path = None
    nerf_checkpoints_path: pathlib.Path = (
        dexgraspnet_data_root / "2023-09-04_nerfcheckpoints_trial"
    )
    output_filepath: Optional[pathlib.Path] = None
    plot_only_one: bool = False
    nerf_visualize_index: Optional[int] = 0
    grasp_visualize_index: Optional[int] = 0
    save_dataset: bool = True
    print_timing: bool = True
    limit_num_configs: Optional[int] = None  # None for no limit
    max_num_data_points_per_file: Optional[
        int
    ] = None  # None for count actual num data points
    plot_all_high_density_points: bool = True
    plot_alphas_each_finger_1D: bool = True
    plot_alpha_images_each_finger: bool = True

    fingertip_config: Optional[UnionFingertipConfig] = EvenlySpacedFingertipConfig()

    def __post_init__(self):
        if self.output_filepath is None:
            self.output_filepath = (
                pathlib.Path(
                    str(self.evaled_grasp_config_dicts_path) + "_learned_metric_dataset"
                )
                / f"{CONFIG_DATETIME_STR}_learned_metric_dataset.h5"
            )

    @property
    def config_filepath(self) -> pathlib.Path:
        return self.output_filepath.parent / "config.yml"


@dataclass
class GridNerfDataConfig(BaseNerfDataConfig):
    plot_alphas_each_finger_1D: bool = True
    plot_alpha_images_each_finger: bool = True


@dataclass
class DepthImageNerfDataConfig(BaseNerfDataConfig):
    fingertip_camera_config: CameraConfig = field(default_factory=CameraConfig)

    def __post_init__(self):
        self.fingertip_camera_config.set_intrisics_from_fingertip_config(
            self.fingertip_config
        )


# Variant of GridNerfDataConfig that uses grasp-conditioned data
class GraspConditionedGridDataConfig(GridNerfDataConfig):
    pass


UnionNerfDataConfig = tyro.extras.subcommand_type_from_defaults(
    {
        "grid": GridNerfDataConfig(),
        "depth-image": DepthImageNerfDataConfig(
            fingertip_config=EvenlySpacedFingertipConfig(
                finger_width_mm=20.0, finger_height_mm=20.0
            ),
            fingertip_camera_config=CameraConfig(H=40, W=40),
        ),
        "grasp-conditioned-grid": GraspConditionedGridDataConfig(),
    }
)

if __name__ == "__main__":
    cfg = tyro.cli(UnionNerfDataConfig)
    print(cfg)
