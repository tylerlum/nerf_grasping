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

    dexgraspnet_data_root: pathlib.Path = pathlib.Path(nerf_grasping.get_repo_root())
    dexgraspnet_meshdata_root: pathlib.Path = (
        dexgraspnet_data_root / "2023-08-29_meshdata_trial"
    )
    evaled_grasp_config_dicts_path: pathlib.Path = (
        dexgraspnet_data_root / "2023-08-29_evaled_grasp_config_dicts_trial"
    )
    nerf_checkpoints_path: pathlib.Path = (
        dexgraspnet_data_root / "2023-08-29_nerfcheckpoints_trial"
    )
    output_filepath: Optional[pathlib.Path] = None
    plot_only_one: bool = False
    save_dataset: bool = True
    print_timing: bool = True
    limit_num_configs: Optional[int] = None  # None for no limit
    max_num_data_points_per_file: int = 2500
    plot_all_high_density_points: bool = True
    config_filepath: Optional[pathlib.Path] = None

    fingertip_config: UnionFingertipConfig = (
        EvenlySpacedFingertipConfig.from_dimensions()
    )

    def __post_init__(self):
        if self.output_filepath is None:
            self.output_filepath = (
                pathlib.Path(
                    str(self.evaled_grasp_config_dicts_path) + "_learned_metric_dataset"
                )
                / f"{CONFIG_DATETIME_STR}_learned_metric_dataset.h5"
            )
        if self.config_filepath is None:
            self.config_filepath = self.output_filepath.parent / "config.yml"


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


UnionNerfDataConfig = tyro.extras.subcommand_type_from_defaults(
    {
        "grid": GridNerfDataConfig(),
        "depth-image": DepthImageNerfDataConfig(
            fingertip_config=EvenlySpacedFingertipConfig.from_dimensions(
                finger_width_mm=20.0, finger_height_mm=20.0
            ),
            fingertip_camera_config=CameraConfig(H=40, W=40),
        ),
    }
)

if __name__ == "__main__":
    cfg = tyro.cli(UnionNerfDataConfig)
    print(cfg)
