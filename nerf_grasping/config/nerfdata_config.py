from dataclasses import dataclass
from nerf_grasping.config.fingertip_config import (
    UnionFingertipConfig,
    EvenlySpacedFingertipConfig,
)
import pathlib
from typing import Optional
from datetime import datetime
import tyro
import nerf_grasping
from nerf_grasping.config.base import CONFIG_DATETIME_STR


@dataclass
class NerfDataConfig:
    """Top-level config for NeRF data generation."""

    fingertip_config: UnionFingertipConfig = (
        EvenlySpacedFingertipConfig.from_dimensions()
    )
    dexgraspnet_data_root: pathlib.Path = (
        pathlib.Path(nerf_grasping.get_repo_root()) / "data"
    )
    dexgraspnet_meshdata_root: pathlib.Path = (
        dexgraspnet_data_root / "2023-08-29_meshdata_trial"
    )
    evaled_grasp_config_dicts_path: pathlib.Path = (
        dexgraspnet_data_root / "2023-08-29_evaled_grasp_config_dicts_trial_big"
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
    plot_alphas_each_finger_1D: bool = True
    plot_alpha_images_each_finger: bool = True

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


if __name__ == "__main__":
    cfg = tyro.cli(NerfDataConfig)
    print(cfg)
