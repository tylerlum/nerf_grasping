from dataclasses import dataclass, field
import tyro
import pathlib
import nerf_grasping
from nerf_grasping.config.optimizer_config import (
    SGDOptimizerConfig,
    CEMOptimizerConfig,
)
from nerf_grasping.config.grasp_metric_config import (
    GraspMetricConfig,
)
from typing import Optional, Literal, Union
from nerf_grasping.config.base import WandbConfig

DEFAULT_WANDB_PROJECT = "optimize_metric"

@dataclass
class OptimizationConfig:
    """Top-level config for optimizing grasp metric."""
    optimizer: Union[SGDOptimizerConfig, CEMOptimizerConfig]
    grasp_metric: GraspMetricConfig = field(default_factory=GraspMetricConfig)
    init_grasp_config_dict_path: pathlib.Path = (
        pathlib.Path(nerf_grasping.get_repo_root())
        / "data"
        / "2023-01-03_mugs_smaller0-075_noise_lightshake_mid_opt"
        / "evaled_grasp_config_dicts"
        / "core-mug-10f6e09036350e92b3f21f1137c3c347_0_0750.npy"
    )
    output_path: Optional[pathlib.Path] = None
    wandb: Optional[WandbConfig] = field(
        default_factory=lambda: WandbConfig(
            project=DEFAULT_WANDB_PROJECT
        )
    )
    use_rich: bool = False
    """Whether to use rich for logging (rich is nice but makes breakpoint() not work)."""
    print_freq: int = 5
    save_grasps_freq: int = 5

    def __post_init__(self):
        """
        Set default output path if not specified.
        """
        if self.output_path is None:
            print("Using default output path.")
            filename = self.grasp_metric.object_name
            input_folder_path = self.init_grasp_config_dict_path.parent
            output_folder_path = (
                input_folder_path.parent / f"{input_folder_path.name}_optimized"
            )
            self.output_path = output_folder_path / f"{filename}.npy"


if __name__ == "__main__":
    cfg = tyro.cli(OptimizationConfig)
    print(cfg)
