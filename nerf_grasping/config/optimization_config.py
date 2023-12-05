from dataclasses import dataclass, field
import tyro
import pathlib
import nerf_grasping
from nerf_grasping.config.optimizer_config import (
    UnionGraspOptimizerConfig,
)
from nerf_grasping.config.grasp_metric_config import (
    GraspMetricConfig,
)
from typing import Optional, Literal
from nerf_grasping.config.base import WandbConfig, CONFIG_DATETIME_STR


@dataclass
class OptimizationConfig:
    """Top-level config for optimizing grasp metric."""
    optimizer: UnionGraspOptimizerConfig
    grasp_metric: GraspMetricConfig
    init_grasp_config_dict_path: pathlib.Path = (
        pathlib.Path(nerf_grasping.get_repo_root())
        / "data"
        / "2023-11-17_rubikscube_0"
        / "evaled_grasp_config_dicts"
        / "ddg-gd_rubik_cube_poisson_004_0_1000.npy"
    )
    output_path: Optional[pathlib.Path] = None
    grasp_split: Literal["train", "val", "test"] = "val"
    wandb: Optional[WandbConfig] = field(
        default_factory=lambda: WandbConfig(
            project="learned_metric", name=CONFIG_DATETIME_STR
        )
    )
    use_rich: bool = False
    """Whether to use rich for logging (rich is nice but makes breakpoint() not work)."""

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
            self.output_path = output_folder_path / filename


if __name__ == "__main__":
    cfg = tyro.cli(OptimizationConfig)
    print(cfg)
