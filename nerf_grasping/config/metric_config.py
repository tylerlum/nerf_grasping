from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Literal, Union
from nerf_grasping.config.base import WandbConfig, CONFIG_DATETIME_STR
from nerf_grasping.config.classifier_config import ClassifierConfig
from nerf_grasping.config.optimizer_config import (
    UnionOptimizerConfig,
    SGDOptimizerConfig,
)
import tyro
import pathlib
import nerf_grasping


@dataclass
class GraspMetricConfig:
    """Top-level config for grasp metric training."""

    classifier_config: ClassifierConfig
    classifier_config_path: Optional[pathlib.Path] = (
        pathlib.Path(nerf_grasping.get_repo_root())
        / "Train_DexGraspNet_NeRF_Grasp_Metric_workspaces"
        / "2023-08-30_02-14-46"
        / "config.yaml"
    )
    classifier_checkpoint: int = -1  # Load latest checkpoint if -1.
    optimizer: UnionOptimizerConfig = SGDOptimizerConfig()
    init_grasp_config_dicts_path: Optional[pathlib.Path] = (
        pathlib.Path(nerf_grasping.get_repo_root())
        / "data"
        / "2023-08-29_evaled_grasp_config_dicts_trial_big"
        / "mug_0_1000.npy"
    )
    output_path: Optional[pathlib.Path] = None
    grasp_split: Literal["train", "val", "test"] = "val"
    nerf_checkpoint_path: pathlib.Path = (
        pathlib.Path(nerf_grasping.get_repo_root())
        / "data"
        / "2023-08-29_nerfcheckpoints_trial"
        / "mug_0_1000"
        / "nerfacto"
        / "2023-08-30_014749"
        / "config.yml"
    )
    wandb: Optional[WandbConfig] = field(
        default_factory=lambda: WandbConfig(
            project="learned_metric", name=CONFIG_DATETIME_STR
        )
    )

    def __post_init__(self):
        """
        Load classifier config from file if classifier config is not None.
        """
        if self.classifier_config_path is not None:
            print(f"Loading classifier config from {self.classifier_config_path}")
            self.classifier_config = tyro.extras.from_yaml(
                ClassifierConfig, self.classifier_config_path.open()
            )
        else:
            print("Loading default classifier config.")
            self.classifier_config = ClassifierConfig()

        if self.output_path is None:
            print("Using default output path.")
            filename = self.init_grasp_config_dicts_path.name
            input_folder_path = self.init_grasp_config_dicts_path.parent
            output_folder_path = input_folder_path.parent / f"{input_folder_path.name}_optimized"
            self.output_path = output_folder_path / filename


if __name__ == "__main__":
    cfg = tyro.cli(GraspMetricConfig)
    print(cfg)
