from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Literal, Union
from datetime import datetime
from nerf_grasping.config.base import WandbConfig
from nerf_grasping.config.classifier_config import ClassifierConfig
from nerf_grasping.config.optimizer_config import GraspOptimizerConfig
import tyro
import pathlib
import nerf_grasping
import yaml

METRIC_DATETIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


@dataclass
class GraspMetricConfig:
    """Top-level config for grasp metric training."""

    classifier_config: ClassifierConfig
    classifier_config_path: Optional[pathlib.Path] = (
        pathlib.Path(nerf_grasping.get_repo_root())
        / "Train_DexGraspNet_NeRF_Grasp_Metric_workspaces"
        / "2023-08-28_14-24-03"
        / "config.yaml"
    )
    classifier_checkpoint: int = -1  # Load latest checkpoint if -1.
    optimizer: GraspOptimizerConfig = GraspOptimizerConfig()
    init_grasps_path: Optional[pathlib.Path] = (
        pathlib.Path(nerf_grasping.get_repo_root())
        / "data"
        / "2023-08-26_overfit_evaled_grasp_config_dicts"
        / "mug_0_10.npy"
    )
    output_path: Optional[pathlib.Path] = None
    grasp_split: Literal["train", "val", "test"] = "val"
    nerf_checkpoint_path: pathlib.Path = (
        pathlib.Path(nerf_grasping.get_repo_root())
        / "data"
        / "nerfcheckpoints_trial"
        / "mug_0_10"
        / "nerfacto"
        / "2023-08-25_130206"
        / "config.yml"
    )
    wandb: Optional[WandbConfig] = field(
        default_factory=lambda: WandbConfig(
            project="learned_metric", name=METRIC_DATETIME_STR
        )
    )

    def __post_init__(self):
        """
        Load classifier config from file if classifier config is not None.
        """
        if self.classifier_config_path is not None:
            self.classifier_config = tyro.extras.from_yaml(
                ClassifierConfig, self.classifier_config_path.open()
            )
        else:
            self.classifier_config = ClassifierConfig()

        if self.output_path is None:
            self.output_path = pathlib.Path(
                str(self.init_grasps_path).replace(".npy", "_optimized.npy")
            )


if __name__ == "__main__":
    cfg = tyro.cli(GraspMetricConfig)
    print(cfg)
