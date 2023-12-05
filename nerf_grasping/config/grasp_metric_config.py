from dataclasses import dataclass
import tyro
import pathlib
import nerf_grasping
from nerf_grasping.config.classifier_config import (
    DEFAULTS_DICT as CLASSIFIER_DEFAULTS_DICT,
    ClassifierConfig,
)
from typing import Optional

@dataclass
class GraspMetricConfig:
    """Top-level config for creating a grasp metric."""

    classifier_config: ClassifierConfig = CLASSIFIER_DEFAULTS_DICT["simple-cnn-2d-1d"]
    classifier_config_path: Optional[pathlib.Path] = (
        pathlib.Path(nerf_grasping.get_repo_root())
        / "Train_DexGraspNet_NeRF_Grasp_Metric_workspaces"
        / "2023-08-30_02-14-46"
        / "config.yaml"
    )
    classifier_checkpoint: int = -1  # Load latest checkpoint if -1.
    nerf_checkpoint_path: pathlib.Path = (
        pathlib.Path(nerf_grasping.get_repo_root())
        / "data"
        / "2023-08-29_nerfcheckpoints_trial"
        / "mug_0_1000"
        / "nerfacto"
        / "2023-08-25_130206"
        / "config.yml"
    )

    def __post_init__(self):
        """
        Load classifier config from file if classifier config is not None.
        """
        if self.classifier_config_path is not None:
            print(f"Loading classifier config from {self.classifier_config_path}")
            self.classifier_config = tyro.extras.from_yaml(
                type(self.classifier_config), self.classifier_config_path.open()
            )
        else:
            print("Loading default classifier config.")

    @property
    def object_name(self) -> str:
        return self.nerf_checkpoint_path.parents[2].stem

if __name__ == "__main__":
    cfg = tyro.cli(GraspMetricConfig)
    print(cfg)
