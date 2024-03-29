from dataclasses import dataclass
import tyro
import pathlib
import nerf_grasping
from nerf_grasping.config.classifier_config import (
    DEFAULTS_DICT as CLASSIFIER_DEFAULTS_DICT,
    ClassifierConfig,
)
from typing import Optional
import numpy as np


@dataclass
class GraspMetricConfig:
    """Top-level config for creating a grasp metric."""

    classifier_config: ClassifierConfig = CLASSIFIER_DEFAULTS_DICT["grasp-cond-simple-cnn-2d-1d"]
    classifier_config_path: Optional[pathlib.Path] = (
        pathlib.Path(nerf_grasping.get_repo_root())
        / "Train_DexGraspNet_NeRF_Grasp_Metric_workspaces"
        / "mugs_grid_grasp-cond-simple-cnn-2d-1d"
        / "config.yaml"
    )
    classifier_checkpoint: int = -1  # Load latest checkpoint if -1.
    nerf_checkpoint_path: pathlib.Path = (
        pathlib.Path(nerf_grasping.get_repo_root())
        / "data"
        / "2023-01-03_mugs_smaller0-075_noise_lightshake_mid_opt"
        / "nerfcheckpoints"
        / "core-mug-10f6e09036350e92b3f21f1137c3c347_0_0750"
        / "nerfacto"
        / "2024-01-03_235839"
        / "config.yml"
    )
    object_transform_world_frame: Optional[np.ndarray] = None

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

        if self.object_transform_world_frame is None:
            self.object_transform_world_frame = np.eye(4)

    @property
    def object_name(self) -> str:
        return self.nerf_checkpoint_path.parents[2].stem


if __name__ == "__main__":
    cfg = tyro.cli(GraspMetricConfig)
    print(cfg)
