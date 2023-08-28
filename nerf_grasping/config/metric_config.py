from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Literal, Union
from datetime import datetime
from nerf_grasping.config.base import WandbConfig
from nerf_grasping.config.classifier_config import ClassifierConfig
from nerf_grasping.config.optimizer_config import GraspOptimizerConfig
import tyro
import pathlib

METRIC_DATETIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


@dataclass
class GraspMetricConfig:
    """Top-level config for grasp metric training."""

    classifier_config: ClassifierConfig
    classifier_config_path: Optional[pathlib.Path] = None
    classifier_checkpoint: int = -1  # Load latest checkpoint if -1.
    optimizer_config: GraspOptimizerConfig = GraspOptimizerConfig()

    def __post_init__(self):
        """
        Load classifier config from file if classifier config is not None.
        """
        if self.classifier_config_path is not None:
            self.classifier_config = tyro.extras.from_yaml(
                ClassifierConfig, self.classifier_config_path
            )
        else:
            self.classifier_config = ClassifierConfig()


if __name__ == "__main__":
    cfg = tyro.cli(GraspMetricConfig)
    print(cfg)
