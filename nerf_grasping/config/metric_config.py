from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Literal, Union
from datetime import datetime
from nerf_grasping.config.base import WandbConfig
from nerf_grasping.config.fingertip_config import (
    UnionFingertipConfig,
    EvenlySpacedFingertipConfig,
)
from nerf_grasping.config.classifier_config import ClassifierConfig
import tyro
import pathlib

METRIC_DATETIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


@dataclass
class GraspMetricConfig:
    """Top-level config for grasp metric training."""

    classifier_config: ClassifierConfig
    classifier_config_path: Optional[pathlib.Path] = None

    random_seed: int = 42
    """Seed for RNG."""

    dry_run: bool = False
    """Flag to dry run dataset loading and classifier setup."""

    def __post_init__(self):
        """
        Load classifier config from file if classifier config is not None.
        """
        if self.classifier_config_path is not None:
            self.classifier_config = tyro.extras.from_yaml(
                ClassifierConfig, self.classifier_config_path
            )


if __name__ == "__main__":
    cfg = tyro.cli(GraspMetricConfig)
    print(cfg)
