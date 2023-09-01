from dataclasses import dataclass
import pathlib
import nerf_grasping
from typing import Union


@dataclass
class SGDOptimizerConfig:
    num_steps: int = 35
    num_grasps: int = 64
    lr: float = 1e-4
    momentum: float = 0.9
    print_freq: int = 5
    save_grasps_freq: int = 5


@dataclass
class CEMOptimizerConfig:
    num_samples: int = 1000
    elite_frac: float = 0.1
    num_iters: int = 10


UnionOptimizerConfig = Union[SGDOptimizerConfig, CEMOptimizerConfig]
