from dataclasses import dataclass
import pathlib
import nerf_grasping


@dataclass
class GraspOptimizerConfig:
    num_steps: int = 35
    num_grasps: int = 64
    lr: float = 1e-4
    momentum: float = 0.9
    print_freq: int = 5
    save_grasps_freq: int = 5
