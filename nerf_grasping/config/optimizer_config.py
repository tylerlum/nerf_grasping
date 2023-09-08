from dataclasses import dataclass
import tyro


@dataclass
class BaseOptimizerConfig:
    print_freq: int = 5
    save_grasps_freq: int = 5


@dataclass
class SGDOptimizerConfig(BaseOptimizerConfig):
    num_steps: int = 35
    num_grasps: int = 256
    finger_lr: float = 5e-2
    grasp_dir_lr: float = 5e-2
    wrist_lr: float = 5e-4
    momentum: float = 0.9
    opt_wrist_pose: bool = True
    opt_grasp_dirs: bool = True


@dataclass
class CEMOptimizerConfig(BaseOptimizerConfig):
    num_init_samples: int = 250
    num_samples: int = 250
    num_elite: int = 50
    num_steps: int = 30
    min_cov_std: float = 1e-2


UnionGraspOptimizerConfig = tyro.extras.subcommand_type_from_defaults(
    {
        "cem": CEMOptimizerConfig(),
        "sgd": SGDOptimizerConfig(),
    }
)
