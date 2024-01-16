from dataclasses import dataclass
import tyro


@dataclass
class BaseOptimizerConfig:
    pass


@dataclass
class SGDOptimizerConfig(BaseOptimizerConfig):
    num_steps: int = 1000
    num_grasps: int = 10
    finger_lr: float = 5e-5
    grasp_dir_lr: float = 5e-5
    wrist_lr: float = 5e-5
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
