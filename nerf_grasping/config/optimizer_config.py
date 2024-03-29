from dataclasses import dataclass
import tyro


@dataclass
class BaseOptimizerConfig:
    num_grasps: int = 20
    num_steps: int = 30


@dataclass
class SGDOptimizerConfig(BaseOptimizerConfig):
    num_steps: int = 200
    finger_lr: float = 1e-4
    grasp_dir_lr: float = 1e-4
    wrist_lr: float = 1e-4
    momentum: float = 0.9
    opt_wrist_pose: bool = True
    opt_grasp_dirs: bool = True


@dataclass
class CEMOptimizerConfig(BaseOptimizerConfig):
    num_steps: int = 30
    num_samples: int = 5
    num_elite: int = 2
    min_cov_std: float = 1e-2


UnionGraspOptimizerConfig = tyro.extras.subcommand_type_from_defaults(
    {
        "cem": CEMOptimizerConfig(),
        "sgd": SGDOptimizerConfig(),
    }
)
