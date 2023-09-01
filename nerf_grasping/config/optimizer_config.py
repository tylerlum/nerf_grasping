from dataclasses import dataclass
import tyro


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
    num_init_samples: int = 2500
    num_samples: int = 1000
    num_elite: int = 100
    num_iters: int = 10


UnionGraspOptimizerConfig = tyro.extras.subcommand_type_from_defaults(
    {
        "cem": CEMOptimizerConfig(),
        "sgd": SGDOptimizerConfig(),
    }
)
