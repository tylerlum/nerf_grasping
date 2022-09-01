import dataclasses
import enum


class GradType(enum.Enum):
    GAUSSIAN = enum.auto()
    AVERAGE = enum.auto()
    CENTRAL_DIFFERENCE = enum.auto()


@dataclasses.dataclass
class GradConfig:
    # Style of gradient estimation to use.
    method: GradType = GradType.GAUSSIAN

    # Variance to use for gradient estimation.
    variance: float = 7.5e-3

    # Number of samples to use for gradient estimation.
    num_samples: int = 250


grad_configs = {
    "grasp_opt": GradConfig(),
    "sim": GradConfig(variance=5e-3, num_samples=1000),
}


@dataclasses.dataclass
class NeRFConfig:
    # Number of initial steps for finger rendering.
    num_steps: int = 128

    # Number of importance samples to add for finger rendering.
    upsample_steps: int = 256

    # Near bound for finger rendering.
    render_near_bound: float = 0.0001

    # Far bound for finger rendering.
    render_far_bound: float = 0.15

    # Flag to add noise to samples during rendering.
    render_perturb_samples: bool = True

    # Config object for gradient estimation.
    grad_config: GradConfig = grad_configs["grasp_opt"]

    # Desired z-distance for fingers.
    des_z_dist: float = 0.025

    # Number of iterations for z_dist correction.
    num_z_dist_iters: int = 10
