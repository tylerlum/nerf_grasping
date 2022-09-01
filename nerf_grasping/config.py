import dataclasses
import enum

from typing import Optional, Union


class GradType(enum.Enum):
    GAUSSIAN = enum.auto()
    AVERAGE = enum.auto()
    CENTRAL_DIFFERENCE = enum.auto()


class CostType(enum.Enum):
    L1 = enum.auto()
    PSV = enum.auto()
    MSV = enum.auto()


class ObjectType(enum.Enum):
    BANANA = enum.auto()
    BOX = enum.auto()
    TEDDY_BEAR = enum.auto()
    POWER_DRILL = enum.auto()
    MUG = enum.auto()
    BLEACH_CLEANSER = enum.auto()


@dataclasses.dataclass
class GradEst:
    # Style of gradient estimation to use.
    method: GradType = GradType.GAUSSIAN

    # Variance to use for gradient estimation.
    variance: float = 7.5e-3

    # Number of samples to use for gradient estimation.
    num_samples: int = 250


@dataclasses.dataclass
class ControllerParams:
    """Grasp and PD Object Position Controller parameters"""

    # Grasp target normal force to apply with fingers
    target_normal = 0.5

    # Proportional position gain
    kp = 10.0

    # Derivative position gain
    kd = 0.1

    # Proportional rotation gain
    kp_angle = 0.04

    # Derivative rotation gain
    kd_angle = 0.001


@dataclasses.dataclass
class RobotConfig:
    """Params to initialize FingertipRobot with controller config"""

    # Target height to lift to
    target_height: float = 0.7

    # use groundtruth mesh normals for contact surface normals
    gt_normals: bool = False

    # offset from object surface to start initial grasp trajectory from
    des_z_dist: float = 0.1

    # fingertip friction coefficient
    mu: float = 1.0

    # fingertip sphere radius (also used when computing approximate contact point)
    sphere_radius: float = 0.01

    # PD controller parameters
    controller_params: ControllerParams = ControllerParams()

    # use for debugging Robot controller
    verbose: bool = False


grad_configs = {
    "grasp_opt": GradEst(),
    "sim": GradEst(variance=5e-3, num_samples=1000),
}


@dataclasses.dataclass
class NeRF:
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
    grad_config: GradEst = grad_configs["grasp_opt"]

    # Desired z-distance for fingers.
    des_z_dist: float = 0.025

    # Number of iterations for z_dist correction.
    num_z_dist_iters: int = 10


@dataclasses.dataclass
class Mesh:

    # What level set to extract with marching cubes; if None, uses gt mesh.
    level_set: Optional[int] = None


@dataclasses.dataclass
class ExperimentConfig:

    # Which object is used in experiment.
    object: ObjectType = ObjectType.BANANA

    # Configuration for object model; dispatch on NeRF vs. mesh.
    model_config: Union[NeRF, Mesh] = NeRF()

    # Number of grasps to generate / test.
    num_grasps: int = 50

    # Which cost function to optimize.
    cost_function: CostType = CostType.L1

    # Flag to use ground truth normals for lifting.
    gt_normals: bool = False

    # Risk sensitivity value to use in cost.
    risk_sensitivity: Optional[float] = None

    # Flag to use "dicing the grasp" to optimize grasp.
    dice_grasp: bool = False


def mesh_file(exp_config: ExperimentConfig):
    obj_name = exp_config.object.name.lower()

    if exp_config.model_config.level_set:
        return f"grasp_data/meshes/{obj_name}_{exp_config.model_config.level_set}.obj"
    else:
        return f"grasp_data/meshes/{obj_name}.obj"


def grasp_file(exp_config: ExperimentConfig):
    outfile = f"grasp_data/{exp_config.object.name.lower()}"

    if isinstance(exp_config.model_config, NeRF):
        outfile += "_nerf"
        outfile += f"_{exp_config.cost_function.name.lower()}"
        if exp_config.model_config.risk_sensitivity:
            outfile += f"_rs{exp_config.risk_sensitivity}"

    else:
        if exp_config.model_config.level_set:
            outfile += f"_{exp_config.model_config.level_set}"
        if exp_config.dice_grasp:
            outfile += "_diced"

    outfile += ".npy"

    return outfile
