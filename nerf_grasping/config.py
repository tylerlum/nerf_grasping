import dataclasses
import dcargs
import enum
import pickle

from typing import Optional, Union, Tuple


class GradType(enum.Enum):
    GAUSSIAN = enum.auto()
    AVERAGE = enum.auto()
    CENTRAL_DIFFERENCE = enum.auto()


class CostType(enum.Enum):
    L1 = enum.auto()
    PSV = enum.auto()
    MSV = enum.auto()
    FC = enum.auto()


class ObjectType(enum.Enum):
    BANANA = enum.auto()
    BOX = enum.auto()
    TEDDY_BEAR = enum.auto()
    POWER_DRILL = enum.auto()
    MUG = enum.auto()
    BLEACH_CLEANSER = enum.auto()


@dataclasses.dataclass(frozen=True)
class GradEst:
    # Style of gradient estimation to use.
    method: GradType = GradType.GAUSSIAN

    # Variance to use for gradient estimation.
    variance: float = 7.5e-3

    # Number of samples to use for gradient estimation.
    num_samples: int = 250


grad_configs = {
    "grasp_opt": GradEst(),
    "sim": GradEst(variance=5e-3, num_samples=1000),
}


@dataclasses.dataclass(frozen=True)
class ControllerParams:
    """Grasp and PD Object Position Controller parameters"""

    # Proportional gain during reaching
    kp_reach: float = 0.5

    # Derivative gain during reaching
    kd_reach: float = 0.003

    # Proportional gain during grasping
    kp_grasp: float = 5.0

    # Derivative gain during grasping
    kd_grasp: float = 1.0  # 10. # 0.25

    # Grasp normal vector scaling applied during grasping
    normal_scale_grasp: float = 0.05  # 1e-4  # 0.1

    # Grasp target normal force to apply with fingers
    target_normal: float = 3.0  # 1.0  # 0.5

    # Proportional position gain during lifting
    kp_lift: float = 10.0  # 10.0

    # Derivative positi.on gain during lifting
    kd_lift: float = 1.0  # 0.1

    # Proportional rotation gain during lifting
    kp_rot_lift: float = 0.2  # 0.04

    # Derivative rotation gain during lifting
    kd_rot_lift: float = 5e-3  # 0.001


@dataclasses.dataclass(frozen=True)
class RobotConfig:
    """Params to initialize FingertipRobot with controller config"""

    # Number of fingers robot has.
    num_fingers: int = 3

    # Target height to lift to.
    target_height: float = 0.07

    # Use groundtruth mesh normals for contact surface normals.
    gt_normals: bool = False

    # Offset from object surface to start initial grasp trajectory from.
    des_z_dist: float = 0.01

    # Fingertip friction coefficient.
    mu: float = 1.0

    # Fingertip sphere radius (also used when computing approximate contact point).
    sphere_radius: float = 0.01

    # PD controller parameters.
    controller_params: ControllerParams = ControllerParams()

    # Gradient estimation parameters.
    grad_config: GradEst = grad_configs["sim"]

    # use for debugging Robot controller
    verbose: bool = False


@dataclasses.dataclass(frozen=True)
class Nerf:
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

    # Commented due to possible bug in dcargs.
    # Config object for gradient estimation.
    grad_config: GradEst = grad_configs["sim"]

    # Flag to use expected surface point.
    expected_surface: bool = False

    # Flag to use gradient at expected surface.
    expected_gradient: bool = False

    # Desired z-distance for fingers.
    des_z_dist: float = 0.05

    # Number of iterations for z_dist correction.
    num_z_dist_iters: int = 10


@dataclasses.dataclass(frozen=True)
class Mesh:

    # What level set to extract with marching cubes; if None, uses gt mesh.
    level_set: Optional[float] = None

    # How far fingers should be positioned from surface.
    des_z_dist: float = 0.025


@dataclasses.dataclass(frozen=True)
class Experiment:

    # Configuration for object model; dispatch on Nerf vs. mesh.
    model_config: Union[Nerf, Mesh] = Nerf()

    # Which object is used in experiment.
    object: ObjectType = ObjectType.BANANA

    # Configuration for robot.
    robot_config: RobotConfig = RobotConfig()

    # Number of grasps to generate / test.
    num_grasps: int = 50

    # Which cost function to optimize.
    cost_function: CostType = CostType.L1

    # How many samples to use for CEM.
    cem_num_samples: int = 500

    # Number of iterations to run CEM.
    cem_num_iters: int = 15

    # Elite fraction for CEM.
    cem_elite_frac: float = 0.1

    # Risk sensitivity value to use in cost.
    risk_sensitivity: Optional[float] = None

    # Flag to use "dicing the grasp" to optimize grasp.
    dice_grasp: bool = False

    # Rejection parameter for "dicing the grasp."
    dice_mu: float = 0.5

    # Enable visualization to see grasping policy
    visualize: bool = False

    # Number of grasp samples to draw to compute expectations.
    num_grasp_samples: int = 10


@dataclasses.dataclass(frozen=True)
class EvalExperiment(Experiment):
    # Optional, containing either a single index or tuple of (start, end) indices
    grasp_idx: Union[None, int, Tuple[int, int]] = None
    grasp_data: Optional[str] = None


def mesh_file(exp_config: Experiment):
    """Gets mesh filename from experiment config."""
    obj_name = exp_config.object.name.lower()

    if exp_config.model_config.level_set:
        return (
            f"grasp_data/meshes/{obj_name}_{int(exp_config.model_config.level_set)}.obj"
        )
    else:
        return f"grasp_data/meshes/{obj_name}.obj"


def grasp_file(exp_config: Experiment):
    """Generates grasp data filenames from experiment config."""

    outfile = f"grasp_data/{exp_config.object.name.lower()}"

    if isinstance(exp_config.model_config, Nerf):
        outfile += "_nerf"
        outfile += f"_{exp_config.cost_function.name.lower()}"
        if exp_config.risk_sensitivity:
            outfile += f"_rs{exp_config.risk_sensitivity}"

    else:
        if exp_config.model_config.level_set:
            outfile += f"_{int(exp_config.model_config.level_set)}"
        if exp_config.dice_grasp:
            outfile += "_diced"

    return outfile


def save(exp_config: Experiment):
    outfile = grasp_file(exp_config)

    with open(f"{outfile}.pkl", "wb") as f:
        pickle.dump(exp_config, f)

    # Deprecated due to bug in dcargs
    # with open(f"{outfile}.yaml", "w") as file:
    #     file.write(dcargs.extras.to_yaml(exp_config))


def load(infile):
    with open(infile, "rb") as f:
        return pickle.load(f)

    # Deprecated due to bug in dcargs.
    # with open(infile, "r") as file:
    #     return dcargs.extras.from_yaml(Experiment, file)
