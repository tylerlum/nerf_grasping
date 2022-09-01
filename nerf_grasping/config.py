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


grad_configs = {
    "grasp_opt": GradConfig(),
    "sim": GradConfig(variance=5e-3, num_samples=1000),
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
    level_set: Optional[float] = None


@dataclasses.dataclass
class Experiment:

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


def mesh_file(exp_config: Experiment):
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
            outfile += f"_rs{exp_config.model_config.risk_sensitivity}"

    else:
        if exp_config.model_config.level_set:
            outfile += f"_{exp_config.model_config.level_set}"
        if exp_config.dice_grasp:
            outfile += "_diced"

    outfile += ".npy"

    return outfile
