import numpy as np
import pinocchio as pin
import os
import yaml
from typing import Union, List

from dataclasses import dataclass
from nerf_grasping import kinematics_utils

# Note: this is an ugly hack to avoid installing the config files -- we can figure out something better eventually.
BASEDIR = os.path.dirname(__file__)


@dataclass
class PosControlConfig:
    """Helper class to store controller params."""

    Kp: Union[float, List[float]]
    Kd: Union[float, List[float]]
    damping: float


def load_config(config_file=os.path.join(BASEDIR, "pos_control.yaml")):
    """Util to load in config parameters from yaml + build config object."""
    with open(config_file) as file:
        config = yaml.safe_load(file)

    # Quick hack to make sure YAML params are loaded in as floats + not strings.
    conf = {}
    for k, v in config.items():
        if isinstance(v, float):
            pass
        elif "," in v:
            v = [float(x) for x in v.split(",")]
        else:
            v = float(v)
        conf[k] = v

    return PosControlConfig(**conf)


def get_joint_torques(p_des, model, data, q, dq, config):
    """Implements an IK controller to move fingertips to desired position.

    Args:
        p_des: Set of desired fingertip positions, [3, 3].
        model: Pinocchio model describing trifinger.
        data: Pinnochio model data.
        q: Current joint angles of robot.
        dq: Current joint velocities of robot.
        config_file: path to config file with position control params.

    Returns joint torques for each finger joint.
    """
    # Compute current fingertip positions, velocities, and Jacobians.
    p_curr, v_curr = kinematics_utils.get_fingertip_pos_vel(model, data, q, dq)
    J = kinematics_utils.get_fingertip_jacobian(model, data, q)

    # Implement PD controller in task space, and map back to joint space.
    Kp, Kd = config.Kp, config.Kd
    if isinstance(config.Kp, list):
        Kp = np.array(Kp)
    if isinstance(config.Kp, list):
        Kd = np.array(Kd)
    a_des = config.Kp * (p_des - p_curr) - config.Kd * v_curr

    # print(np.linalg.norm(p_des - p_curr))
    ddq_des = np.linalg.inv(J.T @ J + config.damping * np.eye(9)) @ J.T @ a_des

    # Compute mass matrix + feedforward term for control.
    ddq0 = np.zeros_like(dq)
    b = pin.rnea(model, data, q, dq, ddq0)
    M = pin.crba(model, data, q)

    return M @ ddq_des + b
