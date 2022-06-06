import numpy as np
import kinematics_utils
import pinocchio as pin
import yaml

def load_pos_control_config(config_file):
    with open(config_file) as file:
        return yaml.safe_load(file)

def fingertip_pos_control(p_des, model, data, q, dq, config):
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
    p_curr, v_curr = kinematics_utils.get_fingertip_pos_vel(model, data, q, dq)

    J = get_fingertip_jacobian(model, data, q)

    a_des = config['Kp'] * (p_des - p_curr) - config['Kd'] * v_curr
    ddq_des = np.linalg.inv(J.T @ J + config['damping'] * np.eye(9)) @ J.T @ a_des

    ddq0 = np.zeros_like(dq)
    b = pin.rnea(model, data, q, dq, ddq0)
    M = pin.crba(model, data, q)

    return M @ a_des + b