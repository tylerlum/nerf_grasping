import pathlib
from typing import Optional, Tuple

import numpy as np
import torch
import transforms3d
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from nerf_grasping.curobo_fr3_algr_zed2i.fr3_algr_zed2i_world import (
    get_dummy_collision_dict,
    get_object_collision_dict,
    get_table_collision_dict,
)

# Third Party
import torch

# cuRobo
from curobo.cuda_robot_model.cuda_robot_model import (
    CudaRobotModel,
    CudaRobotModelConfig,
)
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_path, join_path, load_yaml


def quat_wxyz_to_matrix(quat_wxyzs: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quat_wxyzs to rotation matrices.
    Args:
        quat_wxyzs: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quat_wxyzs, -1)
    two_s = 2.0 / (quat_wxyzs * quat_wxyzs).sum(-1)

    mat = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return mat.reshape(quat_wxyzs.shape[:-1] + (3, 3))


def solve_fk(
    q_fr3: np.ndarray,
    q_algr: np.ndarray,
) -> np.ndarray:
    assert q_fr3.shape == (7,)
    assert q_algr.shape == (16,)

    X_W_H = solve_fks(
        q_fr3s=q_fr3[None, ...],
        q_algrs=q_algr[None, ...],
    ).squeeze(axis=0)
    assert X_W_H.shape == (4, 4)

    return X_W_H


def solve_fks(
    q_fr3s: np.ndarray,
    q_algrs: np.ndarray,
) -> np.ndarray:
    N = q_fr3s.shape[0]
    assert q_fr3s.shape == (
        N,
        7,
    )
    assert q_algrs.shape == (
        N,
        16,
    )

    robot_file = "fr3_algr_zed2i.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )
    kin_model = CudaRobotModel(robot_cfg.kinematics)

    q = torch.from_numpy(np.concatenate([q_fr3s, q_algrs], axis=1)).float().cuda()
    assert q.shape == (N, 23)

    state = kin_model.get_state(q)
    trans = state.ee_position.detach().cpu().numpy()
    rot_matrix = quat_wxyz_to_matrix(state.ee_quaternion).detach().cpu().numpy()

    X_W_H = np.eye(4).reshape(1, 4, 4).repeat(N, axis=0)
    X_W_H[:, :3, :3] = rot_matrix
    X_W_H[:, :3, 3] = trans
    return X_W_H


def main() -> None:
    DEFAULT_Q_FR3 = np.array([0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854])
    DEFAULT_Q_ALGR = np.array(
        [
            2.90945620e-01,
            7.37109400e-01,
            5.10859200e-01,
            1.22637060e-01,
            1.20125350e-01,
            5.84513500e-01,
            3.43829930e-01,
            6.05035000e-01,
            -2.68431900e-01,
            8.78457900e-01,
            8.49713500e-01,
            8.97218400e-01,
            1.33282830e00,
            3.47787830e-01,
            2.09215670e-01,
            -6.50969000e-03,
        ]
    )

    X_W_H = solve_fk(q_fr3=DEFAULT_Q_FR3, q_algr=DEFAULT_Q_ALGR)
    print(f"X_W_H: {X_W_H}")


if __name__ == "__main__":
    main()
