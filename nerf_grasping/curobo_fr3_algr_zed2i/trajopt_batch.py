import pathlib
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from curobo.cuda_robot_model.cuda_robot_model import (
    CudaRobotModel,
)

from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig, JointState
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

from nerf_grasping.curobo_fr3_algr_zed2i.fr3_algr_zed2i_world import (
    get_world_cfg,
)
from nerf_grasping.curobo_fr3_algr_zed2i.trajopt_fr3_algr_zed2i import (
    DEFAULT_Q_FR3,
    DEFAULT_Q_ALGR,
)

from nerf_grasping.optimizer_utils import (
    is_in_limits,
)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    subgradient is zero where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quat_wxyz(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quat_wxyz.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        quat_wxyz with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def solve_trajopt_batch(
    X_W_Hs: np.ndarray,
    q_algrs: np.ndarray,
    q_fr3_starts: Optional[np.ndarray] = None,
    collision_check_object: bool = True,
    obj_filepath: Optional[pathlib.Path] = pathlib.Path(
        "/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-05-02_16-19-22/nerf_to_mesh/mug_330/coacd/decomposed.obj"
    ),
    obj_xyz: Tuple[float, float, float] = (0.65, 0.0, 0.0),
    obj_quat_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    collision_check_table: bool = True,
    use_cuda_graph: bool = False,  # Getting some errors from setting this to True
    enable_graph: bool = True,
    enable_opt: bool = False,  # Getting some errors from setting this to True
    timeout: float = 5.0,
):
    start_time = time.time()
    print("Step 1: Prepare cfg")
    N_GRASPS = X_W_Hs.shape[0]
    assert X_W_Hs.shape == (N_GRASPS, 4, 4), f"X_W_Hs.shape: {X_W_Hs.shape}"
    assert q_algrs.shape == (N_GRASPS, 16), f"q_algrs.shape: {q_algrs.shape}"
    assert is_in_limits(q_algrs).all(), f"q_algrs: {q_algrs}"

    if q_fr3_starts is None:
        q_fr3_starts = DEFAULT_Q_FR3[None, ...].repeat(N_GRASPS, axis=0)
    assert q_fr3_starts.shape == (
        N_GRASPS,
        7,
    ), f"q_fr3_starts.shape: {q_fr3_starts.shape}"

    trans = X_W_Hs[:, :3, 3]
    rot_matrix = X_W_Hs[:, :3, :3]
    quat_wxyz = matrix_to_quat_wxyz(torch.from_numpy(rot_matrix).float().cuda())

    target_pose = Pose(
        torch.from_numpy(trans).float().cuda(),
        quaternion=quat_wxyz,
    )

    tensor_args = TensorDeviceType()
    robot_file = "fr3_algr_zed2i_with_fingertips.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )

    world_cfg = get_world_cfg(
        collision_check_object=collision_check_object,
        obj_filepath=obj_filepath,
        obj_xyz=obj_xyz,
        obj_quat_wxyz=obj_quat_wxyz,
        collision_check_table=collision_check_table,
    )

    print("Step 2: Solve IK for wrist pose")
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        rotation_threshold=0.01,
        position_threshold=0.001,
        num_seeds=20,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=use_cuda_graph,
    )
    ik_solver = IKSolver(ik_config)

    ik_result = ik_solver.solve_batch(target_pose)

    print("Step 3: Solve FK for fingertip poses")
    kin_model = CudaRobotModel(robot_cfg.kinematics)
    q_fr3s = ik_result.solution[..., :7].squeeze(dim=1).detach().cpu().numpy()
    q = torch.from_numpy(np.concatenate([q_fr3s, q_algrs], axis=1)).float().cuda()
    assert q.shape == (N_GRASPS, 23)
    state = kin_model.get_state(q)

    print("Step 4: Solve IK for wrist pose and fingertip poses")
    ik_config2 = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=20,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=use_cuda_graph,
    )
    ik_solver2 = IKSolver(ik_config2)
    ik_result2 = ik_solver2.solve_batch(
        goal_pose=target_pose, link_poses=state.link_pose
    )
    # print(f"ik_result2.success: {ik_result2.success.nonzero()}")

    print("Step 5: Solve FK for new fingertip poses")
    kin_model2 = CudaRobotModel(robot_cfg.kinematics)
    q2 = ik_result2.solution.squeeze(dim=1)

    assert q2.shape == (N_GRASPS, 23)
    state2 = kin_model2.get_state(q2)

    print("Step 6: Solve motion generation")
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.wrap.reacher.motion_gen import (
        MotionGen,
        MotionGenConfig,
        MotionGenPlanConfig,
    )

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        use_cuda_graph=use_cuda_graph,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup(batch=N_GRASPS)

    start_state = JointState.from_position(
        torch.from_numpy(
            np.concatenate(
                [
                    q_fr3_starts,
                    q_algrs,
                ],
                axis=1,
            )
        )
        .float()
        .cuda()
    )

    target_pose2 = Pose(
        state2.ee_position,
        quaternion=state2.ee_quaternion,
    )
    motion_result = motion_gen.plan_batch(
        start_state=start_state,
        goal_pose=target_pose2,
        plan_config=MotionGenPlanConfig(
            enable_graph=enable_graph,
            enable_opt=enable_opt,
            max_attempts=10,
            num_trajopt_seeds=10,
            num_graph_seeds=1,
            timeout=timeout,
        ),
        link_poses=state2.link_pose,
    )

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")

    return motion_result


def main() -> None:
    X_W_H_feasible = np.array(
        [
            [0, 0, 1, 0.4],
            [0, 1, 0, 0.0],
            [-1, 0, 0, 0.2],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    X_W_H_feasible2 = np.array(
        [
            [0, 0, 1, 0.5],
            [0, 1, 0, 0.0],
            [-1, 0, 0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    X_W_Hs = np.stack([X_W_H_feasible, X_W_H_feasible2], axis=0)
    q_algrs = np.stack([DEFAULT_Q_ALGR, DEFAULT_Q_ALGR], axis=0)

    obj_filepath = pathlib.Path(
        "/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-05-02_16-19-22/nerf_to_mesh/mug_330/coacd/decomposed.obj"
    )

    result = solve_trajopt_batch(
        X_W_Hs=X_W_Hs,
        q_algrs=q_algrs,
        collision_check_object=True,
        obj_filepath=obj_filepath,
        obj_xyz=(0.65, 0.0, 0.0),
        obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        collision_check_table=True,
        use_cuda_graph=False,
        enable_graph=True,
        enable_opt=True,
        timeout=10.0,
    )
    paths = result.get_paths()
    assert len(paths) == 2

    q = paths[0].position.detach().cpu().numpy()
    q2 = paths[1].position.detach().cpu().numpy()
    N_TIMESTEPS = q.shape[0]
    assert q.shape == (N_TIMESTEPS, 23)
    print(f"Success: {result.success}")
    print(f"q.shape: {q.shape}, q2.shape: {q2.shape}")
    print(f"q[0]: {q[0]}")
    print(f"q2[0]: {q2[0]}")


if __name__ == "__main__":
    main()