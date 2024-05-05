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
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig, IKResult

from nerf_grasping.curobo_fr3_algr_zed2i.fr3_algr_zed2i_world import (
    get_world_cfg,
)


def solve_ik(
    X_W_H: np.ndarray,
    world_cfg: WorldConfig,
    q_algr_constraint: Optional[np.ndarray] = None,
    raise_if_no_solution: bool = True,
    warn_if_no_solution: bool = False,
    use_cuda_graph: bool = True,
) -> Tuple[np.ndarray, IKResult, IKSolver]:
    assert X_W_H.shape == (4, 4), f"X_W_H.shape: {X_W_H.shape}"
    trans = X_W_H[:3, 3]
    rot_matrix = X_W_H[:3, :3]
    quat_wxyz = transforms3d.quaternions.mat2quat(rot_matrix)

    target_pose = Pose(
        torch.from_numpy(trans).float().cuda(),
        quaternion=torch.from_numpy(quat_wxyz).float().cuda(),
    )

    tensor_args = TensorDeviceType()
    robot_file = "fr3_algr_zed2i.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )

    # Apply joint limits
    if q_algr_constraint is not None:
        assert q_algr_constraint.shape == (
            16,
        ), f"q_algr_constraint.shape: {q_algr_constraint.shape}"
        assert robot_cfg.kinematics.kinematics_config.joint_limits.position.shape == (
            2,
            23,
        )
        robot_cfg.kinematics.kinematics_config.joint_limits.position[0, 7:] = (
            torch.from_numpy(q_algr_constraint).float().cuda() - 0.01
        )
        robot_cfg.kinematics.kinematics_config.joint_limits.position[1, 7:] = (
            torch.from_numpy(q_algr_constraint).float().cuda() + 0.01
        )

    ik_config = IKSolverConfig.load_from_robot_config(
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
    ik_solver = IKSolver(ik_config)

    result = ik_solver.solve_single(target_pose)
    if not result.success.item():
        if raise_if_no_solution:
            raise RuntimeError("IK failed to find a valid solution.")
        elif warn_if_no_solution:
            print("WARNING: IK failed to find a valid solution.")

    assert result.solution.shape == (1, 1, 23)
    return (
        result.solution.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy(),
        result,
        ik_solver,
    )


def max_penetration_from_q(
    q: np.ndarray,
    world_cfg: WorldConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    assert q.shape == (23,), f"q.shape: {q.shape}"

    robot_file = "fr3_algr_zed2i.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )

    config = RobotWorldConfig.load_from_config(
        robot_cfg, world_cfg, collision_activation_distance=0.0
    )
    curobo_fn = RobotWorld(config)
    d_world, d_self = curobo_fn.get_world_self_collision_distance_from_joints(
        torch.from_numpy(q[None, ...]).float().cuda()
    )
    return (
        d_world.squeeze(dim=0).detach().cpu().numpy(),
        d_self.squeeze(dim=0).detach().cpu().numpy(),
    )


def max_penetration_from_qs(
    qs: np.ndarray,
    world_cfg: WorldConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    N = qs.shape[0]
    assert qs.shape == (
        N,
        23,
    ), f"qs.shape: {qs.shape}"

    robot_file = "fr3_algr_zed2i.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )

    config = RobotWorldConfig.load_from_config(
        robot_cfg, world_cfg, collision_activation_distance=0.0
    )
    curobo_fn = RobotWorld(config)
    d_world, d_self = curobo_fn.get_world_self_collision_distance_from_joints(
        torch.from_numpy(qs).float().cuda()
    )
    return (
        d_world.detach().cpu().numpy(),
        d_self.detach().cpu().numpy(),
    )


def max_penetration_from_X_W_H(
    X_W_H: np.ndarray,
    world_cfg: WorldConfig,
    q_algr_constraint: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    q_solution, _, _ = solve_ik(
        X_W_H=X_W_H,
        world_cfg=world_cfg,
        q_algr_constraint=q_algr_constraint,
        raise_if_no_solution=False,
    )
    return max_penetration_from_q(
        q=q_solution,
        world_cfg=world_cfg,
    )


def main() -> None:
    X_W_H_feasible = np.array(
        [
            [0, 0, 1, 0.4],
            [0, 1, 0, 0.0],
            [-1, 0, 0, 0.2],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    X_W_H_collide_object = np.array(
        [
            [0, 0, 1, 0.65],
            [0, 1, 0, 0.0],
            [-1, 0, 0, 0.2],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    X_W_H_collide_table = np.array(
        [
            [0, 0, 1, 0.4],
            [0, 1, 0, 0.0],
            [-1, 0, 0, 0.10],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    q_algr_pre = np.array(
        [
            0.29094562,
            0.7371094,
            0.5108592,
            0.12263706,
            0.12012535,
            0.5845135,
            0.34382993,
            0.605035,
            -0.2684319,
            0.8784579,
            0.8497135,
            0.8972184,
            1.3328283,
            0.34778783,
            0.20921567,
            -0.00650969,
        ]
    )
    q_feasible, _, _ = solve_ik(
        X_W_H=X_W_H_feasible,
        world_cfg=get_world_cfg(include_object=False),
        q_algr_constraint=q_algr_pre,
    )
    q_feasible_2, _, _ = solve_ik(
        X_W_H=X_W_H_feasible,
        world_cfg=get_world_cfg(include_object=True),
        q_algr_constraint=q_algr_pre,
    )
    q_feasible_3, _, _ = solve_ik(
        X_W_H=X_W_H_collide_object,
        world_cfg=get_world_cfg(include_object=False),
        q_algr_constraint=q_algr_pre,
    )
    print(f"q_feasible: {q_feasible}")
    print(f"q_feasible_2: {q_feasible_2}")
    print(f"q_feasible_3: {q_feasible_3}")

    print("=" * 80)
    try:
        q_collide_object, _, _ = solve_ik(
            X_W_H=X_W_H_collide_object,
            world_cfg=get_world_cfg(include_object=True),
            q_algr_constraint=q_algr_pre,
        )
        raise ValueError("Collision check failed to detect collision.")
    except RuntimeError:
        print("Collision check successfully detected collision.")
        max_penetration_collide_object = max_penetration_from_X_W_H(
            X_W_H=X_W_H_collide_object,
            world_cfg=get_world_cfg(),
            q_algr_constraint=q_algr_pre,
        )
        print(f"max_penetration_collide_object = {max_penetration_collide_object}")
        max_penetration_collide_object_turn_off_object = max_penetration_from_X_W_H(
            X_W_H=X_W_H_collide_object,
            world_cfg=get_world_cfg(include_object=False),
            q_algr_constraint=q_algr_pre,
        )
        print(
            f"max_penetration_collide_object_turn_off_object = {max_penetration_collide_object_turn_off_object}"
        )
        max_penetration_collide_object_turn_off_table = max_penetration_from_X_W_H(
            X_W_H=X_W_H_collide_object,
            world_cfg=get_world_cfg(include_table=False),
            q_algr_constraint=q_algr_pre,
        )
        print(
            f"max_penetration_collide_object_turn_off_table = {max_penetration_collide_object_turn_off_table}"
        )
    print("=" * 80 + "\n")

    print("=" * 80)
    try:
        q_collide_table, _, _ = solve_ik(
            X_W_H=X_W_H_collide_table,
            world_cfg=get_world_cfg(),
            q_algr_constraint=q_algr_pre,
        )
        raise ValueError("Collision check failed to detect collision.")
    except RuntimeError:
        print("Collision check successfully detected collision.")
        max_penetration_collide_table = max_penetration_from_X_W_H(
            X_W_H=X_W_H_collide_table,
            world_cfg=get_world_cfg(),
            q_algr_constraint=q_algr_pre,
        )
        print(f"max_penetration_collide_table = {max_penetration_collide_table}")
        max_penetration_collide_table_turn_off_table = max_penetration_from_X_W_H(
            X_W_H=X_W_H_collide_table,
            world_cfg=get_world_cfg(include_table=False),
            q_algr_constraint=q_algr_pre,
        )
        print(
            f"max_penetration_collide_table_turn_off_table = {max_penetration_collide_table_turn_off_table}"
        )
        max_penetration_collide_table_turn_off_object = max_penetration_from_X_W_H(
            X_W_H=X_W_H_collide_table,
            world_cfg=get_world_cfg(include_object=False),
            q_algr_constraint=q_algr_pre,
        )
        print(
            f"max_penetration_collide_table_turn_off_object = {max_penetration_collide_table_turn_off_object}"
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
