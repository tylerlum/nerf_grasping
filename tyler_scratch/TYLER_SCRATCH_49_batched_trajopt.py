# %%
import pathlib
import time
from typing import Optional, Tuple

import numpy as np
import pybullet as pb
import torch
import torch.nn.functional as F
import trimesh
import yaml
from curobo.cuda_robot_model.cuda_robot_model import (
    CudaRobotModel,
)
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from tqdm import tqdm

from nerf_grasping.curobo_fr3_algr_zed2i.fk_fr3_algr_zed2i import solve_fks
from nerf_grasping.curobo_fr3_algr_zed2i.fr3_algr_zed2i_world import (
    get_dummy_collision_dict,
    get_object_collision_dict,
    get_table_collision_dict,
)
from nerf_grasping.curobo_fr3_algr_zed2i.ik_fr3_algr_zed2i import (
    max_penetration_from_qs,
)
from nerf_grasping.curobo_fr3_algr_zed2i.pybullet_utils import (
    draw_collision_spheres,
    remove_collision_spheres,
)
from nerf_grasping.curobo_fr3_algr_zed2i.trajopt_fr3_algr_zed2i import (
    DEFAULT_Q,
    DEFAULT_Q_ALGR,
    DEFAULT_Q_FR3,
)

# %%
FR3_ALGR_ZED2I_URDF_PATH = pathlib.Path(
    "/juno/u/tylerlum/github_repos/nerf_grasping/nerf_grasping/fr3_algr_ik/allegro_ros2/models/fr3_algr_zed2i.urdf"
)
assert FR3_ALGR_ZED2I_URDF_PATH.exists()

GRASP_CONFIG_DICTS_PATH = pathlib.Path(
    "/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-05-02_16-19-22/optimized_grasp_config_dicts/mug_330_0_9999.npy"
)
assert GRASP_CONFIG_DICTS_PATH.exists()

OBJECT_OBJ_PATH = pathlib.Path(
    "/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-05-02_16-19-22/nerf_to_mesh/mug_330/coacd/decomposed.obj"
)
OBJECT_URDF_PATH = pathlib.Path(
    "/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-05-02_16-19-22/nerf_to_mesh/mug_330/coacd/coacd.urdf"
)
assert OBJECT_OBJ_PATH.exists()
assert OBJECT_URDF_PATH.exists()

# %%
COLLISION_SPHERES_YAML_PATH = load_yaml(
    join_path(get_robot_configs_path(), "fr3_algr_zed2i_with_fingertips.yml")
)["robot_cfg"]["kinematics"]["collision_spheres"]
COLLISION_SPHERES_YAML_PATH = pathlib.Path(
    join_path(get_robot_configs_path(), COLLISION_SPHERES_YAML_PATH)
)
assert COLLISION_SPHERES_YAML_PATH.exists()


# %%
grasp_config_dict = np.load(GRASP_CONFIG_DICTS_PATH, allow_pickle=True).item()
all_joint_angles = grasp_config_dict["joint_angles"]

# %%
MODE = "JOINTS_OPEN"
print("!" * 80)
print(f"MODE: {MODE}")
print("!" * 80 + "\n")
if MODE == "DEFAULT":
    all_joint_angles = all_joint_angles
elif MODE == "EXTRA_OPEN":
    raise NotImplementedError
elif MODE == "JOINTS_OPEN":
    DELTA = 0.1
    all_joint_angles[:, 1] -= DELTA
    all_joint_angles[:, 2] -= DELTA
    all_joint_angles[:, 3] -= DELTA

    all_joint_angles[:, 5] -= DELTA
    all_joint_angles[:, 6] -= DELTA
    all_joint_angles[:, 7] -= DELTA

    all_joint_angles[:, 9] -= DELTA
    all_joint_angles[:, 10] -= DELTA
    all_joint_angles[:, 11] -= DELTA
else:
    raise ValueError(f"Invalid MODE: {MODE}")

# %%
X_W_N = trimesh.transformations.translation_matrix([0.65, 0, 0])
X_O_Oy = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
obj_centroid = trimesh.load(OBJECT_OBJ_PATH).centroid
print(f"obj_centroid = {obj_centroid}")
X_N_O = trimesh.transformations.translation_matrix(obj_centroid)
X_W_Oy = X_W_N @ X_N_O @ X_O_Oy

# %%
N_GRASPS = grasp_config_dict["trans"].shape[0]
X_Oy_Hs = np.eye(4)[None, ...].repeat(N_GRASPS, axis=0)
X_Oy_Hs[:, :3, :3] = grasp_config_dict["rot"]
X_Oy_Hs[:, :3, 3] = grasp_config_dict["trans"]

# %%
X_W_Hs = []
for i in range(N_GRASPS):
    X_Oy_H = X_Oy_Hs[i]
    X_W_H = X_W_Oy @ X_Oy_H
    X_W_Hs.append(X_W_H)
X_W_Hs = np.stack(X_W_Hs, axis=0)


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


# %%
from localscope import localscope


@localscope.mfc(allowed=["DEFAULT_Q_FR3"])
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
    raise_if_fail: bool = True,
    warn_if_fail: bool = False,
    timeout: float = 5.0,
):
    start_time = time.time()
    print("Step 1: Prepare cfg")
    N_GRASPS = X_W_Hs.shape[0]
    assert X_W_Hs.shape == (N_GRASPS, 4, 4), f"X_W_Hs.shape: {X_W_Hs.shape}"
    assert q_algrs.shape == (N_GRASPS, 16), f"q_algrs.shape: {q_algrs.shape}"

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

    world_dict = {}
    if collision_check_table:
        world_dict.update(get_table_collision_dict())
    if collision_check_object and obj_filepath is not None:
        world_dict.update(
            get_object_collision_dict(
                file_path=obj_filepath, xyz=obj_xyz, quat_wxyz=obj_quat_wxyz
            )
        )
    if len(world_dict) == 0:
        world_dict.update(get_dummy_collision_dict())
    world_cfg = WorldConfig.from_dict(world_dict)

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

    result = ik_solver.solve_batch(target_pose)

    print("Step 3: Solve FK for fingertip poses")
    q_fr3s = result.solution[..., :7].squeeze(dim=1).detach().cpu().numpy()

    kin_model = CudaRobotModel(robot_cfg.kinematics)
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
    ik_result = ik_solver2.solve_batch(
        goal_pose=target_pose, link_poses=state.link_pose
    )
    qs = ik_result.solution.detach().cpu().numpy()

    print("Step 5: Solve motion generation")
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

    from curobo.types.robot import JointState

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

    motion_result = motion_gen.plan_batch(
        start_state=start_state,
        goal_pose=target_pose,
        plan_config=MotionGenPlanConfig(
            enable_graph=True,
            enable_opt=False,
            max_attempts=10,
            num_trajopt_seeds=10,
            num_graph_seeds=1,
            timeout=10,
        ),
        link_poses=state.link_pose,
    )
    paths = motion_result.get_paths()

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")

    return motion_result


# %%

RESULT = solve_trajopt_batch(
    X_W_Hs=X_W_Hs,
    q_algrs=all_joint_angles,
    collision_check_object=True,
    obj_filepath=OBJECT_OBJ_PATH,
    obj_xyz=(0.65, 0.0, 0.0),
    obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    collision_check_table=True,
    use_cuda_graph=False,
    enable_graph=True,
    enable_opt=False,
    raise_if_fail=False,
    warn_if_fail=False,
    timeout=5.0,
)

# %%
RESULT.success.nonzero()

# %%
all_qs = RESULT.optimized_plan.position.detach().cpu().numpy()

# %%
all_qs.shape


# %%
if not hasattr(pb, "HAS_BEEN_INITIALIZED"):
    pb.HAS_BEEN_INITIALIZED = True

    pb.connect(pb.GUI)
    r = pb.loadURDF(
        str(FR3_ALGR_ZED2I_URDF_PATH),
        useFixedBase=True,
        basePosition=[0, 0, 0],
        baseOrientation=[0, 0, 0, 1],
    )
    num_total_joints = pb.getNumJoints(r)
    assert num_total_joints == 39

    obj = pb.loadURDF(
        str(OBJECT_URDF_PATH),
        useFixedBase=True,
        basePosition=[
            0.65,
            0,
            0,
        ],
        baseOrientation=[0, 0, 0, 1],
    )

# %%
joint_names = [
    pb.getJointInfo(r, i)[1].decode("utf-8")
    for i in range(num_total_joints)
    if pb.getJointInfo(r, i)[2] != pb.JOINT_FIXED
]
link_names = [
    pb.getJointInfo(r, i)[12].decode("utf-8")
    for i in range(num_total_joints)
    if pb.getJointInfo(r, i)[2] != pb.JOINT_FIXED
]

actuatable_joint_idxs = [
    i for i in range(num_total_joints) if pb.getJointInfo(r, i)[2] != pb.JOINT_FIXED
]
num_actuatable_joints = len(actuatable_joint_idxs)
assert num_actuatable_joints == 23
arm_actuatable_joint_idxs = actuatable_joint_idxs[:7]
hand_actuatable_joint_idxs = actuatable_joint_idxs[7:]

for i, joint_idx in enumerate(arm_actuatable_joint_idxs):
    pb.resetJointState(r, joint_idx, DEFAULT_Q_FR3[i])

for i, joint_idx in enumerate(hand_actuatable_joint_idxs):
    pb.resetJointState(r, joint_idx, DEFAULT_Q_ALGR[i])

# %%
collision_config = yaml.safe_load(
    open(
        COLLISION_SPHERES_YAML_PATH,
        "r",
    )
)
draw_collision_spheres(
    robot=r,
    config=collision_config,
)


# %%
TRAJ_IDX = 2
qs = all_qs[TRAJ_IDX]
N_pts = qs.shape[0]
TOTAL_TIME = 5.0
dt = TOTAL_TIME / N_pts

# %%
remove_collision_spheres()

last_update_time = time.time()
for i in tqdm(range(N_pts)):
    position = qs[i]
    assert position.shape == (23,)
    # print(f"{i} / {N_pts} {position}")

    for i, joint_idx in enumerate(arm_actuatable_joint_idxs):
        pb.resetJointState(r, joint_idx, position[i])
    for i, joint_idx in enumerate(hand_actuatable_joint_idxs):
        pb.resetJointState(r, joint_idx, position[i + 7])

    time_since_last_update = time.time() - last_update_time
    if time_since_last_update <= dt:
        time.sleep(dt - time_since_last_update)
    last_update_time = time.time()

# %%
from nerf_grasping.curobo_fr3_algr_zed2i.ik_fr3_algr_zed2i import (
    max_penetration_from_qs,
)

d_world, d_self = max_penetration_from_qs(
    qs=qs,
    include_object=True,
    obj_filepath=OBJECT_OBJ_PATH,
    obj_xyz=(0.65, 0.0, 0.0),
    obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    include_table=True,
)

# %%
np.max(d_world), np.max(d_self)

# %%
