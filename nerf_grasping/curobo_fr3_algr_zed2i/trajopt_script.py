# %%
import pathlib
import time

import numpy as np
import pybullet as pb
import trimesh
import yaml

from nerf_grasping.fr3_algr_trajopt.trajopt import TrajOptParams
from nerf_grasping.fr3_algr_trajopt.trajopt import solve_trajopt as solve_trajopt_drake

from nerf_grasping.fr3_algr_ik.ik import solve_ik as solve_ik_drake
from nerf_grasping.curobo_fr3_algr_zed2i.trajopt_fr3_algr_zed2i import (
    DEFAULT_Q_ALGR, DEFAULT_Q_FR3, DEFAULT_Q,
)

from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from nerf_grasping.curobo_fr3_algr_zed2i.ik_fr3_algr_zed2i import (
    max_penetration_from_q,
    max_penetration_from_qs,
    max_penetration_from_X_W_H,
    solve_ik,
)
from nerf_grasping.curobo_fr3_algr_zed2i.pybullet_utils import (
    draw_collision_spheres,
    remove_collision_spheres,
)
from tqdm import tqdm
from nerf_grasping.curobo_fr3_algr_zed2i.trajopt_fr3_algr_zed2i import (
    DEFAULT_Q_ALGR,
    DEFAULT_Q_FR3,
    solve_trajopt,
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
    join_path(get_robot_configs_path(), "fr3_algr_zed2i.yml")
)["robot_cfg"]["kinematics"]["collision_spheres"]
COLLISION_SPHERES_YAML_PATH = pathlib.Path(
    join_path(get_robot_configs_path(), COLLISION_SPHERES_YAML_PATH)
)
assert COLLISION_SPHERES_YAML_PATH.exists()


# %%
grasp_config_dict = np.load(GRASP_CONFIG_DICTS_PATH, allow_pickle=True).item()
BEST_IDX = 4
GOOD_IDX = 0
GOOD_IDX_2 = 1

# SELECTED_IDX = 2
SELECTED_IDX = 0
# SELECTED_IDX = BEST_IDX

trans = grasp_config_dict["trans"][SELECTED_IDX]
rot = grasp_config_dict["rot"][SELECTED_IDX]
joint_angles = grasp_config_dict["joint_angles"][SELECTED_IDX]
X_Oy_H = np.eye(4)
X_Oy_H[:3, :3] = rot
X_Oy_H[:3, 3] = trans

# %%
X_W_N = trimesh.transformations.translation_matrix([0.65, 0, 0])
X_O_Oy = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
obj_centroid = trimesh.load(OBJECT_OBJ_PATH).centroid
print(f"obj_centroid = {obj_centroid}")
X_N_O = trimesh.transformations.translation_matrix(obj_centroid)
X_W_Oy = X_W_N @ X_N_O @ X_O_Oy

X_W_H = X_W_Oy @ X_Oy_H
q_algr_pre = joint_angles

# %%
d_world, d_self = max_penetration_from_X_W_H(
    X_W_H=X_W_H,
    q_algr_constraint=q_algr_pre,
    include_object=True,
    obj_filepath=OBJECT_OBJ_PATH,
    obj_xyz=(0.65, 0.0, 0.0),
    obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    include_table=True,
)
print(f"d_world = {d_world}, d_self = {d_self}")
if d_world.item() > 0.0:
    print("WARNING: penetration with world detected")
if d_self.item() > 0.0:
    print("WARNING: self collision detected")

d_world, d_self = max_penetration_from_X_W_H(
    X_W_H=X_W_H,
    q_algr_constraint=q_algr_pre,
    include_object=False,
    obj_filepath=OBJECT_OBJ_PATH,
    obj_xyz=(0.65, 0.0, 0.0),
    obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    include_table=True,
)
print(f"Without object: d_world = {d_world}, d_self = {d_self}")

d_world, d_self = max_penetration_from_X_W_H(
    X_W_H=X_W_H,
    q_algr_constraint=q_algr_pre,
    include_object=True,
    obj_filepath=OBJECT_OBJ_PATH,
    obj_xyz=(0.65, 0.0, 0.0),
    obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    include_table=False,
)
print(f"Without table: d_world = {d_world}, d_self = {d_self}")

d_world, d_self = max_penetration_from_X_W_H(
    X_W_H=X_W_H,
    q_algr_constraint=q_algr_pre,
    include_object=False,
    obj_filepath=OBJECT_OBJ_PATH,
    obj_xyz=(0.65, 0.0, 0.0),
    obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    include_table=False,
)
print(f"Without object or table: d_world = {d_world}, d_self = {d_self}")

open_hand_q_algr = q_algr_pre.copy()
DELTA = 0.1
open_hand_q_algr[1] -= DELTA
open_hand_q_algr[2] -= DELTA
open_hand_q_algr[3] -= DELTA

open_hand_q_algr[5] -= DELTA
open_hand_q_algr[6] -= DELTA
open_hand_q_algr[7] -= DELTA

open_hand_q_algr[9] -= DELTA
open_hand_q_algr[10] -= DELTA
open_hand_q_algr[11] -= DELTA
d_world, d_self = max_penetration_from_X_W_H(
    X_W_H=X_W_H,
    q_algr_constraint=open_hand_q_algr,
    include_object=True,
    obj_filepath=OBJECT_OBJ_PATH,
    obj_xyz=(0.65, 0.0, 0.0),
    obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    include_table=True,
)
print(f"DELTA = {DELTA}, d_world = {d_world}, d_self = {d_self}")

open_hand_q_algr = q_algr_pre.copy()
DELTA = 0.2
open_hand_q_algr[1] -= DELTA
open_hand_q_algr[2] -= DELTA
open_hand_q_algr[3] -= DELTA

open_hand_q_algr[5] -= DELTA
open_hand_q_algr[6] -= DELTA
open_hand_q_algr[7] -= DELTA

open_hand_q_algr[9] -= DELTA
open_hand_q_algr[10] -= DELTA
open_hand_q_algr[11] -= DELTA
d_world, d_self = max_penetration_from_X_W_H(
    X_W_H=X_W_H,
    q_algr_constraint=open_hand_q_algr,
    include_object=True,
    obj_filepath=OBJECT_OBJ_PATH,
    obj_xyz=(0.65, 0.0, 0.0),
    obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    include_table=True,
)
print(f"DELTA = {DELTA}, d_world = {d_world}, d_self = {d_self}")

open_hand_q_algr = q_algr_pre.copy()
DELTA = 0.3
open_hand_q_algr[1] -= DELTA
open_hand_q_algr[2] -= DELTA
open_hand_q_algr[3] -= DELTA

open_hand_q_algr[5] -= DELTA
open_hand_q_algr[6] -= DELTA
open_hand_q_algr[7] -= DELTA

open_hand_q_algr[9] -= DELTA
open_hand_q_algr[10] -= DELTA
open_hand_q_algr[11] -= DELTA
d_world, d_self = max_penetration_from_X_W_H(
    X_W_H=X_W_H,
    q_algr_constraint=open_hand_q_algr,
    include_object=True,
    obj_filepath=OBJECT_OBJ_PATH,
    obj_xyz=(0.65, 0.0, 0.0),
    obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    include_table=True,
)
print(f"DELTA = {DELTA}, d_world = {d_world}, d_self = {d_self}")


# %%
print("=" * 80)
print("Trying with full object collision check")
print("=" * 80 + "\n")
q, qd, qdd, dt, result, motion_gen = solve_trajopt(
    X_W_H=X_W_H,
    q_algr_constraint=q_algr_pre,
    collision_check_object=True,
    obj_filepath=OBJECT_OBJ_PATH,
    obj_xyz=(0.65, 0.0, 0.0),
    obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    collision_check_table=True,
    enable_opt=True,
    enable_graph=True,
    raise_if_fail=False,
    use_cuda_graph=False
)
result
if result.success:
    print("SUCCESS TRAJOPT with full object collision check")
    failed = False
else:
    print("FAILED TRAJOPT with full object collision check")
    failed = True

if failed:
    print("=" * 80)
    print("Trying with full object collision check without trajopt")
    print("=" * 80 + "\n")
    q, qd, qdd, dt, result, motion_gen = solve_trajopt(
        X_W_H=X_W_H,
        q_algr_constraint=q_algr_pre,
        collision_check_object=True,
        obj_filepath=OBJECT_OBJ_PATH,
        obj_xyz=(0.65, 0.0, 0.0),
        obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        collision_check_table=True,
        enable_opt=False,
        enable_graph=True,
        raise_if_fail=False,
        use_cuda_graph=False,
    )
    if result.success:
        print("SUCCESS TRAJOPT with full object collision check without trajopt")
        failed = False
    else:
        print("FAILED TRAJOPT with full object collision check without trajopt")
        failed = True

if failed:
    print("=" * 80)
    print("Trying with open hand")
    print("=" * 80 + "\n")
    failed = False
    open_hand_q_algr = q_algr_pre.copy()
    DELTA = 0.1
    open_hand_q_algr[1] -= DELTA
    open_hand_q_algr[2] -= DELTA
    open_hand_q_algr[3] -= DELTA

    open_hand_q_algr[5] -= DELTA
    open_hand_q_algr[6] -= DELTA
    open_hand_q_algr[7] -= DELTA

    open_hand_q_algr[9] -= DELTA
    open_hand_q_algr[10] -= DELTA
    open_hand_q_algr[11] -= DELTA

    old_q_algr_pre = q_algr_pre.copy()
    q_algr_pre = open_hand_q_algr

    try:
        q, qd, qdd, dt, result, motion_gen = solve_trajopt(
            X_W_H=X_W_H,
            q_algr_constraint=open_hand_q_algr,
            collision_check_object=True,
            obj_filepath=OBJECT_OBJ_PATH,
            obj_xyz=(0.65, 0.0, 0.0),
            obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            collision_check_table=True,
        )
        print("SUCCESS TRAJOPT with open hand")
    except RuntimeError as e:
        print(f"FAILED TRAJOPT: {e} with open hand")
        failed = True

if failed:
    print("=" * 80)
    print("Trying without object")
    print("=" * 80 + "\n")
    failed = False
    try:
        q, qd, qdd, dt, result, motion_gen = solve_trajopt(
            X_W_H=X_W_H,
            q_algr_constraint=q_algr_pre,
            collision_check_object=False,
            obj_filepath=OBJECT_OBJ_PATH,
            obj_xyz=(0.65, 0.0, 0.0),
            obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            collision_check_table=True,
        )
        print("SUCCESS TRAJOPT without object collision check")
    except RuntimeError as e:
        print(f"FAILED TRAJOPT: {e} without object collision check")
        failed = True

if failed:
    print("=" * 80)
    print("Trying without object or table")
    print("=" * 80 + "\n")
    failed = False
    try:
        q, qd, qdd, dt, result, motion_gen = solve_trajopt(
            X_W_H=X_W_H,
            q_algr_constraint=q_algr_pre,
            collision_check_object=False,
            obj_filepath=OBJECT_OBJ_PATH,
            obj_xyz=(0.65, 0.0, 0.0),
            obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            collision_check_table=False,
        )
        print("SUCCESS TRAJOPT without object or table collision check")
    except RuntimeError as e:
        print(f"FAILED TRAJOPT: {e} without object or table collision check")
        failed = True

# %%
print(f"q.shape = {q.shape}, qd.shape = {qd.shape}, qdd.shape = {qdd.shape}, dt = {dt}")
N_pts = q.shape[0]
assert q.shape == (N_pts, 23)

# %%
print(f"result.success = {result.success}")
print(f"result.valid_query = {result.valid_query}")
print(f"result.optimized_plan.shape = {result.optimized_plan.shape}")
print(f"result.optimized_dt = {result.optimized_dt}")
print(f"result.position_error = {result.position_error}")
print(f"result.rotation_error = {result.rotation_error}")
print(f"result.cspace_error = {result.cspace_error}")
print(f"result.solve_time = {result.solve_time}")
print(f"result.ik_time = {result.ik_time}")
print(f"result.graph_time = {result.graph_time}")
print(f"result.trajopt_time = {result.trajopt_time}")
print(f"result.finetune_time = {result.finetune_time}")
print(f"result.total_time = {result.total_time}")
print(f"result.interpolated_plan.shape = {result.interpolated_plan.shape}")
print(f"result.interpolation_dt = {result.interpolation_dt}")
print(f"result.path_buffer_last_tstep = {result.path_buffer_last_tstep}")
print(f"result.debug_info = {result.debug_info}")
print(f"result.status = {result.status}")
print(f"result.attempts = {result.attempts}")
print(f"result.trajopt_attempts = {result.trajopt_attempts}")
print(f"result.used_graph = {result.used_graph}")
print(f"result.graph_plan.position.shape = {result.graph_plan.position.shape}")
print(f"result.goalset_index = {result.goalset_index}")

# %%
import matplotlib.pyplot as plt

# %%
plt.plot(q)

# %%
# from curobo.types.state import JointState
# import torch
# motion_gen.check_constraints(
#     JointState(
#         position=    torch.from_numpy(q).float().cuda(),
#         velocity=    torch.from_numpy(qd).float().cuda(),
#         acceleration=torch.from_numpy(qdd).float().cuda(),
#     )
# )

# %%
result.optimized_plan.shape

# %%

d_world, d_self = max_penetration_from_qs(
    qs=q,
    include_object=True,
    obj_filepath=OBJECT_OBJ_PATH,
    obj_xyz=(0.65, 0.0, 0.0),
    obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    include_table=True,
)

# %%
np.max(d_world), np.max(d_self)


# %%
(d_world > 0).sum()

# %%
(d_world > 0).nonzero()

# %%
np.argmax(d_world)

# %%
# for i, joint_idx in enumerate(arm_actuatable_joint_idxs):
#     pb.resetJointState(r, joint_idx, q[98][i])
# for i, joint_idx in enumerate(hand_actuatable_joint_idxs):
#     pb.resetJointState(r, joint_idx, q[98][i + 7])
# 
# draw_collision_spheres(
#     robot=r,
#     config=collision_config,
# )

# %%
total_time = N_pts * dt
print(f"total_time = {total_time}")

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
remove_collision_spheres()

last_update_time = time.time()
for i in tqdm(range(N_pts)):
    position = q[i]
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
draw_collision_spheres(
    robot=r,
    config=collision_config,
)

# %%
q_solution = solve_ik(
    X_W_H=X_W_H,
    q_algr_constraint=q_algr_pre,
    collision_check_object=True,
    obj_filepath=OBJECT_OBJ_PATH,
    obj_xyz=(0.65, 0.0, 0.0),
    obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    collision_check_table=True,
    raise_if_no_solution=True,
)

# %%
remove_collision_spheres()
# %%
for i, joint_idx in enumerate(arm_actuatable_joint_idxs):
    pb.resetJointState(r, joint_idx, q_solution[i])
for i, joint_idx in enumerate(hand_actuatable_joint_idxs):
    pb.resetJointState(r, joint_idx, q_solution[i + 7])

# %%
for i, joint_idx in enumerate(arm_actuatable_joint_idxs):
    pb.resetJointState(r, joint_idx, q[-1][i])
for i, joint_idx in enumerate(hand_actuatable_joint_idxs):
    pb.resetJointState(r, joint_idx, q[-1][i + 7])

# %%
q_solution - q[-1]

# %%
draw_collision_spheres(
    robot=r,
    config=collision_config,
)


# %%
 
# %%

# for i, joint_idx in enumerate(arm_actuatable_joint_idxs):
#     pb.resetJointState(r, joint_idx, q_solution[0, i])
# for i, joint_idx in enumerate(hand_actuatable_joint_idxs):
#     pb.resetJointState(r, joint_idx, q_solution[0, i + 7])

# # %%
#
#
# q_solution = solve_ik(
#     X_W_H=X_W_H,
#     q_algr_constraint=q_algr_pre,
#     collision_check_object=True,
#     obj_filepath=OBJECT_OBJ_PATH,
#     obj_xyz=(0.65, 0.0, 0.0),
#     obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
#     collision_check_table=True,
#     raise_if_no_solution=True,
# )
# # %%
#
# max_penetration_from_X_W_H(
#     X_W_H=X_W_H,
#     q_algr_constraint=q_algr_pre,
#     include_object=True,
#     obj_filepath=OBJECT_OBJ_PATH,
#     obj_xyz=(0.65, 0.0, 0.0),
#     obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
#     include_table=True,
# )
# # %%
# q_solution = solve_ik(
#     X_W_H=X_W_H,
#     q_algr_constraint=q_algr_pre,
#     collision_check_object=True,
#     obj_filepath=OBJECT_OBJ_PATH,
#     obj_xyz=(0.65, 0.0, 0.0),
#     obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
#     collision_check_table=True,
#     raise_if_no_solution=False,
# )
# # %%
# q_solution
# # %%
# max_penetration_from_q(
#     q=q_solution,
#     include_object=True,
#     obj_filepath=OBJECT_OBJ_PATH,
#     obj_xyz=(0.65, 0.0, 0.0),
#     obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
#     include_table=True,
# )
#
# # %%
#
# for i, joint_idx in enumerate(arm_actuatable_joint_idxs):
#     pb.resetJointState(r, joint_idx, q_solution[i])
# for i, joint_idx in enumerate(hand_actuatable_joint_idxs):
#     pb.resetJointState(r, joint_idx, q_solution[i + 7])
#
# # %%
# q_solution2 = solve_ik(
#     X_W_H=X_W_H,
#     q_algr_constraint=q_algr_pre,
#     collision_check_object=False,
#     obj_filepath=OBJECT_OBJ_PATH,
#     obj_xyz=(0.65, 0.0, 0.0),
#     obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
#     collision_check_table=True,
#     raise_if_no_solution=True,
# )
# # %%
# for i, joint_idx in enumerate(arm_actuatable_joint_idxs):
#     pb.resetJointState(r, joint_idx, q_solution2[i])
# for i, joint_idx in enumerate(hand_actuatable_joint_idxs):
#     pb.resetJointState(r, joint_idx, q_solution2[i + 7])
#
# # %%
#
# %%
# %%
NERF_FRAME_OFFSET = 0.65
cfg = TrajOptParams(
    num_control_points=21,
    min_self_coll_dist=0.005,
    influence_dist=0.01,
    nerf_frame_offset=NERF_FRAME_OFFSET,
    s_start_self_col=0.5,
    lqr_pos_weight=1e-1,
    lqr_vel_weight=20.0,
    presolve_no_collision=True,
)

# import nerf_grasping
# mesh_path = (
#     pathlib.Path(nerf_grasping.get_repo_root())
#     / "experiments/2024-05-01_15-39-42/nerf_to_mesh/mug_330/coacd/decomposed.obj"
# )
mesh_path = OBJECT_OBJ_PATH

# %%

assert X_W_H.shape == (4, 4)
assert q_algr_pre.shape == (16,)

q_robot_0 = np.concatenate([DEFAULT_Q_FR3, q_algr_pre])
q_robot_f = solve_ik_drake(X_W_H, q_algr_pre, visualize=False)
# mesh_path = None
try:
    spline, dspline, T_traj, trajopt = solve_trajopt_drake(
        q_fr3_0=q_robot_0[:7],
        q_algr_0=q_robot_0[7:],
        q_fr3_f=q_robot_f[:7],
        q_algr_f=q_robot_f[7:],
        cfg=cfg,
        mesh_path=mesh_path,
        visualize=True,
        verbose=True,
        ignore_obj_collision=False,
    )
    print("Trajectory optimization succeeded!")
except RuntimeError as e:
    print("Trajectory optimization failed")

# %%
q_robot_f

# %%
q[-1]



# %%
remove_collision_spheres()
for i, joint_idx in enumerate(arm_actuatable_joint_idxs):
    pb.resetJointState(r, joint_idx, q_robot_f[i])
for i, joint_idx in enumerate(hand_actuatable_joint_idxs):
    pb.resetJointState(r, joint_idx, q_robot_f[i + 7])

# %%
remove_collision_spheres()
for i, joint_idx in enumerate(arm_actuatable_joint_idxs):
    pb.resetJointState(r, joint_idx, q[-1][i])
for i, joint_idx in enumerate(hand_actuatable_joint_idxs):
    pb.resetJointState(r, joint_idx, q[-1][i + 7])

# %%
max_penetration_from_q(
    q=q_robot_f,
    include_object=True,
    obj_filepath=OBJECT_OBJ_PATH,
    obj_xyz=(0.65, 0.0, 0.0),
    obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    include_table=True,
)


# %%
n_grasps = grasp_config_dict["trans"].shape[0]

# %%
no_collision_idxs = []
pass_curobo_trajopt_idxs = []
pass_curobo_trajopt_without_object_idxs = []
OPEN_HAND = True
for i in tqdm(range(n_grasps), desc="Curobo"):
    trans = grasp_config_dict["trans"][i]
    rot = grasp_config_dict["rot"][i]
    joint_angles = grasp_config_dict["joint_angles"][i]
    X_Oy_H = np.eye(4)
    X_Oy_H[:3, :3] = rot
    X_Oy_H[:3, 3] = trans

    X_W_N = trimesh.transformations.translation_matrix([0.65, 0, 0])
    X_O_Oy = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    obj_centroid = trimesh.load(OBJECT_OBJ_PATH).centroid
    X_N_O = trimesh.transformations.translation_matrix(obj_centroid)
    X_W_Oy = X_W_N @ X_N_O @ X_O_Oy

    X_W_H = X_W_Oy @ X_Oy_H
    q_algr_pre = joint_angles

    if OPEN_HAND:
        open_hand_q_algr = q_algr_pre.copy()
        DELTA = 0.1
        open_hand_q_algr[1] -= DELTA
        open_hand_q_algr[2] -= DELTA
        open_hand_q_algr[3] -= DELTA

        open_hand_q_algr[5] -= DELTA
        open_hand_q_algr[6] -= DELTA
        open_hand_q_algr[7] -= DELTA

        open_hand_q_algr[9] -= DELTA
        open_hand_q_algr[10] -= DELTA
        open_hand_q_algr[11] -= DELTA
        q_algr_pre = open_hand_q_algr

    d_world, d_self = max_penetration_from_X_W_H(
        X_W_H=X_W_H,
        q_algr_constraint=q_algr_pre,
        include_object=True,
        obj_filepath=OBJECT_OBJ_PATH,
        obj_xyz=(0.65, 0.0, 0.0),
        obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        include_table=True,
    )
    if d_world.item() == 0.0 and d_self.item() == 0.0:
        no_collision_idxs.append(i)

    try:
        print("=" * 80)
        print("Trying with full object collision check")
        print("=" * 80 + "\n")
        q, qd, qdd, dt, result, motion_gen = solve_trajopt(
            X_W_H=X_W_H,
            q_algr_constraint=q_algr_pre,
            collision_check_object=True,
            obj_filepath=OBJECT_OBJ_PATH,
            obj_xyz=(0.65, 0.0, 0.0),
            obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            collision_check_table=True,
        )
        print("SUCCESS TRAJOPT with full object collision check")
        pass_curobo_trajopt_idxs.append(i)
    except RuntimeError as e:
        print(f"FAILED TRAJOPT: {e} with full object collision check")

    try:
        print("=" * 80)
        print("Trying with no object collision check")
        print("=" * 80 + "\n")
        q, qd, qdd, dt, result, motion_gen = solve_trajopt(
            X_W_H=X_W_H,
            q_algr_constraint=q_algr_pre,
            collision_check_object=False,
            obj_filepath=OBJECT_OBJ_PATH,
            obj_xyz=(0.65, 0.0, 0.0),
            obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            collision_check_table=True,
        )
        print("SUCCESS TRAJOPT with no object collision check")
        pass_curobo_trajopt_without_object_idxs.append(i)
    except RuntimeError as e:
        print(f"FAILED TRAJOPT: {e} with full object collision check")
if OPEN_HAND:
    print(f"WITH open hand DELTA = {DELTA}")
print(f"no_collision_idxs = {no_collision_idxs} ({len(no_collision_idxs)} / {n_grasps} = {len(no_collision_idxs) / n_grasps})")
print(f"pass_curobo_trajopt_idxs = {pass_curobo_trajopt_idxs} ({len(pass_curobo_trajopt_idxs)} / {n_grasps} = {len(pass_curobo_trajopt_idxs) / n_grasps})")
print(f"pass_curobo_trajopt_without_object_idxs = {pass_curobo_trajopt_without_object_idxs} ({len(pass_curobo_trajopt_without_object_idxs)} / {n_grasps} = {len(pass_curobo_trajopt_without_object_idxs) / n_grasps})")

# %%
pass_ik_idxs = []
pass_drake_trajopt_idxs = []
pass_drake_trajopt_without_object_idxs = []
for i in tqdm(range(n_grasps), desc="Drake"):
    trans = grasp_config_dict["trans"][i]
    rot = grasp_config_dict["rot"][i]
    joint_angles = grasp_config_dict["joint_angles"][i]
    X_Oy_H = np.eye(4)
    X_Oy_H[:3, :3] = rot
    X_Oy_H[:3, 3] = trans

    X_W_N = trimesh.transformations.translation_matrix([0.65, 0, 0])
    X_O_Oy = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    obj_centroid = trimesh.load(OBJECT_OBJ_PATH).centroid
    X_N_O = trimesh.transformations.translation_matrix(obj_centroid)
    X_W_Oy = X_W_N @ X_N_O @ X_O_Oy

    X_W_H = X_W_Oy @ X_Oy_H
    q_algr_pre = joint_angles

    q_robot_0 = np.concatenate([DEFAULT_Q_FR3, q_algr_pre])
    try:
        q_robot_f = solve_ik_drake(X_W_H, q_algr_pre, visualize=False)
        pass_ik_idxs.append(i)
    except RuntimeError as e:
        print(f"FAILED IK: {e}")
        continue

    # mesh_path = None
    try:
        spline, dspline, T_traj, trajopt = solve_trajopt_drake(
            q_fr3_0=q_robot_0[:7],
            q_algr_0=q_robot_0[7:],
            q_fr3_f=q_robot_f[:7],
            q_algr_f=q_robot_f[7:],
            cfg=cfg,
            mesh_path=mesh_path,
            visualize=False,
            verbose=False,
            ignore_obj_collision=False,
        )
        print("Trajectory optimization succeeded!")
        pass_drake_trajopt_idxs.append(i)
    except RuntimeError as e:
        print("Trajectory optimization failed")

    try:
        spline, dspline, T_traj, trajopt = solve_trajopt_drake(
            q_fr3_0=q_robot_0[:7],
            q_algr_0=q_robot_0[7:],
            q_fr3_f=q_robot_f[:7],
            q_algr_f=q_robot_f[7:],
            cfg=cfg,
            mesh_path=mesh_path,
            visualize=False,
            verbose=False,
            ignore_obj_collision=True,
        )
        print("Trajectory optimization succeeded!")
        pass_drake_trajopt_without_object_idxs.append(i)
    except RuntimeError as e:
        print("Trajectory optimization failed")
print(f"pass_ik_idxs = {pass_ik_idxs} ({len(pass_ik_idxs)} / {n_grasps} = {len(pass_ik_idxs) / n_grasps})")
print(f"pass_drake_trajopt_idxs = {pass_drake_trajopt_idxs} ({len(pass_drake_trajopt_idxs)} / {n_grasps} = {len(pass_drake_trajopt_idxs) / n_grasps})")
print(f"pass_drake_trajopt_without_object_idxs = {pass_drake_trajopt_without_object_idxs} ({len(pass_drake_trajopt_without_object_idxs)} / {n_grasps} = {len(pass_drake_trajopt_without_object_idxs) / n_grasps})")

# %%
spline, dspline, T_traj, trajopt = solve_trajopt_drake(
    q_fr3_0=q_robot_0[:7],
    q_algr_0=q_robot_0[7:],
    q_fr3_f=q_robot_f[:7],
    q_algr_f=q_robot_f[7:],
    cfg=cfg,
    mesh_path=mesh_path,
    visualize=True,
    verbose=True,
    ignore_obj_collision=True,
)
# %%
q.shape
# %%

from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState, RobotConfig
tensor_args = TensorDeviceType()
robot_file = "fr3_algr_zed2i.yml"
robot_cfg = RobotConfig.from_dict(
    load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
)
# %%
robot_cfg.kinematics.kinematics_config.joint_limits.velocity

# %%
q_limits = robot_cfg.kinematics.kinematics_config.joint_limits.position.detach().cpu().numpy()
qd_limits = robot_cfg.kinematics.kinematics_config.joint_limits.velocity.detach().cpu().numpy()
qdd_limits = robot_cfg.kinematics.kinematics_config.joint_limits.acceleration.detach().cpu().numpy()
qddd_limits = robot_cfg.kinematics.kinematics_config.joint_limits.jerk.detach().cpu().numpy()
print(q_limits.shape)

# %%
qd

# %%
TOTAL_TIME = 2.0
DT = TOTAL_TIME / N_pts
new_qd = np.diff(q, axis=0) / DT
new_qdd = np.diff(new_qd, axis=0) / DT
new_qddd = np.diff(new_qdd, axis=0) / DT

# %%
new_qd.shape

for i in range(7):
    if np.any(new_qd[:, i] < qd_limits[0, i]) or np.any(new_qd[:, i] > qd_limits[1, i]):
        print(f"qd_{i} exceeded")
    if np.any(new_qdd[:, i] < qdd_limits[0, i]) or np.any(new_qdd[:, i] > qdd_limits[1, i]):
        print(f"qdd_{i} exceeded")
    if np.any(new_qddd[:, i] < qddd_limits[0, i]) or np.any(new_qddd[:, i] > qddd_limits[1, i]):
        print(f"qddd_{i} exceeded")

# %%
new_qd_max = np.absolute(new_qd).max()
new_qdd_max = np.absolute(new_qdd).max()
new_qddd_max = np.absolute(new_qddd).max()
print(f"new_qd_max = {new_qd_max}, new_qdd_max = {new_qdd_max}, new_qddd_max = {new_qddd_max}")

# %%
new_qd[3000] - qd[3000]

NUM_JOINTS = 7
nrows = 7
ncols = 1
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
for i in range(NUM_JOINTS):
    axes[i].plot(q[:, i], label=f"q_{i}")
    axes[i].plot(q_limits[0, i] * np.ones_like(q[:, i]), label=f"q_{i}_min")
    axes[i].plot(q_limits[1, i] * np.ones_like(q[:, i]), label=f"q_{i}_max")
    axes[i].legend()
plt.show()

# %%
NUM_JOINTS = 7
nrows = 7
ncols = 1
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
for i in range(NUM_JOINTS):
    axes[i].plot(new_qd[:, i], label=f"qd_{i}")
    axes[i].plot(qd_limits[0, i] * np.ones_like(new_qd[:, i]), label=f"qd_{i}_min")
    axes[i].plot(qd_limits[1, i] * np.ones_like(new_qd[:, i]), label=f"qd_{i}_max")
    axes[i].legend()
plt.show()

# %%
NUM_JOINTS = 7
nrows = 7



# %%
nrows = 4
ncols = 1
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
axes = axes.flatten()
axes[0].plot(q[:, :], label="q")
axes[1].plot(new_qd[:, :], label="qd")
axes[2].plot(new_qdd[:, :], label="qdd")
axes[3].plot(new_qddd[:, :], label="qddd")
plt.show()


# %%

X_W_H_feasible = np.array(
    [
        [0, 0, 1, 0.4],
        [0, 1, 0, 0.0],
        [-1, 0, 0, 0.15],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
X_W_H_collide_object = np.array(
    [
        [0, 0, 1, 0.65],
        [0, 1, 0, 0.0],
        [-1, 0, 0, 0.15],
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

# %%
q, qd, qdd, dt, result, _ = solve_trajopt(
    X_W_H=X_W_H_feasible,
    q_algr_constraint=q_algr_pre,
    enable_opt=True,
)

# %%
q, qd, qdd, dt, result, _ = solve_trajopt(
    X_W_H=X_W_H,
    q_algr_constraint=q_algr_pre,
    collision_check_object=True,
    obj_filepath=OBJECT_OBJ_PATH,
    obj_xyz=(0.65, 0.0, 0.0),
    obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    collision_check_table=True,
    enable_opt=False,
    enable_graph=True,
    raise_if_fail=False,
    use_cuda_graph=False
)

# %%
