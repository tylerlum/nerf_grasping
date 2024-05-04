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
# SELECTED_IDX = 3
SELECTED_IDX = BEST_IDX

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
# HACK
# X_W_H[0, 3] -= 0.05

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
# d_world, d_self = max_penetration_from_X_W_H(
#     X_W_H=X_W_H,
#     q_algr_constraint=q_algr_pre,
#     include_object=True,
#     obj_filepath=OBJECT_OBJ_PATH,
#     obj_xyz=(0.65, 0.0, 0.0),
#     obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
#     include_table=True,
# )
# print(f"d_world = {d_world}, d_self = {d_self}")
# if d_world.item() > 0.0:
#     print("WARNING: penetration with world detected")
# if d_self.item() > 0.0:
#     print("WARNING: self collision detected")
# 
# d_world, d_self = max_penetration_from_X_W_H(
#     X_W_H=X_W_H,
#     q_algr_constraint=q_algr_pre,
#     include_object=False,
#     obj_filepath=OBJECT_OBJ_PATH,
#     obj_xyz=(0.65, 0.0, 0.0),
#     obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
#     include_table=True,
# )
# print(f"Without object: d_world = {d_world}, d_self = {d_self}")
# 
# d_world, d_self = max_penetration_from_X_W_H(
#     X_W_H=X_W_H,
#     q_algr_constraint=q_algr_pre,
#     include_object=True,
#     obj_filepath=OBJECT_OBJ_PATH,
#     obj_xyz=(0.65, 0.0, 0.0),
#     obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
#     include_table=False,
# )
# print(f"Without table: d_world = {d_world}, d_self = {d_self}")
# 
# d_world, d_self = max_penetration_from_X_W_H(
#     X_W_H=X_W_H,
#     q_algr_constraint=q_algr_pre,
#     include_object=False,
#     obj_filepath=OBJECT_OBJ_PATH,
#     obj_xyz=(0.65, 0.0, 0.0),
#     obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
#     include_table=False,
# )
# print(f"Without object or table: d_world = {d_world}, d_self = {d_self}")
# 
# open_hand_q_algr = q_algr_pre.copy()
# DELTA = 0.1
# open_hand_q_algr[1] -= DELTA
# open_hand_q_algr[2] -= DELTA
# open_hand_q_algr[3] -= DELTA
# 
# open_hand_q_algr[5] -= DELTA
# open_hand_q_algr[6] -= DELTA
# open_hand_q_algr[7] -= DELTA
# 
# open_hand_q_algr[9] -= DELTA
# open_hand_q_algr[10] -= DELTA
# open_hand_q_algr[11] -= DELTA
# d_world, d_self = max_penetration_from_X_W_H(
#     X_W_H=X_W_H,
#     q_algr_constraint=open_hand_q_algr,
#     include_object=True,
#     obj_filepath=OBJECT_OBJ_PATH,
#     obj_xyz=(0.65, 0.0, 0.0),
#     obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
#     include_table=True,
# )
# print(f"DELTA = {DELTA}, d_world = {d_world}, d_self = {d_self}")
# 
# open_hand_q_algr = q_algr_pre.copy()
# DELTA = 0.2
# open_hand_q_algr[1] -= DELTA
# open_hand_q_algr[2] -= DELTA
# open_hand_q_algr[3] -= DELTA
# 
# open_hand_q_algr[5] -= DELTA
# open_hand_q_algr[6] -= DELTA
# open_hand_q_algr[7] -= DELTA
# 
# open_hand_q_algr[9] -= DELTA
# open_hand_q_algr[10] -= DELTA
# open_hand_q_algr[11] -= DELTA
# d_world, d_self = max_penetration_from_X_W_H(
#     X_W_H=X_W_H,
#     q_algr_constraint=open_hand_q_algr,
#     include_object=True,
#     obj_filepath=OBJECT_OBJ_PATH,
#     obj_xyz=(0.65, 0.0, 0.0),
#     obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
#     include_table=True,
# )
# print(f"DELTA = {DELTA}, d_world = {d_world}, d_self = {d_self}")
# 
# open_hand_q_algr = q_algr_pre.copy()
# DELTA = 0.3
# open_hand_q_algr[1] -= DELTA
# open_hand_q_algr[2] -= DELTA
# open_hand_q_algr[3] -= DELTA
# 
# open_hand_q_algr[5] -= DELTA
# open_hand_q_algr[6] -= DELTA
# open_hand_q_algr[7] -= DELTA
# 
# open_hand_q_algr[9] -= DELTA
# open_hand_q_algr[10] -= DELTA
# open_hand_q_algr[11] -= DELTA
# d_world, d_self = max_penetration_from_X_W_H(
#     X_W_H=X_W_H,
#     q_algr_constraint=open_hand_q_algr,
#     include_object=True,
#     obj_filepath=OBJECT_OBJ_PATH,
#     obj_xyz=(0.65, 0.0, 0.0),
#     obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
#     include_table=True,
# )
# print(f"DELTA = {DELTA}, d_world = {d_world}, d_self = {d_self}")
# 

# %%
failed = False
try:
    print("=" * 80)
    print("Trying with full object collision check")
    print("=" * 80 + "\n")
    q, qd, qdd, dt, _ = solve_trajopt(
        X_W_H=X_W_H,
        q_algr_constraint=q_algr_pre,
        collision_check_object=True,
        obj_filepath=OBJECT_OBJ_PATH,
        obj_xyz=(0.65, 0.0, 0.0),
        obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        collision_check_table=True,
    )
    print("SUCCESS TRAJOPT with full object collision check")
except RuntimeError as e:
    print(f"FAILED TRAJOPT: {e} with full object collision check")
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
        q, qd, qdd, dt, _ = solve_trajopt(
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
        q, qd, qdd, dt, _ = solve_trajopt(
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
        q, qd, qdd, dt, _ = solve_trajopt(
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

print(f"q.shape = {q.shape}, qd.shape = {qd.shape}, qdd.shape = {qdd.shape}, dt = {dt}")
N_pts = q.shape[0]
assert q.shape == (N_pts, 23)

# %%
remove_collision_spheres()
for i in tqdm(range(N_pts)):
    position = q[i]
    assert position.shape == (23,)
    # print(f"{i} / {N_pts} {position}")

    for i, joint_idx in enumerate(arm_actuatable_joint_idxs):
        pb.resetJointState(r, joint_idx, position[i])
    for i, joint_idx in enumerate(hand_actuatable_joint_idxs):
        pb.resetJointState(r, joint_idx, position[i + 7])
    time.sleep(0.01)

# %%
draw_collision_spheres(
    robot=r,
    config=collision_config,
)

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

    # d_world, d_self = max_penetration_from_X_W_H(
    #     X_W_H=X_W_H,
    #     q_algr_constraint=q_algr_pre,
    #     include_object=True,
    #     obj_filepath=OBJECT_OBJ_PATH,
    #     obj_xyz=(0.65, 0.0, 0.0),
    #     obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    #     include_table=True,
    # )
    # if d_world.item() == 0.0 and d_self.item() == 0.0:
    #     no_collision_idxs.append(i)

    try:
        print("=" * 80)
        print("Trying with full object collision check")
        print("=" * 80 + "\n")
        q, qd, qdd, dt, _ = solve_trajopt(
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
        q, qd, qdd, dt, _ = solve_trajopt(
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
