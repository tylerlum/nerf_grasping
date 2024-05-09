# %%
import numpy as np
import trimesh
from tqdm import tqdm
from nerf_grasping.fr3_algr_ik.ik import solve_ik

# %%
# grasp_config_dict = np.load("/juno/u/tylerlum/Downloads/conditioner/grasp_config_dicts/conditioner_0_9999.npy", allow_pickle=True).item()
# grasp_config_dict = np.load("/juno/u/tylerlum/Downloads/conditioner/grasp_config_dicts/mid_optimization/10/conditioner_0_9999.npy", allow_pickle=True).item()
# mesh = trimesh.load_mesh("/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-05-01_12-05-27/nerf_to_mesh/new_mug/coacd/decomposed.obj")
# grasp_config_dict = np.load("/juno/u/tylerlum/Downloads/conditioner/grasp_config_dicts/mid_optimization/50/conditioner_0_9999.npy", allow_pickle=True).item()

grasp_config_dict = np.load("/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-05-01_14-58-59/optimized_grasp_config_dicts/new_mug_0_9999.npy", allow_pickle=True).item()

# %%
# mesh.centroid
centroid = np.array([0.01965157, -0.00010462, 0.05522743])

# %%
X_W_N = trimesh.transformations.translation_matrix([0.65, 0, 0])
X_O_Oy = trimesh.transformations.rotation_matrix(
    np.pi / 2, [1, 0, 0]
)
X_N_O = trimesh.transformations.translation_matrix(centroid)

# %%
trans = grasp_config_dict['trans']
rot = grasp_config_dict['rot']
joint_angles = grasp_config_dict['joint_angles']

n_grasps = trans.shape[0]
assert trans.shape == (n_grasps, 3)
assert rot.shape == (n_grasps, 3, 3)
assert joint_angles.shape == (n_grasps, 16)
# %%
q_stars = []
for i in tqdm(range(n_grasps)):
    X_Oy_H = np.eye(4)
    X_Oy_H[:3, :3] = rot[i]
    X_Oy_H[:3, 3] = trans[i]

    X_W_H = X_W_N @ X_N_O @ X_O_Oy @ X_Oy_H
    q = joint_angles[i]

    try: 
        q_star = solve_ik(X_W_H, q, position_constraint_tolerance=0.001, angular_constraint_tolerance=0.05)
        print(f"{i}) SUCCESS")
        q_stars.append(q_star)
    except RuntimeError as e:
        print(f"{i}) FAIL")
        q_stars.append(None)

# %%
num_success = len([q_star for q_star in q_stars if q_star is not None])
print(f"num_success / n_grasps = {num_success} / {n_grasps} = {num_success / n_grasps}")
pass_idxs = set([i for i, q_star in enumerate(q_stars) if q_star is not None])

# %%
def is_in_limits(joint_angles: np.ndarray) -> np.ndarray:
    N = joint_angles.shape[0]
    assert joint_angles.shape == (N, 16)

    from nerf_grasping.dexgraspnet_utils.hand_model import HandModel
    from nerf_grasping.dexgraspnet_utils.hand_model_type import (
        HandModelType,
    )
    from nerf_grasping.dexgraspnet_utils.pose_conversion import (
        hand_config_to_pose,
    )

    device = "cuda"
    hand_model_type = HandModelType.ALLEGRO_HAND
    hand_model = HandModel(hand_model_type=hand_model_type, device=device, n_surface_points=1000)

    hand_pose = hand_config_to_pose(trans, rot, joint_angles).to(device)
    hand_model.set_parameters(hand_pose)

    joints_upper = hand_model.joints_upper.detach().cpu().numpy()
    joints_lower = hand_model.joints_lower.detach().cpu().numpy()
    assert joints_upper.shape == (16,)
    assert joints_lower.shape == (16,)

    in_limits = np.all(np.logical_and(joint_angles >= joints_lower[None, ...], joint_angles <= joints_upper[None, ...]), axis=1)
    assert in_limits.shape == (N,)
    return in_limits

is_in_limits(joint_angles)

# %%
# trans_2 = grasp_config_dict_2['trans'][GRASP_IDX]
# rot_2 = grasp_config_dict_2['rot'][GRASP_IDX]
# joint_angles_2 = grasp_config_dict_2['joint_angles'][GRASP_IDX]
# 
# # %%
# trans, trans_2
# 
# # %%
# rot, rot_2
# 
# # %%
# det_rot = np.linalg.det(rot)
# det_rot_2 = np.linalg.det(rot_2)
# print(f"det_rot = {det_rot}, det_rot_2 = {det_rot_2}")
# 
# # %%
# rot @ rot.T
# 
# # %%
# rot_2 @ rot_2.T
# 
# # %%
# joint_angles, joint_angles_2
# 
# # %%
# 
# # %%
# # %%
# joints_upper
# 
# # %%
# joints_lower
# 
# # %%
# joint_angles < joints_lower, joint_angles > joints_upper
# 
# # %%
# joint_angles_2 < joints_lower, joint_angles_2 > joints_upper
# 
# # %%
# 