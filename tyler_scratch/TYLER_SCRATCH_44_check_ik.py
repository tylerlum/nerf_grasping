# %%
import numpy as np
import trimesh
from tqdm import tqdm
from nerf_grasping.fr3_algr_ik.ik import solve_ik

# %%
grasp_config_dict = np.load("/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-05-01_12-05-27/optimized_grasp_config_dicts/new_mug_0_9999.npy", allow_pickle=True).item()
mesh = trimesh.load_mesh("/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-05-01_12-05-27/nerf_to_mesh/new_mug/coacd/decomposed.obj")

# %%
mesh.centroid

# %%
X_W_N = trimesh.transformations.translation_matrix([0.65, 0, 0])
X_O_Oy = trimesh.transformations.rotation_matrix(
    np.pi / 2, [1, 0, 0]
)
X_N_O = trimesh.transformations.translation_matrix(mesh.centroid)

# %%
GRASP_IDX = 0
trans = grasp_config_dict['trans']
rot = grasp_config_dict['rot']
joint_angles = grasp_config_dict['joint_angles']

n_grasps = trans.shape[0]
assert GRASP_IDX < n_grasps
assert trans.shape == (n_grasps, 3)
assert rot.shape == (n_grasps, 3, 3)
assert joint_angles.shape == (n_grasps, 16)
# %%
results = []
for i in tqdm(range(n_grasps)):
    X_Oy_H = np.eye(4)
    X_Oy_H[:3, :3] = rot[i]
    X_Oy_H[:3, 3] = trans[i]

    X_W_H = X_W_N @ X_N_O @ X_O_Oy @ X_Oy_H
    q = joint_angles[i]

    try: 
        q_star = solve_ik(X_W_H, q)
        results.append((i, True))
        print("SUCCESS")
    except RuntimeError as e:
        results.append((i, False))
        print("FAIL")

# %%
total = len(results)
num_success = sum([r[-1] for r in results])
print(f"num_success / total = {num_success} / {total} = {num_success / total}")
pass_idxs = set([r[0] for r in results if r[-1]])

# %%
