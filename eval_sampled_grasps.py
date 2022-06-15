from nerf_grasping.sim import ig_objects, ig_viz_utils
from nerf_grasping import grasp_opt, grasp_utils
import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt


grasps = np.load("grasp_data/powerdrill_10.npy")
residual_dirs = False
cost_fn = "l1"
n_f = 3
obj = ig_objects.PowerDrill()
if obj.name == "teddy_bear":
    obj.use_centroid = True
model = ig_objects.load_nerf(obj.workspace, obj.bound, obj.scale)
centroid = grasp_utils.get_centroid(model)

cost = lambda x: grasp_opt.grasp_cost(
    x,
    n_f,
    model,
    residual_dirs=residual_dirs,
    cost_fn=cost_fn,
    centroid=centroid,
    risk_sensitivity=5.0,
)

ax = ig_viz_utils.plot_grasps(
    grasps[:, :, :3],
    grasps[:, :, 3:] * 0.05,
    obj_mesh=obj.gt_mesh.triangles_center,
)
plt.show()

print(cost(torch.tensor(grasps).cuda().float()))
