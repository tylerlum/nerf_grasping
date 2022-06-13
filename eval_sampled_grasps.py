from nerf_grasping.sim import ig_objects
from nerf_grasping import grasp_opt, grasp_utils
import torch
import numpy as np

grasps = np.load("grasp_data/banana_nerf10psv-rs10.npy")
residual_dirs = False
cost_fn = "psv"
n_f = 3
obj = ig_objects.Banana
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

for x in grasps:
    x = torch.tensor(x).cuda().float()
    print(cost(x))
