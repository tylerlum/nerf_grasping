from nerf_grasping.sim import ig_objects, ig_viz_utils
from nerf_grasping import grasp_opt, grasp_utils
import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt


def get_mesh(obj_name):
    if obj_name == "banana":
        obj_mesh = "assets/objects/meshes/banana/textured.obj"
    elif obj_name == "box":
        obj_mesh = "assets/objects/meshes/cube_multicolor.obj"
    elif obj_name == "teddy_bear":
        obj_mesh = "assets/objects/meshes/isaac_teddy/isaac_bear.obj"
    elif obj_name == "powerdrill":
        obj_mesh = "assets/objects/meshes/power_drill/textured.obj"

    gt_mesh = trimesh.load(obj_mesh, force="mesh")
    # T = np.eye(4)
    # R = scipy.spatial.transform.Rotation.from_euler("Y", [-np.pi / 2]).as_matrix()
    # R = R @ scipy.spatial.transform.Rotation.from_euler("X", [-np.pi / 2]).as_matrix()
    # T[:3, :3] = R
    return gt_mesh


grasps = np.load("grasp_data/banana_nerf_intersect.npy")
residual_dirs = False
cost_fn = "psv"
n_f = 3
obj = ig_objects.Banana
gt_mesh = get_mesh(obj.name)
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
    grasps[:, :, 3:] * 0.1,
    obj_mesh=gt_mesh.triangles_center,
)
plt.show()


for x in grasps:
    x = torch.tensor(x).cuda().float()
    print(cost(x))
