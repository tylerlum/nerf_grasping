from nerf_grasping.sim import ig_objects
from nerf_grasping import grasp_opt, grasp_utils

import os
import scipy.spatial
import torch
import trimesh
import numpy as np

def main(
    obj_name="banana",
    use_nerf=False,
    mesh_in=None,
    min_finger_height=0.01,
    max_finger_dist=0.2,
    outfile=None,
    num_grasps=50,
    risk_sensitivity=None,
):
    if use_nerf:
        if obj_name == "banana":
            obj = ig_objects.Banana
        elif obj_name == "box":
            obj = ig_objects.Box
        elif obj_name == "teddy_bear":
            obj = ig_objects.TeddyBear
            obj.use_centroid = True
        elif obj_name == "powerdrill":
            obj = ig_objects.PowerDrill

        model = ig_objects.load_nerf(obj.workspace, obj.bound, obj.scale)
        centroid = grasp_utils.get_centroid(model)

    else:
        if mesh_in is not None:
            obj_mesh = mesh_in
        elif obj_name == "banana":
            obj_mesh = "assets/objects/meshes/banana/textured.obj"
        elif obj_name == "box":
            obj_mesh = "assets/objects/meshes/cube_multicolor.obj"
        elif obj_name == "teddy_bear":
            obj_mesh = "assets/objects/meshes/isaac_teddy/isaac_bear.obj"
        elif obj_name == "powerdrill":
            obj_mesh = "assets/objects/meshes/power_drill/textured.obj"

        gt_mesh = trimesh.load(obj_mesh, force="mesh")
        T = np.eye(4)
        R = scipy.spatial.transform.Rotation.from_euler('Y', [-np.pi / 2]).as_matrix()
        R = R @ scipy.spatial.transform.Rotation.from_euler('X',
                                                        [-np.pi / 2]).as_matrix()
        T[:3, :3] = R
        gt_mesh.apply_transform(T)

        model = gt_mesh
        centroid = torch.from_numpy(gt_mesh.centroid)

    if outfile is None:
        if use_nerf:
            outfile = obj_name + '_nerf'
        elif mesh_in is not None:
            outfile = os.path.split(mesh_in)[-1]
            outfile = outfile.split('.')[0]
        else:
            outfile = obj_name

    grasp_points = torch.tensor([[0.09, 0.0, -0.025], [-0.09, 0.0, -0.025],
                                 [0, 0.0, 0.09]]).reshape(1, 3, 3).to(centroid)

    grasp_points += centroid.reshape(1,1,3)
    grasp_dirs = torch.zeros_like(grasp_points)

    mu_0 = torch.cat([grasp_points, grasp_dirs], dim=-1).reshape(-1)
    Sigma_0 = torch.diag(
        torch.cat([
            torch.tensor([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2])
            for _ in range(3)])).to(centroid)

    if use_nerf:
        mu_0, Sigma_0 = mu_0.float().cuda(), Sigma_0.float().cuda()
        centroid = centroid.float().cuda()

    def constraint(x):
        """Ensures that all points fall within reasonable range"""
        return torch.logical_and(
            (x.reshape(-1, 3, 6)[..., :3].abs() <= max_finger_dist).all(-1).all(-1),
            (x.reshape(-1, 3, 6)[..., 1] >= min_finger_height).all(-1),
        )

    sampled_grasps = np.zeros((num_grasps, 3, 6))

    for ii in range(num_grasps):
        grasp_points = grasp_opt.get_points_cem(3, model, num_iters=10, mu_0=mu_0, Sigma_0=Sigma_0, constraint=constraint, centroid=centroid)
        # mu_np = grasp_points.cpu().detach().reshape(3, 6)]
        grasp_points = grasp_points.reshape(3,6)
        rays_o, rays_d = grasp_points[:, :3], grasp_points[:, 3:]

        rays_d = grasp_utils.res_to_true_dirs(
                rays_o, rays_d, centroid
            )

        rays_o, rays_d = grasp_utils.nerf_to_ig(rays_o), grasp_utils.nerf_to_ig(rays_d)

        sampled_grasps[ii, :, :3] = rays_o.cpu().numpy()
        sampled_grasps[ii, :, 3:] = rays_d.cpu().numpy()

    os.makedirs('grasp_data', exist_ok=True)
    np.save('grasp_data/' + outfile + '.npy', sampled_grasps)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_grasps", "--n", help="number of grasps to sample", default=50, type=int)
    parser.add_argument("--obj_name", "--o", help="object to use", default="banana")
    parser.add_argument("--use_nerf", "--nerf", help="flag to use NeRF to generate grasps", action="store_true")
    parser.add_argument("--mesh_in", default=None, type=str)
    parser.add_argument("--min_finger_height", default=-0.01, type=float)
    parser.add_argument("--max_finger_dist", default=0.15, type=float)
    parser.add_argument("--outfile", "--out", default=None)
    parser.add_argument("--risk_sensitivity", default=None)
    args = parser.parse_args()

    print(args)

    main(**vars(args))