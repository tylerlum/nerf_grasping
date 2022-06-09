from nerf_grasping import grasp_opt, grasp_utils
import os
import scipy.spatial
import torch
import trimesh
import numpy as np

def main(
    obj_name="banana",
    min_finger_height=-0.01,
    max_finger_dist=0.15,
    outfile=None,
    num_grasps=50,
):
    if obj_name == "banana":
        obj_mesh = "assets/objects/meshes/banana/textured.obj"
    elif obj_name == "box":
        obj_mesh = "assets/objects/meshes/cube_multicolor.obj"
    elif obj_name == "teddy_bear":
        obj_mesh = "assets/objects/meshes/isaac_teddy/isaac_bear.obj"
    elif obj_name == "powerdrill":
        obj_mesh = "assets/objects/meshes/power_drill/textured.obj"

    if outfile is None:
        outfile = obj_name

    gt_mesh = trimesh.load(obj_mesh, force="mesh")
    T = np.eye(4)
    R = scipy.spatial.transform.Rotation.from_euler('Y', [-np.pi / 2]).as_matrix()
    R = R @ scipy.spatial.transform.Rotation.from_euler('X',
                                                    [-np.pi / 2]).as_matrix()
    T[:3, :3] = R
    gt_mesh.apply_transform(T)

    grasp_points = torch.tensor([[0.09, 0.0, -0.025], [-0.09, 0.0, -0.025],
                                 [0, 0.0, 0.09]]).reshape(1, 3, 3)
    grasp_dirs = torch.zeros_like(grasp_points)

    mu_0 = torch.cat([grasp_points, grasp_dirs], dim=-1).reshape(-1)
    Sigma_0 = torch.diag(
        torch.cat([
            torch.tensor([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2])
            for _ in range(3)]))

    def constraint(x):
        """Ensures that all points fall within reasonable range"""
        return torch.logical_and(
            (x.reshape(-1, 3, 6)[..., :3].abs() <= max_finger_dist).all(-1).all(-1),
            (x.reshape(-1, 3, 6)[..., 1] >= min_finger_height).all(-1),
        )

    sampled_grasps = np.zeros((num_grasps, 3, 6))

    for ii in range(num_grasps):
        grasp_points = grasp_opt.get_points_cem(3, gt_mesh, num_iters=25, mu_0=mu_0, Sigma_0=Sigma_0, constraint=constraint)
        mu_np = grasp_points.cpu().detach().reshape(3, 6)
        rays_o, rays_d = mu_np[:, :3], mu_np[:, 3:]

        rays_d = grasp_utils.res_to_true_dirs(
                rays_o, rays_d, torch.from_numpy(gt_mesh.centroid).to(rays_o)
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
    parser.add_argument("--min_finger_height", default=-0.01, type=float)
    parser.add_argument("--max_finger_dist", default=0.15, type=float)
    parser.add_argument("--outfile", "--out", default=None)
    args = parser.parse_args()

    print(args)

    main(**vars(args))