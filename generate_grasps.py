from nerf_grasping.sim import ig_objects
from nerf_grasping import grasp_opt, grasp_utils, mesh_utils
from functools import partial

import os
import scipy.spatial
import torch
import trimesh
import numpy as np


def compute_sampled_grasps(model, grasp_points, centroid):
    rays_o, rays_d = grasp_points[:, :3], grasp_points[:, 3:]
    rays_d = grasp_utils.res_to_true_dirs(rays_o, rays_d, centroid)
    print("optimized vals: ", rays_o)
    if isinstance(model, trimesh.Trimesh):
        rays_o = mesh_utils.correct_z_dists(model, rays_o, rays_d)
    else:
        rays_o = grasp_utils.correct_z_dists(model, grasp_points)
    print("corrected vals:", rays_o, centroid)
    rays_o, rays_d = grasp_utils.nerf_to_ig(rays_o), grasp_utils.nerf_to_ig(rays_d)
    return rays_o, rays_d


def main(
    obj_name="banana",
    use_nerf=False,
    mesh_in=None,
    outfile=None,
    num_grasps=10,
    risk_sensitivity=None,
    dice_grasp=False,
    cost_fn="l1",
):

    object_bounds = grasp_utils.OBJ_BOUNDS

    if obj_name == "banana":
        obj = ig_objects.Banana()
    elif obj_name == "box":
        obj = ig_objects.Box()
    elif obj_name == "teddy_bear":
        obj = ig_objects.TeddyBear()
        obj.use_centroid = True
    elif obj_name == "power_drill":
        obj = ig_objects.PowerDrill()
    elif obj_name == "bleach_cleanser":
        obj = ig_objects.BleachCleanser()

    if use_nerf:
        model = ig_objects.load_nerf(obj.workspace, obj.bound, obj.scale)
        centroid = grasp_utils.get_centroid(model)
        print(f"Estimated Centroid: {centroid}")
        print(f"True Centroid: {obj.gt_mesh.centroid}")
    else:

        if mesh_in is not None:
            obj_mesh = trimesh.load(mesh_in, force="mesh")

        else:
            obj_mesh = obj.gt_mesh

        T = np.eye(4)
        R = scipy.spatial.transform.Rotation.from_euler("Y", [-np.pi / 2]).as_matrix()
        R = (
            R
            @ scipy.spatial.transform.Rotation.from_euler("X", [-np.pi / 2]).as_matrix()
        )
        T[:3, :3] = R
        obj_mesh.apply_transform(T)

        model = obj_mesh
        centroid = torch.from_numpy(obj_mesh.centroid).float()

    if outfile is None:
        if use_nerf:
            outfile = obj_name + "_nerf"
        elif mesh_in is not None:
            outfile = os.path.split(mesh_in)[-1]
            outfile = outfile.split(".")[0]
        else:
            outfile = obj_name

        if dice_grasp:
            outfile += "_diced"

    grasp_points = (
        torch.tensor([[0.09, 0.0, -0.045], [-0.09, 0.0, -0.045], [0, 0.0, 0.09]])
        .reshape(1, 3, 3)
        .to(centroid)
    )

    grasp_points += centroid.reshape(1, 1, 3)
    grasp_dirs = torch.zeros_like(grasp_points)

    mu_0 = torch.cat([grasp_points, grasp_dirs], dim=-1).reshape(-1).to(centroid)
    Sigma_0 = torch.diag(
        torch.cat(
            [torch.tensor([5e-2, 1e-3, 5e-2, 5e-3, 1e-3, 5e-3]) for _ in range(3)]
        )
    ).to(centroid)

    if use_nerf:
        mu_0, Sigma_0 = mu_0.float().cuda(), Sigma_0.float().cuda()
        centroid = centroid.float().cuda()

    # centroid_npy = centroid.detach().cpu().numpy()
    sampled_grasps = np.zeros((num_grasps, 3, 6))
    # max_sample_height = min(2 * centroid_npy[1] - 0.01, 0.05)
    projection_fn = partial(grasp_utils.box_projection, object_bounds=object_bounds)
    num_cem_iters = 15
    grasp_data = []
    for ii in range(num_grasps):
        if dice_grasp:
            rays_o, rays_d = grasp_opt.dice_the_grasp(
                model, cost_fn, centroid=centroid.cpu().numpy()
            )

            rays_o = grasp_utils.nerf_to_ig(torch.from_numpy(rays_o).float().cuda())
            rays_d = grasp_utils.nerf_to_ig(torch.from_numpy(rays_d).float().cuda())

            sampled_grasps[ii, :, :3] = rays_o.cpu()
            sampled_grasps[ii, :, 3:] = rays_d.cpu()

            continue

        print("orig vals: ", mu_0.reshape(3, 6))
        num_samples = 500 if not use_nerf else 500
        mu_f, Sigma_f = mu_0, Sigma_0
        for i in range(3):
            grasp_points, mu_f, Sigma_f = grasp_opt.get_points_cem(
                3,
                model,
                num_iters=num_cem_iters // 3,
                mu_0=mu_f,
                Sigma_0=Sigma_f,
                projection=projection_fn,
                centroid=centroid,
                num_samples=num_samples,
                cost_fn=cost_fn,
                risk_sensitivity=risk_sensitivity,
                return_dec_vars=True,
            )
            grasp_points = grasp_points.reshape(3, 6)
            rays_o, rays_d = compute_sampled_grasps(model, grasp_points, centroid)
            grasp_data.append(
                {
                    "cem_iter": num_cem_iters // 3 * (i + 1),
                    "rays_o": rays_o.cpu().numpy(),
                    "rays_d": rays_d.cpu().numpy(),
                    "mu": mu_f.detach().cpu().numpy(),
                    "Sigma": Sigma_f.detach().cpu().numpy(),
                }
            )

        sampled_grasps[ii, :, :3] = rays_o.cpu().numpy()
        sampled_grasps[ii, :, 3:] = rays_d.cpu().numpy()

    os.makedirs("grasp_data", exist_ok=True)
    print(f"saving to: grasp_data/{outfile}.npy")
    np.save(f"grasp_data/{outfile}.npy", sampled_grasps)
    np.save(f"grasp_data/{outfile}_full.npy", grasp_data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_grasps", "--n", help="number of grasps to sample", default=10, type=int
    )
    parser.add_argument("--obj_name", "--o", help="object to use", default="banana")
    parser.add_argument(
        "--use_nerf",
        "--nerf",
        help="flag to use NeRF to generate grasps",
        action="store_true",
    )
    parser.add_argument("--mesh_in", default=None, type=str)
    parser.add_argument("--outfile", "--out", default=None)
    parser.add_argument("--risk_sensitivity", type=float)
    parser.add_argument("--dice_grasp", action="store_true")
    parser.add_argument("--cost_fn", default="l1", type=str)
    args = parser.parse_args()

    print(args)

    main(**vars(args))
