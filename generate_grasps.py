from nerf_grasping.sim import ig_objects
from nerf_grasping import config, grasp_opt, grasp_utils, mesh_utils, nerf_utils
from functools import partial

import dcargs
import os
import scipy.spatial
import torch
import trimesh
import numpy as np
import yaml


def compute_sampled_grasps(model, grasp_points, centroid):
    rays_o, rays_d = grasp_points[:, :3], grasp_points[:, 3:]
    rays_d = grasp_utils.res_to_true_dirs(rays_o, rays_d, centroid)
    print("optimized vals: ", rays_o)
    if isinstance(model, trimesh.Trimesh):
        rays_o = mesh_utils.correct_z_dists(
            model, rays_o, rays_d, exp_config.model_config
        )
    else:
        rays_o = nerf_utils.correct_z_dists(
            model, grasp_points, exp_config.model_config
        )
    print("corrected vals:", rays_o, centroid)
    rays_o, rays_d = grasp_utils.nerf_to_ig(rays_o), grasp_utils.nerf_to_ig(rays_d)
    return rays_o, rays_d


def main(exp_config: config.Experiment):

    object_bounds = grasp_utils.OBJ_BOUNDS

    obj = ig_objects.load_object(exp_config)

    if isinstance(exp_config.model_config, config.NeRF):
        model = ig_objects.load_nerf(obj.workspace, obj.bound, obj.scale)
        print(f"Estimated Centroid: {model.centroid}")
        print(f"True Centroid: {obj.gt_mesh.centroid}")

        centroid = model.centroid

    else:

        # Load triangle mesh from file.
        obj_mesh = trimesh.load(config.mesh_file(exp_config), force="mesh")

        # Transform triangle mesh to NeRF frame.
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

    outfile = config.grasp_file(exp_config)

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

    if isinstance(exp_config.model_config, config.NeRF):
        mu_0, Sigma_0 = mu_0.float().cuda(), Sigma_0.float().cuda()
        centroid = centroid.float().cuda()

    # centroid_npy = centroid.detach().cpu().numpy()
    sampled_grasps = np.zeros((exp_config.num_grasps, 3, 6))
    # max_sample_height = min(2 * centroid_npy[1] - 0.01, 0.05)
    projection_fn = partial(grasp_utils.box_projection, object_bounds=object_bounds)

    cost_function = grasp_opt.get_cost_function(exp_config, model)

    for ii in range(exp_config.num_grasps):
        if exp_config.dice_grasp:
            rays_o, rays_d = grasp_opt.dice_the_grasp(
                exp_config, model
            )  # TODO(pculbert): fix this function trace.

            rays_o = grasp_utils.nerf_to_ig(torch.from_numpy(rays_o).float().cuda())
            rays_d = grasp_utils.nerf_to_ig(torch.from_numpy(rays_d).float().cuda())

            sampled_grasps[ii, :, :3] = rays_o.cpu()
            sampled_grasps[ii, :, 3:] = rays_d.cpu()

        else:

            mu_f, Sigma_f, cost_history, best_point = grasp_opt.optimize_cem(
                cost_function,
                mu_0,
                Sigma_0,
                num_iters=exp_config.cem_num_iters,
                num_samples=exp_config.cem_num_samples,
                elite_frac=exp_config.cem_elite_frac,
                projection=projection_fn,
            )

            grasp_points = best_point.reshape(3, 6)
            rays_o, rays_d = compute_sampled_grasps(model, grasp_points, centroid)

        sampled_grasps[ii, :, :3] = rays_o.cpu().numpy()
        sampled_grasps[ii, :, 3:] = rays_d.cpu().numpy()

    os.makedirs("grasp_data", exist_ok=True)
    print(f"saving to: {outfile}[.npy, .yaml]")
    np.save(f"{outfile}.npy", sampled_grasps)
    config.save(exp_config)


exp_config = dcargs.cli(config.Experiment)
main(exp_config)
