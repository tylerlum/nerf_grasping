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
    """Converts grasp vars to ray origins/directions; attempts to clip
    grasp to lie above floor + be equidistant from the surface."""

    print("optimized vals: ", grasp_points)
    if isinstance(model, trimesh.Trimesh):
        rays_o, rays_d = grasp_points[:, :3], grasp_points[:, 3:]
        rays_d = grasp_utils.res_to_true_dirs(rays_o, rays_d, centroid)
        rays_o = mesh_utils.correct_z_dists(
            model, rays_o, rays_d, exp_config.model_config
        )
    else:
        rays_o, rays_d = nerf_utils.correct_z_dists(
            model, grasp_points, exp_config.model_config
        )
    print("corrected vals:", rays_o, rays_d, centroid)
    rays_o, rays_d = grasp_utils.nerf_to_ig(rays_o), grasp_utils.nerf_to_ig(rays_d)
    return rays_o, rays_d


def main(exp_config: config.Experiment):

    obj = ig_objects.load_object(exp_config)

    if isinstance(exp_config.model_config, config.Nerf):
        model = ig_objects.load_nerf(
            obj.workspace, obj.bound, obj.scale, obj.new_translation
        )
        print(f"Estimated Centroid: {model.centroid}")
        print(f"True Centroid: {obj.gt_mesh.nerf_centroid}")

        centroid = model.centroid

    else:

        model = mesh_utils.get_mesh(exp_config, obj)

        # Transform triangle mesh to Nerf frame.
        T = np.eye(4)
        R = scipy.spatial.transform.Rotation.from_euler("Y", [-np.pi / 2]).as_matrix()
        R = (
            R
            @ scipy.spatial.transform.Rotation.from_euler("X", [-np.pi / 2]).as_matrix()
        )
        T[:3, :3] = R
        model.apply_transform(T)
        centroid = torch.from_numpy(model.centroid).float()
        model.ig_centroid = model.centroid

    grasp_points = (
        torch.tensor([[0.09, 0.0, -0.045], [-0.09, 0.0, -0.045], [0, 0.0, 0.09]])
        .reshape(1, 3, 3)
        .to(centroid)
    )

    grasp_points += centroid.reshape(1, 1, 3)  # move grasp points in object frame
    grasp_dirs = torch.zeros_like(grasp_points)

    mu_0 = torch.cat([grasp_points, grasp_dirs], dim=-1).reshape(-1).to(centroid)
    Sigma_0 = torch.diag(
        torch.cat(
            [torch.tensor([1e-2, 1e-2, 1e-2, 5e-3, 5e-3, 5e-3]) for _ in range(3)]
        )
    ).to(centroid)

    if isinstance(exp_config.model_config, config.Nerf):
        mu_0, Sigma_0 = mu_0.float().cuda(), Sigma_0.float().cuda()
        centroid = centroid.float().cuda()

    sampled_grasps = np.zeros((exp_config.num_grasps, 3, 6))
    object_bounds = grasp_utils.OBJ_BOUNDS
    projection_fn = partial(grasp_utils.box_projection, object_bounds=object_bounds)

    cost_function = grasp_opt.get_cost_function(exp_config, model)

    for ii in range(exp_config.num_grasps):
        if exp_config.dice_grasp:

            assert isinstance(exp_config.model_config, config.Mesh)

            rays_o, rays_d = grasp_opt.dice_the_grasp(
                model, cost_function, exp_config, projection_fn
            )

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

    outfile = config.grasp_file(exp_config)

    os.makedirs("grasp_data", exist_ok=True)
    print(f"saving to: {outfile}[.npy, .yaml]")
    np.save(f"{outfile}.npy", sampled_grasps)
    config.save(exp_config, outfile)


if __name__ == "__main__":
    exp_config = dcargs.cli(config.Experiment)
    main(exp_config)
