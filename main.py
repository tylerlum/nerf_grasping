# Imports and setup.
import grasp_opt
import grasp_utils
import mesh_utils
import numpy as np
import os
import pypoisson
import torch
import trimesh

from matplotlib import pyplot as plt
from nerf_shared import config_parser, utils

# nerf_shared requires us to default to cuda tensors.
torch.set_default_tensor_type(torch.cuda.FloatTensor)

if __name__ == '__main__':

    # Setup config (since we're not running from command line).
    parser = config_parser.config_parser()

    # Fix some pathing since config, etc. is typically relative to the submodule.
    args = parser.parse_args()
    basedir = os.path.join(*args.config.split(os.sep)[:-2])
    args.basedir = os.path.join(basedir, args.basedir)
    args.datadir = os.path.join(basedir, args.datadir)

    # Load nerf models, params from checkpoint.
    coarse_model, fine_model = utils.create_nerf_models(args)
    optimizer = utils.get_optimizer(coarse_model, fine_model, args)
    utils.load_checkpoint(coarse_model, fine_model, optimizer,
                          args, checkpoint_index=-1)

    images, poses, render_poses, hwf, i_split, K, bds_dict = utils.load_datasets(args)
    bds_dict = {'near': 0.05, 'far': 0.15} # Override rendering bounds for fingers.
    renderer = utils.get_renderer(args, bds_dict)
    H, W, focal = hwf

    # Setup params for grasp optimization.
    num_grasps = 10
    n_f = 3

    # Initialize points more-or-less spread apart.
    grasp_points = torch.tensor([[0.09,0.09,-0.025],
                                 [-0.09, 0.09, 0.025],
                                 [0, -0.125, 0]]).reshape(1, 3, 3)
    grasp_dirs = -grasp_points

    # Convert these to mean/covariance for CEM.
    mu_0 = torch.cat([grasp_points, grasp_dirs], dim=-1).reshape(-1)
    Sigma_0 = torch.diag(torch.cat(
        [torch.tensor([5e-3, 5e-3, 5e-3, 1e-3, 1e-3, 1e-3]) for _ in range(3)]))

    # Setup CEM cost.
    cem_cost = lambda x: grasp_opt.grasp_cost(x, n_f, coarse_model,
                                              fine_model, renderer)

    # Optimize points with CEM.
    with torch.no_grad():
        mu_f, Sigma_f = grasp_opt.optimize_cem(cem_cost, mu_0, Sigma_0, num_iters=25,
                                               elite_frac=.1, num_samples=250)

    gps_numpy = mu_f.cpu().detach().numpy().reshape(3, 6)
    grasp_starts = gps_numpy[:, :3]
    grasp_dirs = gps_numpy[:, 3:]

    print(grasp_starts, grasp_dirs)

    # TODO(pculbert): Return these somehow lol?
