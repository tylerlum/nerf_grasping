"""
Module implementing utils for grasping,
including normal estimation and surface detection.
"""
import logging
import math

import lietorch
import numpy as np
import scipy
import torch
from nerf import renderer, utils
from nerf_grasping import config, grasp_utils

# GRAD_CONFIG = config.grad_configs["grasp_opt"]


class NeRFModel:
    # https://stackoverflow.com/questions/68926132/creation-of-a-class-wrapper-in-python
    def __init__(self, base_model, obj_translation):
        self.base_model = base_model
        # self._centroid = get_centroid(base_model)
        self._centroid = grasp_utils.ig_to_nerf(obj_translation.reshape(1, 3)).reshape(
            -1
        )
        self._obj_translation = obj_translation

    def __getattr__(self, name):
        return getattr(self.base_model, name)

    def __call__(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    @property
    def centroid(self):
        """Approximate NeRF model centroid"""
        # TODO: decide whether to return centroid (from nerf_centroid),
        #       or obj translation centroid
        # never returns direct pointer to attribute, prevents overwriting
        return self._centroid.clone()
        # return self.nerf_centroid

    @property
    def ig_centroid(self):
        """IG centroid when object is loaded into simulator, in IG frame"""
        # grasp_utils.nerf_to_ig(self.centroid.reshape(1, 3), return_tensor=False)
        return self._obj_translation

    @property
    def nerf_centroid(self):
        """IG centroid (when object is loaded into sim) in Nerf frame"""
        return grasp_utils.ig_to_nerf(self._obj_translation.reshape(1, 3)).reshape(-1)


def load_nerf(opt, obj_translation):
    """
    Utility function that loads a torch-ngp style NeRF using a config Dict.

    Args:
        opt: A Dict of configuration params (must contain the params from the
            argparser in nerf.utils.get_config_parser()).

    Returns a torch-ngp.nerf.NeRFNetwork which can be used to render grasp points.
    """

    # Use options to determine proper network structure.
    if opt.ff:
        assert opt.fp16, "fully-fused mode must be used with fp16 mode"
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        assert opt.fp16, "tcnn mode must be used with fp16 mode"
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork

    # Create uninitialized network.
    model = NeRFNetwork(
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
    )

    # Create trainer with NeRF; use its constructor to load network weights from file.
    trainer = utils.Trainer(
        "ngp",
        vars(opt),
        model,
        workspace=opt.workspace,
        criterion=None,
        fp16=opt.fp16,
        metrics=[None],
        use_checkpoint="latest",
    )
    assert len(trainer.stats["checkpoints"]) != 0, "failed to load checkpoint"
    model = NeRFModel(trainer.model, obj_translation)
    return model


def get_grasp_distribution(grasp_vars, model, nerf_config):
    """
    Generates a "grasp distribution," a set of n_f categorical distributions
    for where each finger will contact the object surface, along with the associated
    contact points.
    Args:
        grasp_vars: tensor, shape [B, n_f, 6], of grasp positions ([..., :3]) and
            approach directions ([..., 3:]) to be used for rendering. Approach
            directions need not be normalized.
        model: torch_ngp NeRFNetwork that represents the scene.
        nerf_config: config.Nerf object holding the nerf configuration.

    Returns a tuple (points, probs) defining the grasp distribution.
    """
    # TODO(pculbert): Support grasp_vars without a leading batch dim.
    B, n_f, _ = grasp_vars.shape
    assert grasp_vars.shape[-1] == 6

    # Create ray batch
    rays_o, rays_d = grasp_vars[..., :3], grasp_vars[..., 3:]  # [B, n_f, 3] x 2.
    rays_o = torch.reshape(rays_o, [-1, 3]).float()  # [B * n_f, 3]

    N = rays_o.shape[0]
    device = rays_o.device
    rays_d = torch.reshape(rays_d, [-1, 3]).float()  # [B * n_f, 3]

    # Convert dirs from residual to NeRF frame.
    rays_d = grasp_utils.res_to_true_dirs(
        rays_o, rays_d, model.centroid
    )  # [B * n_f, 3]

    # Get near/far bounds for each ray.
    # First, generate near/far bounds for cube.
    near_cube, far_cube = renderer.near_far_from_bound(
        rays_o, rays_d, model.bound, type="cube"
    )

    # Then setup near/far bounds for fingertip renderer.
    near_finger = nerf_config.render_near_bound * torch.ones_like(near_cube)
    far_finger = nerf_config.render_far_bound * torch.ones_like(far_cube)

    # Keep mask of rays which miss the cube altogether.
    mask = near_cube > 1e8

    # Then take near = max(near_finger, near_cube), far = min(far_finger, far_cube).
    near = torch.maximum(near_finger, near_cube)
    far = torch.minimum(far_finger, far_cube)
    far[mask] = 1e9

    z_vals = torch.linspace(0.0, 1.0, nerf_config.num_steps, device=device).unsqueeze(
        0
    )  # [1, T]
    z_vals = z_vals.expand((N, nerf_config.num_steps))  # [N, T]
    z_vals = near + (far - near) * z_vals  # [N, T], in [near, far]

    # perturb z_vals
    sample_dist = (far - near) / nerf_config.num_steps
    if nerf_config.render_perturb_samples:
        z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist

    # generate pts
    pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(
        -1
    )  # [N, 1, 3] * [N, T, 3] -> [N, T, 3]
    pts = pts.clamp(
        -model.bound, model.bound
    )  # must be strictly inside the bounds, else lead to nan in hashgrid encoder!

    # query SDF and RGB
    dirs = rays_d.unsqueeze(-2).expand_as(pts)

    sigmas, rgbs = model(pts.reshape(-1, 3), dirs.reshape(-1, 3))

    rgbs = rgbs.reshape(N, nerf_config.num_steps, 3)  # [N, T, 3]
    sigmas = sigmas.reshape(N, nerf_config.num_steps)  # [N, T]

    # upsample z_vals (nerf-like)
    if nerf_config.upsample_steps > 0:
        with torch.no_grad():

            deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T-1]
            deltas = torch.cat(
                [deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1
            )

            alphas = 1 - torch.exp(-deltas * sigmas)  # [N, T]
            alphas_shifted = torch.cat(
                [torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1
            )  # [N, T+1]
            weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]  # [N, T]

            # sample new z_vals
            z_vals_mid = z_vals[..., :-1] + 0.5 * deltas[..., :-1]  # [N, T-1]
            new_z_vals = renderer.sample_pdf(
                z_vals_mid,
                weights[:, 1:-1],
                nerf_config.upsample_steps,
                det=not model.training,
            ).detach()  # [N, t]

            new_pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(
                -2
            ) * new_z_vals.unsqueeze(
                -1
            )  # [N, 1, 3] * [N, t, 3] -> [N, t, 3]
            new_pts = new_pts.clamp(-model.bound, model.bound)

        # only forward new points to save computation
        new_dirs = rays_d.unsqueeze(-2).expand_as(new_pts)
        new_sigmas, new_rgbs = model(new_pts.reshape(-1, 3), new_dirs.reshape(-1, 3))
        new_rgbs = new_rgbs.reshape(N, nerf_config.upsample_steps, 3)  # [N, t, 3]
        new_sigmas = new_sigmas.reshape(N, nerf_config.upsample_steps)  # [N, t]

        # re-order
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)  # [N, T+t]
        z_vals, z_index = torch.sort(z_vals, dim=-1)

        sigmas = torch.cat([sigmas, new_sigmas], dim=-1)  # [N, T+t]
        sigmas = torch.gather(sigmas, dim=-1, index=z_index)

        rgbs = torch.cat([rgbs, new_rgbs], dim=-2)  # [N, T+t, 3]
        rgbs = torch.gather(rgbs, dim=-2, index=z_index.unsqueeze(-1).expand_as(rgbs))

    # render core
    deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T-1]
    deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

    alphas = 1 - torch.exp(-deltas * sigmas)  # [N, T]
    alphas_shifted = torch.cat(
        [torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1
    )  # [N, T+1]
    weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]  # [N, T]

    rays_o = rays_o.reshape(B, n_f, 3)
    rays_d = rays_d.reshape(B, n_f, 3)
    total_num_steps = nerf_config.num_steps + nerf_config.upsample_steps
    z_vals = z_vals.reshape(B, n_f, total_num_steps)
    weights = weights.reshape(B, n_f, total_num_steps)

    return (
        rays_o,
        rays_d,
        weights,
        z_vals,
    )


def sample_grasps(grasp_vars, num_grasps, model, nerf_config):
    """
    Generates and samples from a distribution of grasps, returning a batch of
    grasp points and their associated normals.
    Args:
        grasp_vars: tensor, shape [B, n_f, 6], of grasp positions ([..., :3]) and
            approach directions ([..., 3:]), in spherical coordinates,
            to be used for rendering.
        model: NeRF defining the object density.
        nerf_config: config.Nerf object with the current NeRF parameters.

    Returns a batch of sampled grasp points and normals, which can be
    used to compute grasp metrics.
    """
    rays_o, rays_d, weights, z_vals = get_grasp_distribution(
        grasp_vars, model, nerf_config
    )

    B, n_f, _ = rays_o.shape

    # Normalize to ensure rays_d are unit length.
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # Compute where sample points lie in 3D space.
    # [B, n_f, num_points, 3]
    sample_points = rays_o.unsqueeze(-2) + z_vals.unsqueeze(-1) * rays_d.unsqueeze(-2)

    # print('sample points [B, n_f, num_points, 3]: ', sample_points.shape)

    # Mask sample points to clip out object.
    weight_mask = sample_points[..., 1] >= 0.01  # [B, n_f, num_points]
    weight_mask = torch.logical_and(
        weight_mask, torch.all(torch.abs(sample_points) <= 0.1, dim=-1)
    )

    weights = weights * weight_mask  # [B, n_f, num_points]

    print(
        "Fraction of accepted weights:",
        torch.sum(weight_mask) / torch.sum(torch.ones_like(weight_mask)),
    )

    # Generate distribution from which we'll sample grasp points.
    weights = torch.clip(weights, 0.0, 1.0)  # [B, n_f, num_points]

    # Create mask for which rays are empty.
    grasp_mask = torch.sum(weights, -1) > 0.5  # [B, n_f]

    # If using expected surface.
    if nerf_config.expected_surface:
        num_grasps = 1  # Override num_grasps since expectation is at a point.

        grasp_points = torch.sum(sample_points * weights.unsqueeze(-1), dim=-2)
        grasp_points = grasp_points.reshape(B, n_f, 1, 3)
    else:
        # Create categorical distribution object.
        grasp_dist = torch.distributions.Categorical(probs=weights + 1e-15)

        # Sample num_grasps grasp from this distribution.
        grasp_inds = grasp_dist.sample(
            sample_shape=[num_grasps]
        )  # [num_grasps, B, n_f]
        grasp_inds = grasp_inds.permute(2, 0, 1)  # [B, n_f, num_grasps]
        grasp_inds = grasp_inds.reshape(B, n_f, num_grasps, 1).expand(
            B, n_f, num_grasps, 3
        )

        # Collect grasp points.
        grasp_points = torch.gather(
            sample_points, -2, grasp_inds
        )  # [B, n_f, num_grasps, 3]

    # Estimate densities at fingertips.
    density_ests, _ = est_grads_vals(
        model, rays_o.reshape(B, -1, 3), nerf_config.grad_config
    )  # Shape [B, n_f]
    # density_ests, _ = est_grads_vals(
    #     model, rays_o.reshape(B, -1, 3), GRAD_CONFIG
    # )
    # density_ests = density_ests.reshape(B, n_f) # Should be redundant.

    # Mask approach directions that will not collide with object
    grasp_mask = torch.logical_and(grasp_mask, density_ests < 50)  # Shape [B, n_f]
    grasp_mask = grasp_mask.all(-1, keepdim=True)  # Shape [B, 1]

    print(
        "Fraction of accepted grasps:",
        torch.sum(grasp_mask) / torch.sum(torch.ones_like(grasp_mask)),
    )

    # Mask approach directions that will not collide with each other
    # approach_mask = grasp_utils.intersect_grasp_dirs(
    #     grasp_vars, model, model.centroid, B, n_f
    # )
    # grasp_mask = torch.logical_and(grasp_mask, approach_mask)

    # Estimate gradients.
    if nerf_config.expected_gradient:
        expected_surface_points = torch.sum(
            sample_points * weights.unsqueeze(-1), dim=-2
        )  # [B, n_f, 3]

        _, grad_ests = est_grads_vals(
            model, expected_surface_points, nerf_config.grad_config
        )  # [B, n_f, 3]

        # _, grad_ests = est_grads_vals(model, expected_surface_points, GRAD_CONFIG)
        grad_ests = grad_ests.unsqueeze(-2).expand(B, n_f, num_grasps, 3)

    else:

        _, grad_ests = est_grads_vals(
            model, grasp_points.reshape(B, -1, 3), nerf_config.grad_config
        )  # [B, n_f * num_grasps, 3]
        # _, grad_ests = est_grads_vals(
        #     model, grasp_points.reshape(B, -1, 3), GRAD_CONFIG
        # )
        grad_ests = grad_ests.reshape(B, n_f, num_grasps, 3)

    grad_ests = grad_ests / torch.norm(grad_ests, dim=-1, keepdim=True)

    # # Enforce constraint that expected normal is no more than 30 deg from approach dir.
    # grasp_mask = torch.logical_and(
    #     grasp_mask,
    #     torch.median(torch.sum(grad_ests * rays_d.unsqueeze(-2), dim=-1), dim=-1)[0]
    #     >= 0.5,
    # )

    # NOTE: deprecated?
    # Permute dims to put batch dimensions together.
    grasp_points = grasp_points.permute(0, 2, 1, 3)  # [B, num_grasps, n_f, 3]
    grad_ests = grad_ests.permute(0, 2, 1, 3)  # [B, num_grasps, n_f, 3]

    return grasp_points, grad_ests, grasp_mask


def est_grads_vals(nerf, grasp_points, grad_config):
    """
    Uses sampling to estimate gradients and density values for
    a given batch of grasp points.
    Args:
        nerf: NeRF object to query density/grads from.
        grasp_points: tensor, size [B, n_f, 3] of grasp points.
        sigma: standard deviation of distribution used for estimation.
        num_samples: number of samples to draw for estimation.

    Returns densities, shape [B, n_f], and grads, shape [B, n_f, 3]
    """
    B, n_f, _ = grasp_points.shape
    device = grasp_points.device

    if grad_config.method == config.GradType.AVERAGE:
        gps = grasp_points.reshape(B, 1, n_f, 3) + grad_config.variance * torch.randn(
            B, grad_config.num_samples, n_f, 3, device=device
        )
        gps.requires_grad = True

        grads, densities = nerf_grads(nerf, gps.reshape(-1, n_f, 3), ret_densities=True)

        grad_ests = torch.mean(grads.reshape(B, -1, n_f, 3), dim=1)
        density_ests = torch.mean(densities.reshape(B, -1, n_f), dim=1)
    elif grad_config.method == config.GradType.CENTRAL_DIFFERENCE:

        U = (
            torch.cat([torch.eye(3, device=device), -torch.eye(3, device=device)])
            .reshape(1, 6, 1, 3)
            .expand(B, 6, n_f, 3)
        )
        gps = grasp_points.reshape(B, 1, n_f, 3) + grad_config.variance * U
        densities = nerf_densities(nerf, gps.reshape(-1, n_f, 3)).reshape(B, 6, n_f)

        grad_ests = torch.stack(
            [
                (densities[:, ii, :] - densities[:, ii + 3, :])
                / (2 * grad_config.variance)
                for ii in range(3)
            ],
            dim=-1,
        )

        density_ests = torch.mean(densities, dim=1)
    elif grad_config.method == config.GradType.GAUSSIAN:
        dgs = grad_config.variance * torch.randn(
            B, grad_config.num_samples, n_f, 3, device=device
        )
        gps = grasp_points.reshape(B, 1, n_f, 3) + dgs
        # grads, densities = nerf_grads(nerf, gps.reshape(-1, n_f, 3), ret_densities=True)
        densities = nerf_densities(nerf, gps.reshape(-1, n_f, 3))

        # grads = grads.reshape(B, grad_config.num_samples, n_f, 3)
        densities = densities.reshape(B, grad_config.num_samples, n_f)

        origin_densities = nerf_densities(nerf, grasp_points.reshape(B, n_f, 3))
        origin_densities = origin_densities.reshape(B, 1, n_f)
        grad_ests = (
            torch.mean(
                (densities - origin_densities).reshape(
                    B, grad_config.num_samples, n_f, 1
                )
                * dgs,
                dim=1,
            )
            / grad_config.variance
        )
        density_ests = origin_densities.reshape(B, n_f)

    return density_ests, grad_ests


def nerf_densities(nerf, grasp_points):
    """
    Evaluates density of a batch of grasp points, shape [B, n_f, 3].
    """
    B, n_f, _ = grasp_points.shape
    query_points = grasp_points.reshape(1, -1, 3)

    return nerf.density(query_points).reshape(B, n_f)


def nerf_grads(nerf, grasp_points, ret_densities=False):
    """
    Evaluates NeRF densities at a batch of grasp points.
    Args:
        nerf: nerf_shared.NeRF object with density channel to diff.
        grasp_points: set of grasp points at which to take gradients.
    """
    densities = nerf_densities(nerf, grasp_points)
    density_grads = torch.autograd.grad(
        torch.sum(densities), grasp_points, create_graph=False
    )[0]
    if ret_densities:
        return density_grads, densities
    else:
        return density_grads


def check_gradients(
    nerf, face_centers, face_normals, grad_params, nerf_config, chunk=500
):
    """
    Checks gradients of NeRF against the true normals at the face centers.
    Args:
        nerf: NeRF object with density field representing object.
        face_centers: numpy array of face centers, shape [num_faces, 3]
        face_normals: numpy array of face normals, shape [num_faces, 3].
        grad_params: dict of parameters to pass to gradient estimation.
        chunk: (optional) number of normals to query per iteration.
    Returns a list of cosine distances between the
    true normals and the negative (outward-pointing) gradients of the NeRF.
    """
    device, dtype = next(nerf.parameters()).device, next(nerf.parameters()).dtype
    query_points = torch.from_numpy(face_centers).to(device, dtype)

    true_normals = torch.from_numpy(face_normals).to(device, dtype)

    cos_sims = torch.zeros(face_normals.shape[0], device=device, dtype=dtype)

    for ii in range(math.ceil(face_normals.shape[0] / chunk)):
        start, stop = chunk * ii, chunk * (ii + 1)

        _, gradients = est_grads_vals(
            nerf, query_points[start:stop, :].reshape(1, -1, 3), nerf_config.grad_config
        )

        gradients = gradients.reshape(-1, 3)

        cos_sims[start:stop] = grasp_utils.cos_similarity(
            -gradients, true_normals[start:stop, :]
        )

    return cos_sims


def get_centroid(model, num_samples=10000, thresh=None):
    """Computes approximate centroid using NeRF model."""
    min_corner = torch.tensor([bb[0] for bb in grasp_utils.OBJ_BOUNDS])
    max_corner = torch.tensor([bb[-1] for bb in grasp_utils.OBJ_BOUNDS])

    box_center = (1 / 2) * (max_corner + min_corner)
    box_scale = max_corner - min_corner

    sample_points = box_scale.reshape(1, 3) * (
        torch.rand(num_samples, 3) - 0.5
    ) + box_center.reshape(1, 3)
    sample_points = sample_points.cuda().float()

    sample_densities = model.density(sample_points)
    if thresh is not None:
        mask_inds = torch.argwhere(sample_densities > thresh)
        return torch.mean(sample_points[mask_inds], dim=0)
    else:
        sample_densities = sample_densities / torch.sum(sample_densities)

        return torch.sum(sample_densities.reshape(-1, 1) * sample_points, dim=0)


def correct_z_dists(model, grasp_points, nerf_config):
    rays_o = grasp_points[:, :3]
    rays_d_raw = grasp_points[:, 3:]
    rays_d = grasp_utils.res_to_true_dirs(rays_o, rays_d_raw, model.centroid)

    exp_surf_points = None

    for ii in range(nerf_config.num_z_dist_iters):
        grasp_points = torch.cat([rays_o, rays_d_raw], dim=-1).reshape(3, 6)
        _, _, weights, z_vals = get_grasp_distribution(
            grasp_points.reshape(1, 3, 6), model, nerf_config
        )

        exp_dists = torch.sum(weights * z_vals, dim=-1).reshape(3, 1)
        exp_surf_points = rays_o + exp_dists * rays_d

        z_correction = nerf_config.des_z_dist - exp_dists
        rays_o = rays_o - 0.1 * z_correction * rays_d

    # Project points to lie above floor.
    # rays_o[:, 1] = torch.clamp(rays_o[:, 1], min=grasp_utils.OBJ_BOUNDS[1][0])

    # Correct directions to keep surface points consistent.
    exp_dists = torch.norm(rays_o - exp_surf_points, dim=-1, keepdim=True)
    rays_d = (exp_surf_points - rays_o) / exp_dists

    return rays_o, rays_d


def intersect_grasp_dirs(grasp_vars, model, B, n_f, nerf_config):
    grasp_pairs = [[0, 1], [0, 2], [1, 2]]
    grasp_starts = grasp_vars[:, :, :3]
    grasp_dirs = grasp_utils.res_to_true_dirs(
        grasp_starts.reshape(-1, 3),
        grasp_vars[:, :, 3:].reshape(-1, 3),
        centroid=model.centroid,
    ).reshape(-1, n_f, 3)
    _, _, weights, z_vals = get_grasp_distribution(grasp_vars, model, nerf_config)
    z_dists = torch.sum(weights * z_vals, dim=-1)  # shape B x 3
    approach_mask = torch.ones((B, 1), device=grasp_vars.device)
    for i, j in grasp_pairs:
        z1, z2 = z_dists[:, i].view(-1, 1), z_dists[:, j].view(-1, 1)
        p1, p2 = grasp_starts[:, i], grasp_starts[:, j]
        l1, l2 = grasp_dirs[:, i], grasp_dirs[:, j]
        n1 = torch.cross(l1, l2, dim=1)
        n2 = torch.cross(l2, n1, dim=1)
        dist = (n1 * (p1 - p2)).sum(dim=1, keepdim=True) / n1.norm(dim=1, keepdim=True)
        d1 = ((p2 - p1) * n2).sum(dim=1, keepdim=True) / (l1 * n2).sum(
            dim=1, keepdim=True
        )
        d2 = ((p1 - p2) * n1).sum(dim=1, keepdim=True) / (l2 * n1).sum(
            dim=1, keepdim=True
        )
        pair_mask = torch.logical_and(torch.logical_and(dist < 0.02, d1 > z1), d2 > z2)
        approach_mask = torch.logical_and(approach_mask, pair_mask)
    return approach_mask
