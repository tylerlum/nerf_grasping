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


def load_nerf(opt):
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
    return trainer.model


def get_grasp_distribution(
    grasp_vars,
    model,
    num_steps=128,
    upsample_steps=256,
    near_finger=0.0001,
    far_finger=0.15,
    perturb=True,
    residual_dirs=True,
    centroid=0.,
):
    """
    Generates a "grasp distribution," a set of n_f categorical distributions
    for where each finger will contact the object surface, along with the associated
    contact points.
    Args:
        grasp_vars: tensor, shape [B, n_f, 6], of grasp positions ([..., :3]) and
            approach directions ([..., 3:]) to be used for rendering. Approach
            directions need not be normalized.
        model: torch_ngp NeRFNetwork that represents the scene.
    Returns a tuple (points, probs) defining the grasp distribution.
    """
    # TODO(pculbert): Support grasp_vars without a leading batch dim.
    B, n_f, _ = grasp_vars.shape
    assert grasp_vars.shape[-1] == 6

    # Create ray batch
    rays_o, rays_d = grasp_vars[..., :3], grasp_vars[..., 3:]
    rays_o = torch.reshape(rays_o, [-1, 3]).float()

    N = rays_o.shape[0]
    device = rays_o.device
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    if residual_dirs:
        rays_d = res_to_true_dirs(rays_o, rays_d, centroid=centroid)
    else:
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # Get near/far bounds for each ray.
    # First, generate near/far bounds for cube.
    near_cube, far_cube = renderer.near_far_from_bound(
        rays_o, rays_d, model.bound, type="cube"
    )

    # Then setup near/far bounds for fingertip renderer.
    near_finger = near_finger * torch.ones_like(near_cube)
    far_finger = far_finger * torch.ones_like(far_cube)

    # Keep mask of rays which miss the cube altogether.
    mask = near_cube > 1e8

    # Then take near = max(near_finger, near_cube), far = min(far_finger, far_cube).
    near = torch.maximum(near_finger, near_cube)
    far = torch.minimum(far_finger, far_cube)
    far[mask] = 1e9

    z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0)  # [1, T]
    z_vals = z_vals.expand((N, num_steps))  # [N, T]
    z_vals = near + (far - near) * z_vals  # [N, T], in [near, far]

    # perturb z_vals
    sample_dist = (far - near) / num_steps
    if perturb:
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

    rgbs = rgbs.reshape(N, num_steps, 3)  # [N, T, 3]
    sigmas = sigmas.reshape(N, num_steps)  # [N, T]

    # upsample z_vals (nerf-like)
    if upsample_steps > 0:
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
                z_vals_mid, weights[:, 1:-1], upsample_steps, det=not model.training
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
        new_rgbs = new_rgbs.reshape(N, upsample_steps, 3)  # [N, t, 3]
        new_sigmas = new_sigmas.reshape(N, upsample_steps)  # [N, t]

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

    rays_o = rays_o.reshape(-1, n_f, 3)
    rays_d = rays_d.reshape(-1, n_f, 3)
    z_vals = z_vals.reshape(B, n_f, -1)
    weights = weights.reshape(B, n_f, -1)

    return (
        rays_o,
        rays_d,
        weights,
        z_vals,
    )


def sample_grasps(grasp_vars, num_grasps, model, residual_dirs=True, centroid=0.):
    """
    Generates and samples from a distribution of grasps, returning a batch of
    grasp points and their associated normals.
    Args:
        grasp_vars: tensor, shape [B, n_f, 6], of grasp positions ([..., :3]) and
            approach directions ([..., 3:]), in spherical coordinates,
            to be used for rendering.
        nerf: nerf_shared.NeRF object defining the object density.
        renderer: nerf_shared.Renderer object which will be used to generate the
            termination probabilities and points.
    Returns a batch of sampled grasp points and normals, which can be
    used to compute grasp metrics.
    """
    rays_o, rays_d, weights, z_vals = get_grasp_distribution(
        grasp_vars, model, residual_dirs=residual_dirs, centroid=centroid
    )

    B, n_f, _ = rays_o.shape

    # Normalize to ensure rays_d are unit length.
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # Compute where sample points lie in 3D space.
    # [B, n_f, num_points, 3]
    sample_points = rays_o.unsqueeze(-2) + z_vals.unsqueeze(-1) * rays_d.unsqueeze(-2)

    # Mask sample points to clip out object.
    weight_mask = sample_points[..., 1] >= 0.01
    weight_mask = torch.logical_and(weight_mask, torch.all(torch.abs(sample_points) <= 0.1, dim=-1))

    weights = weights * weight_mask

    print(torch.sum(weight_mask)/torch.sum(torch.ones_like(weight_mask)))

    # Generate distribution from which we'll sample grasp points.
    weights = torch.clip(weights, 0.0, 1.0)
    grasp_dist = torch.distributions.Categorical(probs=weights + 1e-15)

    # Create mask for which rays are empty.
    grasp_mask = torch.sum(weights, -1) > 0.5

    print(torch.sum(grasp_mask)/torch.sum(torch.ones_like(grasp_mask)))

    # Sample num_grasps grasp from this distribution.
    grasp_inds = grasp_dist.sample(sample_shape=[num_grasps])  # [num_grasps, B, n_f]
    grasp_inds = grasp_inds.permute(2, 0, 1)  # [B, n_f, num_grasps]
    grasp_inds = grasp_inds.reshape(B, n_f, num_grasps, 1).expand(B, n_f, num_grasps, 3)

    # Collect grasp points.
    grasp_points = torch.gather(
        sample_points, -2, grasp_inds
    )  # [B, n_f, num_grasps, 3]

    # Estimate gradients.
    _, grad_ests = est_grads_vals(model, grasp_points.reshape(B, -1, 3))
    grad_ests = grad_ests.reshape(B, n_f, num_grasps, 3)

    normal_ests = grad_ests / torch.norm(grad_ests, dim=-1, keepdim=True)

    # # Enforce constraint that expected normal is no more than 30 deg from approach dir.
    # grasp_mask = torch.logical_and(
    #      grasp_mask, torch.median(torch.sum(normal_ests * rays_d.unsqueeze(-2), dim=-1), dim=-1)[0] >= 0.5)

    # print(torch.sum(normal_ests * rays_d.unsqueeze(-2), dim=-1))

    # Estimate densities at fingertips.
    density_ests, _ = est_grads_vals(model, rays_o.reshape(B, -1, 3))
    density_ests = density_ests.reshape(B, n_f)

    grasp_mask = torch.logical_and(grasp_mask, density_ests < 50)

    # Permute dims to put batch dimensions together.
    grad_ests = grad_ests.permute(0, 2, 1, 3)

    return grasp_points, grad_ests, grasp_mask

def est_grads_vals(
    nerf, grasp_points, method="gaussian", sigma=7.5e-3, num_samples=250
):
    """
    Uses sampling to estimate gradients and density values for
    a given batch of grasp points.
    Args:
        nerf: NeRF object to query density/grads from.
        grasp_points: tensor, size [B, n_f, 3] of grasp points.
        sigma: standard deviation of distribution used for estimation.
        num_samples: number of samples to draw for estimation.
    """
    B, n_f, _ = grasp_points.shape
    device = grasp_points.device

    if method == "grad_averaging":
        gps = grasp_points.reshape(B, 1, n_f, 3) + sigma * torch.randn(
            B, num_samples, n_f, 3, device=device
        )
        gps.requires_grad = True

        grads, densities = nerf_grads(nerf, gps.reshape(-1, n_f, 3), ret_densities=True)

        grad_ests = torch.mean(grads.reshape(B, -1, n_f, 3), dim=1)
        density_ests = torch.mean(densities.reshape(B, -1, n_f), dim=1)
    elif method == "central_difference":

        U = (
            torch.cat([torch.eye(3, device=device), -torch.eye(3, device=device)])
            .reshape(1, 6, 1, 3)
            .expand(B, 6, n_f, 3)
        )
        gps = grasp_points.reshape(B, 1, n_f, 3) + sigma * U
        densities = nerf_densities(nerf, gps.reshape(-1, n_f, 3)).reshape(B, 6, n_f)

        grad_ests = torch.stack(
            [
                (densities[:, ii, :] - densities[:, ii + 3, :]) / (2 * sigma)
                for ii in range(3)
            ],
            dim=-1,
        )

        density_ests = torch.mean(densities, dim=1)
    elif method == "gaussian":
        dgs = sigma * torch.randn(B, num_samples, n_f, 3, device=device)
        gps = grasp_points.reshape(B, 1, n_f, 3) + dgs
        # grads, densities = nerf_grads(nerf, gps.reshape(-1, n_f, 3), ret_densities=True)
        densities = nerf_densities(nerf, gps.reshape(-1, n_f, 3))

        # grads = grads.reshape(B, num_samples, n_f, 3)
        densities = densities.reshape(B, num_samples, n_f)

        origin_densities = nerf_densities(nerf, grasp_points.reshape(B, n_f, 3))
        origin_densities = origin_densities.reshape(B, 1, n_f)
        grad_ests = (
            torch.mean(
                (densities - origin_densities).reshape(B, num_samples, n_f, 1) * dgs,
                dim=1,
            )
            / sigma
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


def cos_similarity(a, b):
    """
    Returns the cosine similarity of two batches of vectors, a and b.
    """
    return torch.sum(a * b, dim=-1) / (torch.norm(a, dim=-1) * torch.norm(b, dim=-1))


def check_gradients(nerf, face_centers, face_normals, grad_params, chunk=500):
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
            nerf, query_points[start:stop, :].reshape(1, -1, 3), **grad_params
        )

        gradients = gradients.reshape(-1, 3)

        cos_sims[start:stop] = cos_similarity(-gradients, true_normals[start:stop, :])

    return cos_sims


def rejection_sample(mu, Sigma, constraint, num_points, max_iters=1000):
    """
    Samples points from a Gaussian distribution N(mu, Sigma)
    and uses rejection to find only points which satisfy constraint.
    Args:
        mu: Gaussian mean.
        Sigma: Gaussian covariance.
        constraint: function which returns a Boolean flagging if the point
            satisfies a given constraint, or not.
        num_points: number of points to sample.
    """
    n = mu.shape[0]

    sample_points = torch.zeros(num_points, n).to(mu)
    num_accepted = 0
    ii = 0

    while num_accepted < num_points and ii < max_iters:
        new_samples = (
            mu.reshape(1, n, 1)
            + torch.linalg.cholesky(Sigma).reshape(1, n, n)
            @ torch.randn(num_points, n, 1).to(mu)
        ).reshape(num_points, n)

        accept = constraint(new_samples)
        num_curr = torch.sum(accept.int())

        slice_end = min(num_accepted + num_curr, num_points)
        slice_len = min(num_points - num_accepted, num_curr)
        accept_inds = accept.nonzero(as_tuple=True)[0][:slice_len]

        sample_points[num_accepted:slice_end] = new_samples[accept_inds]

        num_accepted += slice_len
        ii += 1
        logging.debug("rejection_sample(): itr=%d, accepted=%d", ii, num_accepted)

    return sample_points


def nerf_to_ig_R():
    R = scipy.spatial.transform.Rotation.from_euler("X", [np.pi / 2])
    R = R * scipy.spatial.transform.Rotation.from_euler("Y", [np.pi / 2])
    return R


def nerf_to_ig(points, translation=None, return_tensor=True):
    """Goes from points in NeRF (blender) world frame to IG world frame"""
    if isinstance(points, torch.Tensor):
        points = points.cpu().detach().numpy()
    T = np.eye(4)
    R = nerf_to_ig_R().as_matrix()
    T[:3, :3] = R
    translation = translation if translation is not None else np.zeros(3)
    T[:3, -1] = translation
    points = np.concatenate([points, np.ones_like(points)[:, :1]], axis=1)
    points = (T @ points.T).T
    points = points[:, :3] / points[:, 3:]
    if return_tensor:
        points = torch.tensor(points).float().cuda()
    return points


def ig_to_nerf(points, translation=None, return_tensor=True):
    """Goes from points in IG world frame to NeRF (blender) world frame"""
    if isinstance(points, torch.Tensor):
        points = points.cpu().detach().numpy()
    T = np.eye(4)
    R = nerf_to_ig_R().inv().as_matrix()
    T[:3, :3] = R
    translation = translation if translation is not None else np.zeros(3)
    T[:3, -1] = translation
    points = np.concatenate([points, np.ones_like(points)[:, :1]], axis=1)
    points = (T @ points.T).T
    points = points[:, :3] / points[:, 3:]
    if return_tensor:
        points = torch.tensor(points).float().cuda()
    return points

def res_to_true_dirs(rays_o, rays_d, centroid=0.):
    """Converts raw directions rays_d to true directions in world frame.

    Args:
        rays_o: ray origins, shape (B, 3).
        rays_d: (relative) ray directions, shape (B, 3).
        centroid: (optional) object centroid location, shape (3,).
    """
    relative_dir = centroid - rays_o
    rays_d = lietorch.SO3.exp(rays_d).matrix()[:, :3, :3] @ relative_dir.unsqueeze(-1)
    rays_d = rays_d.reshape(-1, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    return rays_d


def get_centroid(model, object_bounds=[(-0.15, 0.15), (0., 0.15), (-0.15, 0.15)], num_samples=10000):
    """Computes approximate centroid using NeRF model."""
    min_corner = torch.tensor([bb[0] for bb in object_bounds])
    max_corner = torch.tensor([bb[-1] for bb in object_bounds])

    box_center = (1/2) * (max_corner + min_corner)
    box_scale = max_corner - min_corner

    sample_points = box_scale.reshape(1,3) * (torch.rand(num_samples,3)-0.5) + box_center.reshape(1,3)
    sample_points = sample_points.cuda().float()

    sample_densities = model.density(sample_points)
    sample_densities = sample_densities / torch.sum(sample_densities)

    return torch.sum(sample_densities.reshape(-1, 1) * sample_points,dim=0)

def correct_z_dists(model, grasp_points, centroid=0., des_z_dist=0.025, num_iters=10):
    rays_o = grasp_points[:, :3]
    rays_d_raw = grasp_points[:, 3:]
    rays_d = res_to_true_dirs(rays_o, rays_d_raw, centroid)

    for ii in range(num_iters):
        grasp_points = torch.cat([rays_o, rays_d_raw], dim=-1).reshape(3,6)
        _, _, weights, z_vals = get_grasp_distribution(
                grasp_points.reshape(1, 3, 6),
                model,
                residual_dirs=True,
            )

        z_correction = des_z_dist - torch.sum(weights * z_vals, dim=-1).reshape(3, 1)
        rays_o = rays_o - 0.1 * z_correction * rays_d

    return rays_o

def box_projection(x, object_bounds=[(-0.1, 0.1), (0.01, 0.05), (-0.1, 0.1)]):
    B, _ = x.shape
    x = x.reshape(B, 3, -1)

    lower = torch.tensor([oo[0] for oo in object_bounds]).to(x)
    upper = torch.tensor([oo[1] for oo in object_bounds]).to(x)

    x[..., :3] = x[..., :3].clamp(lower, upper)

    return x.reshape(B, -1)

