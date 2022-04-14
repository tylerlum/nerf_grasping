"""
Module implementing utils for grasping,
including normal estimation and surface detection.
"""
import math
import torch
import logging

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
    model = NeRFNetwork(bound=opt.bound, cuda_ray=opt.cuda_ray)

    # Create trainer with NeRF; use its constructor to load network weights from file.
    trainer = utils.Trainer("ngp",
                            vars(opt),
                            model,
                            workspace=opt.workspace,
                            criterion=None,
                            fp16=opt.fp16,
                            metrics=[None],
                            use_checkpoint="latest")

    return trainer.model


def get_grasp_distribution(grasp_vars,
                           model,
                           num_steps=128,
                           upsample_steps=128,
                           near_finger=0.05,
                           far_finger=0.15,
                           perturb=False,
                           residual_dirs=False):
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

    if residual_dirs:
        rays_d = torch.reshape(rays_d, [-1, 1]).float()
        rays_d = torch.exp(rays_d) @ -rays_o
    else:
        rays_d = torch.reshape(rays_d, [-1, 3]).float()
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # Get near/far bounds for each ray.
    # First, generate near/far bounds for cube.
    near_cube, far_cube = renderer.near_far_from_bound(rays_o,
                                                       rays_d,
                                                       model.bound,
                                                       type="cube")

    # Then setup near/far bounds for fingertip renderer.
    near_finger = near_finger * torch.ones_like(near_cube)
    far_finger = far_finger * torch.ones_like(far_cube)

    # Keep mask of rays which miss the cube altogether.
    mask = near_cube > 1e8

    # Then take near = max(near_finger, near_cube), far = min(far_finger, far_cube).
    near = torch.maximum(near_finger, near_cube)
    far = torch.minimum(far_finger, far_cube)
    far[mask] = 1e9

    z_vals = torch.linspace(0.0, 1.0, num_steps,
                            device=device).unsqueeze(0)  # [1, T]
    z_vals = z_vals.expand((N, num_steps))  # [N, T]
    z_vals = near + (far - near) * z_vals  # [N, T], in [near, far]

    # perturb z_vals
    sample_dist = (far - near) / num_steps
    if perturb:
        z_vals = z_vals + \
            (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist

    # generate pts
    pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(
        -1)  # [N, 1, 3] * [N, T, 3] -> [N, T, 3]
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
                [deltas, sample_dist * torch.ones_like(deltas[..., :1])],
                dim=-1)

            alphas = 1 - torch.exp(-deltas * sigmas)  # [N, T]
            alphas_shifted = torch.cat(
                [torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15],
                dim=-1)  # [N, T+1]
            weights = alphas * torch.cumprod(alphas_shifted,
                                             dim=-1)[..., :-1]  # [N, T]

            # sample new z_vals
            z_vals_mid = z_vals[..., :-1] + 0.5 * deltas[..., :-1]  # [N, T-1]
            new_z_vals = renderer.sample_pdf(
                z_vals_mid,
                weights[:, 1:-1],
                upsample_steps,
                det=not model.training).detach()  # [N, t]

            new_pts = rays_o.unsqueeze(
                -2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(
                    -1)  # [N, 1, 3] * [N, t, 3] -> [N, t, 3]
            new_pts = new_pts.clamp(-model.bound, model.bound)

        # only forward new points to save computation
        new_dirs = rays_d.unsqueeze(-2).expand_as(new_pts)
        new_sigmas, new_rgbs = model(new_pts.reshape(-1, 3),
                                     new_dirs.reshape(-1, 3))
        new_rgbs = new_rgbs.reshape(N, upsample_steps, 3)  # [N, t, 3]
        new_sigmas = new_sigmas.reshape(N, upsample_steps)  # [N, t]

        # re-order
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)  # [N, T+t]
        z_vals, z_index = torch.sort(z_vals, dim=-1)

        sigmas = torch.cat([sigmas, new_sigmas], dim=-1)  # [N, T+t]
        sigmas = torch.gather(sigmas, dim=-1, index=z_index)

        rgbs = torch.cat([rgbs, new_rgbs], dim=-2)  # [N, T+t, 3]
        rgbs = torch.gather(rgbs,
                            dim=-2,
                            index=z_index.unsqueeze(-1).expand_as(rgbs))

    # render core
    deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T-1]
    deltas = torch.cat(
        [deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

    alphas = 1 - torch.exp(-deltas * sigmas)  # [N, T]
    alphas_shifted = torch.cat(
        [torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15],
        dim=-1)  # [N, T+1]
    weights = alphas * torch.cumprod(alphas_shifted,
                                     dim=-1)[..., :-1]  # [N, T]

    return rays_o, rays_d, weights, z_vals


def sample_grasps(grasp_vars,
                  num_grasps,
                  coarse_model,
                  fine_model,
                  renderer,
                  chunk=1024 * 32):
    """
    Generates and samples from a distribution of grasps, returning a batch of
    grasp points and their associated normals.

    Args:
        grasp_vars: tensor, shape [B, n_f, 5], of grasp positions ([..., :3]) and
            approach directions ([..., 3:]), in spherical coordinates,
            to be used for rendering.
        nerf: nerf_shared.NeRF object defining the object density.
        renderer: nerf_shared.Renderer object which will be used to generate the
            termination probabilities and points.

    Returns a batch of sampled grasp points and normals, which can be
    used to compute grasp metrics.
    """
    rays, weights, z_vals = get_grasp_distribution(grasp_vars,
                                                   coarse_model,
                                                   fine_model,
                                                   renderer,
                                                   chunk=chunk)

    B, n_f, _ = rays.shape

    # Unpack ray origins and dirs from ray batch.
    rays_o, rays_d = rays[..., :3], rays[..., 3:6]

    # Normalize to ensure rays_d are unit length.
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # Compute where sample points lie in 3D space.
    # [B, n_f, num_points, 3]
    sample_points = rays_o.unsqueeze(
        -2) + z_vals.unsqueeze(-1) * rays_d.unsqueeze(-2)

    # Generate distribution from which we'll sample grasp points.
    grasp_dist = torch.distributions.Categorical(probs=weights + 1e-15)

    # Create mask for which rays are empty.
    grasp_mask = torch.sum(weights, -1) > 1e-8

    # Sample num_grasps grasp from this distribution.
    grasp_inds = grasp_dist.sample(sample_shape=[num_grasps
                                                 ])  # [num_grasps, B, n_f]
    grasp_inds = grasp_inds.permute(2, 0, 1)  # [B, n_f, num_grasps]
    grasp_inds = grasp_inds.reshape(B, n_f, num_grasps,
                                    1).expand(B, n_f, num_grasps, 3)

    # Collect grasp points.
    grasp_points = torch.gather(sample_points, -2,
                                grasp_inds)  # [B, n_f, num_grasps, 3]

    # Estimate gradients.
    _, grad_ests = est_grads_vals(fine_model, grasp_points.reshape(B, -1, 3))
    grad_ests = grad_ests.reshape(B, n_f, num_grasps, 3)

    # Estimate densities at fingertips.
    density_ests, _ = est_grads_vals(fine_model, rays_o.reshape(B, -1, 3))
    density_ests = density_ests.reshape(B, n_f)

    grasp_mask = torch.logical_and(grasp_mask, density_ests < 150)

    # Permute dims to put batch dimensions together.
    grad_ests = grad_ests.permute(0, 2, 1, 3)

    return grasp_points, grad_ests, grasp_mask


def est_grads_vals(nerf,
                   grasp_points,
                   method="central_difference",
                   sigma=1e-3,
                   num_samples=1000):
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

    if method == "grad_averaging":
        gps = grasp_points.reshape(
            B, 1, n_f, 3) + sigma * torch.randn(B, num_samples, n_f, 3)
        gps.requires_grad = True

        grads, densities = nerf_grads(nerf,
                                      gps.reshape(-1, n_f, 3),
                                      ret_densities=True)

        grad_ests = torch.mean(grads.reshape(B, -1, n_f, 3), dim=1)
        density_ests = torch.mean(densities.reshape(B, -1, n_f), dim=1)
    elif method == "central_difference":

        U = (torch.cat([torch.eye(3),
                        -torch.eye(3)]).reshape(1, 6, 1,
                                                3).expand(B, 6, n_f, 3))
        gps = grasp_points.reshape(B, 1, n_f, 3) + sigma * U
        densities = nerf_densities(nerf, gps.reshape(-1, n_f,
                                                     3)).reshape(B, 6, n_f)

        grad_ests = torch.stack(
            [(densities[:, ii, :] - densities[:, ii + 3, :]) / (2 * sigma)
             for ii in range(3)],
            dim=-1,
        )

        density_ests = torch.mean(densities, dim=1)
    elif method == "gaussian":
        dgs = sigma * torch.randn(B, num_samples, n_f, 3)
        gps = grasp_points.reshape(B, 1, n_f, 3) + dgs
        # grads, densities = nerf_grads(nerf, gps.reshape(-1, n_f, 3), ret_densities=True)
        densities = nerf_densities(nerf, gps.reshape(-1, n_f, 3))

        # grads = grads.reshape(B, num_samples, n_f, 3)
        densities = densities.reshape(B, num_samples, n_f)

        origin_densities = nerf_densities(nerf,
                                          grasp_points.reshape(B, n_f, 3))
        origin_densities = origin_densities.reshape(B, 1, n_f)
        grad_ests = (torch.mean(
            (densities - origin_densities).reshape(B, num_samples, n_f, 1) *
            dgs,
            dim=1,
        ) / sigma)
        density_ests = origin_densities.reshape(B, n_f)

    return density_ests, grad_ests


def nerf_densities(nerf, grasp_points):
    """
    Evaluates density of a batch of grasp points, shape [B, n_f, 3].
    """
    B, n_f, _ = grasp_points.shape
    query_points = grasp_points.reshape(1, -1, 3)

    return nerf.get_density(query_points).reshape(B, n_f)


def nerf_grads(nerf, grasp_points, ret_densities=False):
    """
    Evaluates NeRF densities at a batch of grasp points.
    Args:
        nerf: nerf_shared.NeRF object with density channel to diff.
        grasp_points: set of grasp points at which to take gradients.
    """
    densities = nerf_densities(nerf, grasp_points)
    density_grads = torch.autograd.grad(torch.sum(densities),
                                        grasp_points,
                                        create_graph=False)[0]
    if ret_densities:
        return density_grads, densities
    else:
        return density_grads


def cos_similarity(a, b):
    """
    Returns the cosine similarity of two batches of vectors, a and b.
    """
    return torch.sum(a * b,
                     dim=-1) / (torch.norm(a, dim=-1) * torch.norm(b, dim=-1))


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
    device, dtype = next(nerf.parameters()).device, next(
        nerf.parameters()).dtype
    query_points = torch.from_numpy(face_centers).to(device, dtype)

    true_normals = torch.from_numpy(face_normals).to(device, dtype)

    cos_sims = torch.zeros(face_normals.shape[0], device=device, dtype=dtype)

    for ii in range(math.ceil(face_normals.shape[0] / chunk)):
        start, stop = chunk * ii, chunk * (ii + 1)

        _, gradients = est_grads_vals(
            nerf, query_points[start:stop, :].reshape(1, -1, 3), **grad_params)

        gradients = gradients.reshape(-1, 3)

        cos_sims[start:stop] = cos_similarity(-gradients,
                                              true_normals[start:stop, :])

    return cos_sims


def rejection_sample(mu, Sigma, constraint, num_points):
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

    sample_points = torch.zeros(num_points, n)
    num_accepted = 0
    ii = 0

    while num_accepted < num_points:
        new_samples = (mu.reshape(1, n, 1) +
                       torch.linalg.cholesky(Sigma).reshape(1, n, n)
                       @ torch.randn(num_points, n, 1)).reshape(num_points, n)

        accept = constraint(new_samples)
        num_curr = torch.sum(accept.int())

        slice_end = min(num_accepted + num_curr, num_points)
        slice_len = min(num_points - num_accepted, num_curr)
        accept_inds = accept.nonzero(as_tuple=True)[0][:slice_len]

        sample_points[num_accepted:slice_end] = new_samples[accept_inds]

        num_accepted += slice_len
        ii += 1
        logging.debug("rejection_sample(): itr=%d, accepted=%d", ii,
                      num_accepted)

    return sample_points
