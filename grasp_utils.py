"""
Module implementing utils for grasping,
including normal estimation and surface detection.
"""
import math
import torch
import logging


def get_grasp_distribution(
    grasp_vars,
    coarse_model,
    fine_model,
    renderer,
    chunk=1024 * 32,
    residual_dirs=False,
):
    """
    Generates a "grasp distribution," a set of n_f categorical distributions
    for where each finger will contact the object surface, along with the associated
    contact points.

    Args:
        grasp_vars: tensor, shape [B, n_f, 6], of grasp positions ([..., :3]) and
            approach directions ([..., 3:]) to be used for rendering. Approach
            directions need not be normalized.
        nerf: nerf_shared.NeRF object defining the object density.
        renderer: nerf_shared.Renderer object which will be used to generate the
            termination probabilities and points.
        chunk: Max number of rays to render at once on the GPU.
        residual_dirs: Whether or not dirs decision variable is an actual direction
            or residual direction from point to origin

    Returns a tuple (points, probs) defining the grasp distribution.
    """
    # TODO(pculbert): Support grasp_vars without a leading batch dim.
    B, n_f, _ = grasp_vars.shape
    assert grasp_vars.shape[-1] == 6

    # Create ray batch
    rays_o, rays_d = grasp_vars[..., :3], grasp_vars[..., 3:]
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    if residual_dirs:
        rays_d = torch.reshape(rays_d, [-1, 1]).float()
        rays_d = torch.exp(rays_d) @ -rays_o
    else:
        rays_d = torch.reshape(rays_d, [-1, 3]).float()
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # Create near/far bounds and append to ray batch.
    near = renderer.near * torch.ones_like(rays_d[..., :1])
    far = renderer.far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    if renderer.use_viewdirs:
        rays = torch.cat([rays, rays_d], dim=-1)

    # Use render_batch to get termination weights for all points.
    render_results = renderer.render_batch(
        coarse_model, fine_model, rays, chunk, retweights=True
    )

    weights = render_results["weights"].reshape(B, n_f, -1)
    z_vals = render_results["z_vals"].reshape(B, n_f, -1)
    rays = rays.reshape(B, n_f, -1)

    return rays, weights, z_vals


def sample_grasps(
    grasp_vars, num_grasps, coarse_model, fine_model, renderer, chunk=1024 * 32
):
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
    rays, weights, z_vals = get_grasp_distribution(
        grasp_vars, coarse_model, fine_model, renderer, chunk=chunk
    )

    B, n_f, _ = rays.shape

    # Unpack ray origins and dirs from ray batch.
    rays_o, rays_d = rays[..., :3], rays[..., 3:6]

    # Normalize to ensure rays_d are unit length.
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # Compute where sample points lie in 3D space.
    # [B, n_f, num_points, 3]
    sample_points = rays_o.unsqueeze(-2) + z_vals.unsqueeze(-1) * rays_d.unsqueeze(-2)

    # Generate distribution from which we'll sample grasp points.
    grasp_dist = torch.distributions.Categorical(probs=weights + 1e-15)

    # Create mask for which rays are empty.
    grasp_mask = torch.sum(weights, -1) > 1e-8

    # Sample num_grasps grasp from this distribution.
    grasp_inds = grasp_dist.sample(sample_shape=[num_grasps])  # [num_grasps, B, n_f]
    grasp_inds = grasp_inds.permute(2, 0, 1)  # [B, n_f, num_grasps]
    grasp_inds = grasp_inds.reshape(B, n_f, num_grasps, 1).expand(B, n_f, num_grasps, 3)

    # Collect grasp points.
    grasp_points = torch.gather(
        sample_points, -2, grasp_inds
    )  # [B, n_f, num_grasps, 3]

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


def est_grads_vals(
    nerf, grasp_points, method="central_difference", sigma=1e-3, num_samples=1000
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

    if method == "grad_averaging":
        gps = grasp_points.reshape(B, 1, n_f, 3) + sigma * torch.randn(
            B, num_samples, n_f, 3
        )
        gps.requires_grad = True

        grads, densities = nerf_grads(nerf, gps.reshape(-1, n_f, 3), ret_densities=True)

        grad_ests = torch.mean(grads.reshape(B, -1, n_f, 3), dim=1)
        density_ests = torch.mean(densities.reshape(B, -1, n_f), dim=1)
    elif method == "central_difference":

        U = (
            torch.cat([torch.eye(3), -torch.eye(3)])
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
        dgs = sigma * torch.randn(B, num_samples, n_f, 3)
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

    return nerf.get_density(query_points).reshape(B, n_f)


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
        new_samples = (
            mu.reshape(1, n, 1)
            + torch.linalg.cholesky(Sigma).reshape(1, n, n)
            @ torch.randn(num_points, n, 1)
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
