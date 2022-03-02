"""
Module implementing utils for grasping,
including normal estimation and surface detection.
"""
import math
import render_utils
import torch

def generate_grasp_rays(grasp_vars):
    """
    Generates rays in the format needed for a nerf_shared.Renderer.

    Args:
        grasp_vars: tensor, shape [B, n_f, 5], of grasp positions ([..., :3]) and
            approach directions ([..., 3:]), in spherical coordinates,
            to be used for rendering.

    Returns (rays_o, rays_d) expected for rendering.
    """
    # Construct ray origins (first three elements of decision vars).
    rays_o = grasp_vars[:, :, :3]

    # Construct ray directions (using spherical coords).
    phi, theta = grasp_vars[..., 3], grasp_vars[..., 4]
    rays_d = torch.cat([torch.cos(phi) * torch.sin(theta),
                        torch.sin(phi) * torch.sin(theta),
                        torch.cos(theta)], dim=-1)

    return rays_o, rays_d

def get_grasp_distribution(grasp_vars,
                           coarse_model,
                           fine_model,
                           renderer,
                           chunk=1024*32):
    """
    Generates a "grasp distribution," a set of n_f categorical distributions
    for where each finger will contact the object surface, along with the associated
    contact points.

    Args:
        grasp_vars: tensor, shape [B, n_f, 5], of grasp positions ([..., :3]) and
            approach directions ([..., 3:]), in spherical coordinates,
            to be used for rendering.
        nerf: nerf_shared.NeRF object defining the object density.
        renderer: nerf_shared.Renderer object which will be used to generate the
            termination probabilities and points.
        chunk: Max number of rays to render at once on the GPU.

    Returns a tuple (points, probs) defining the grasp distribution.
    """
    B, n_f, _ = grasp_vars.shape

    # Create ray batch
    rays_o, rays_d = generate_grasp_rays(grasp_vars)
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near = renderer.near * torch.ones_like(rays_d[...,:1])
    far = renderer.far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    if renderer.use_viewdirs:
        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()
        rays = torch.cat([rays, viewdirs], -1)

    render_results = renderer.render_batch(coarse_model, fine_model, rays,
                                           chunk, retweights=True)

    weights = render_results['weights'].reshape(B, n_f, -1)
    z_vals = render_results['z_vals'].reshape(B, n_f, -1)

    return rays, weights, z_vals

def sample_grasps(grasp_vars, nerf, renderer):
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
    raise NotImplementedError

def est_grads_vals(nerf, grasp_points, method='gaussian', sigma=1e-3, num_samples=2500):
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

    if method == 'grad_averaging':
        gps = (grasp_points.reshape(B, 1, n_f, 3)
               + sigma * torch.randn(B, num_samples, n_f, 3))
        gps.requires_grad = True

        grads, densities = nerf_grads(nerf, gps.reshape(-1, n_f, 3), ret_densities=True)

        grad_ests = torch.mean(grads.reshape(B, -1, n_f, 3), dim=1)
        density_ests = torch.mean(densities.reshape(B, -1, n_f), dim=1)
    elif method == 'central_difference':

        U = torch.cat([torch.eye(3), -torch.eye(3)]).reshape(1, 6, 1, 3).expand(B,6, n_f, 3)
        gps = grasp_points.reshape(B, 1, n_f, 3) + sigma*U

        densities = nerf_densities(nerf, gps.reshape(-1, n_f, 3)).reshape(B, 6, n_f)

        grad_ests = torch.stack(
            [(densities[:, ii, :] - densities[:, ii+3, :])
             / (2 * sigma) for ii in range(3)],
            dim=-1)

        density_ests = torch.mean(densities, dim=1)
    elif method == 'gaussian':
        dgs = sigma * torch.randn(B, num_samples, n_f, 3)
        gps = grasp_points.reshape(B, 1, n_f, 3) + dgs
        # grads, densities = nerf_grads(nerf, gps.reshape(-1, n_f, 3), ret_densities=True)
        densities = nerf_densities(nerf, gps.reshape(-1, n_f, 3))

        # grads = grads.reshape(B, num_samples, n_f, 3)
        densities = densities.reshape(B, num_samples, n_f)

        origin_densities = nerf_densities(nerf, grasp_points.reshape(B, n_f, 3))
        origin_densities = origin_densities.reshape(B, 1, n_f)

        grad_ests = torch.mean(
            (densities - origin_densities).reshape(B, num_samples, n_f, 1) * dgs,
            dim = 1) / sigma

        density_ests = origin_densities.reshape(B, n_f)

    return density_ests, grad_ests

def cos_similarity(a, b):
    """
    Returns the cosine similarity of two batches of vectors, a and b.
    """
    return torch.sum(a * b, dim=-1)/(torch.norm(a, dim=-1) * torch.norm(b, dim=-1))

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
        start, stop = chunk*ii, chunk*(ii+1)

        _, gradients = est_grads_vals(nerf, query_points[start:stop, :].reshape(1,-1,3),
                                      **grad_params)

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
        new_samples = (mu.reshape(1, n, 1)
             + torch.linalg.cholesky(Sigma).reshape(1, n, n)
             @ torch.randn(num_points, n, 1)).reshape(num_points, n)

        accept = constraint(new_samples)
        num_curr = torch.sum(accept.int())

        slice_end = min(num_accepted+num_curr, num_points)
        slice_len = min(num_points - num_accepted, num_curr)
        accept_inds = accept.nonzero(as_tuple=True)[0][:slice_len]

        sample_points[num_accepted:slice_end] = new_samples[accept_inds]

        num_accepted += slice_len
        ii += 1
        print(ii, num_accepted)

    return sample_points