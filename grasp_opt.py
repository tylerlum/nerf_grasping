"""
Module implementing methods for grasp optimization with NeRFs.
"""
import torch

def grasp_matrix(grasp_points, normals):
    """
    Constructs a grasp matrix for the object represented by the NeRF density,
    evaluated at a set of grasp points.
    Args:
        nerf: NeRF model whose density represents the object.
        grasp_points: a list of grasp points (torch.Tensors in 3D) at which to
            construct the grasp matrix, shape [B, n_f, 3].
    Returns a grasp matrix (torch.Tensor) for the given grasp.
    """
    B, n_f, _ = grasp_points.shape

    normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-5)

    R = rot_from_vec(normals.reshape(-1, 3)).reshape(B, n_f, 3, 3)
    p_cross = skew(grasp_points.reshape(-1,3)).reshape(B, n_f, 3, 3)

    grasp_mats = torch.cat([R, p_cross @ R], dim=-2)
    return torch.cat([grasp_mats[:, ii, :, :] for ii in range(n_f)], dim=-1)

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
    density_grads = torch.autograd.grad(torch.sum(densities), grasp_points,
                                        create_graph=False)[0]

    if ret_densities:
        return density_grads, densities
    else:
        return density_grads

def rot_from_vec(n_z):
    """
    Creates rotation matrix which maps the basis vector e_3 to a vector n_z.
    Gets poorly conditioned when n_z ≅ ±e_3.
    Args:
        n_z: Batch of normal dirs, shape [B, 3].
    """
    # Construct constants.
    n_z = n_z.reshape(-1, 3)
    I = torch.eye(3).reshape(1,3,3).expand(n_z.shape[0], 3, 3)
    e3 = I[:, :, 2]

    # Compute cross product to find axis of rotation.
    v = torch.cross(e3, n_z, dim=-1)
    theta = torch.arccos(torch.sum(e3 * n_z, dim=-1)).reshape(-1, 1, 1)
    K = skew(v)

    ans = I + torch.sin(theta)*K + (1-torch.cos(theta))*K@K

    return ans

def skew(v):
    """
    Returns the skew-symmetric form of a batch of 3D vectors v (shape [B, 3]).
    """
    v = v.reshape(-1,3)

    K = torch.zeros(v.shape[0], 3, 3)

    K[:,0,1] = -v[:, 2]
    K[:,0,2] = v[:, 1]
    K[:,1,0] = v[:, 2]
    K[:,1,2] = -v[:, 0]
    K[:,2,0] = -v[:, 1]
    K[:,2,1] = v[:, 0]

    return K

def msv(grasp_points, normals):
    """
    Evaluates the minimum-singular-value grasp metric proposed in Li and Sastry '88.
    Args:
        grasp_points: a list of grasp points (torch.Tensors in 3D) at which to evaluate
             the grasp metric.
    Returns the minimum singular value of the grasp matrix formed by these points.
    """
    G = grasp_matrix(grasp_points, normals)
    return torch.prod(torch.linalg.svdvals(G), dim=-1)
    # return torch.min(torch.linalg.svdvals(G), dim=-1)[0]

def optimize_cem(cost, mu_0, Sigma_0, num_iters=25, num_samples=250, elite_frac=0.1, constraint=None):
    """
    Implements the cross-entropy method to optimize a given cost function.
    Args:
        cost: a cost function mapping variables x to their cost J(x).
        mu_0: mean of the initial sample distribution.
        Sigma_0: covariance of initial sample distribution.
        num_iters: number of iterations of CEM to run.
        num_samples: number of samples to draw per iteration.
        elite_frac: fraction of samples to use to re-fit sample distribution, in (0, 1).
    """
    n = mu_0.shape[0]
    mu, Sigma = mu_0, Sigma_0
    num_elite = int(elite_frac * num_samples)
    for ii in range(num_iters):
        # Sample points from current distribution.
        if constraint:
            x = rejection_sample(mu, Sigma, constraint, num_samples)
        else:
            x = (mu.reshape(1, n, 1)
                 + torch.linalg.cholesky(Sigma).reshape(1, n, n)
                 @ torch.randn(num_samples, n, 1)).reshape(num_samples, n)

        # Evaluate costs of each point.
        cost_vals = cost(x)

        # Get elite indices.
        _, inds = torch.sort(cost_vals)
        elite_inds = inds[:num_elite]

        # Refit the sample distribution.
        mu = torch.mean(x[elite_inds,:], dim=0)
        residuals = x[elite_inds,:] - mu.reshape(1,n)
        Sigma = (1/(num_elite-1)) * torch.sum(torch.stack(
            [residuals[ii,:][:,None] @ residuals[ii,:][None,:]
             for ii in range(num_elite)], dim=0), dim=0)

    return mu, Sigma

def clip_loss(densities, lb=100, ub=200):
    """
    Helper function, provides "double hinge" loss to encourage fingers
    to lie in a desired density band [lb, ub].
    """
    return torch.mean(torch.maximum(
        torch.zeros_like(densities),
        torch.maximum(lb-densities, densities-ub)), dim=-1)

def grasp_cost(x, nerf, n_f, msv_weight=1e2, grad_weight=1e-2, density_weight=1e1):

    gps = x.reshape(-1, n_f, 3)
    gps.requires_grad=True

    densities, grads = est_grads_vals(nerf, gps)

    msv_loss = msv(gps, grads)
    grad_norms = torch.norm(grads, dim=-1)

    density_loss = clip_loss(densities, lb=500, ub=1000)
    grad_loss = torch.mean(
        torch.maximum(1e5-grad_norms, torch.zeros_like(grad_norms)))

    return density_weight * density_loss - msv_weight * msv_loss + grad_weight * grad_loss

def get_points_cem(nerf,
                   n_f,
                   mu_scale=5e-3,
                   sigma_scale=1e-2,
                   mu_0=None,
                   Sigma_0=None,
                   constraint=None):

    if not mu_0:
        mu_0 = mu_scale * torch.randn(3*n_f)

    if not Sigma_0:
        Sigma_0 = sigma_scale * torch.eye(3*n_f)

    cost = lambda x: grasp_cost(x, nerf, n_f)

    mu_f, Sigma_f = optimize_cem(cost, mu_0, Sigma_0,
                                 num_iters=25, num_samples=500, constraint=constraint)

    return mu_f.reshape(n_f, 3)

