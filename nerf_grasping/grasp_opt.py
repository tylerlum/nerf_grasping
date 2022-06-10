"""
Module implementing methods for grasp optimization with NeRFs.
"""
import logging
import sys
from functools import partial

import cvxopt as cvx
import numpy as np
import torch
from pyhull import convex_hull as cvh

import trimesh

from nerf_grasping import grasp_utils, mesh_utils


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
    p_cross = skew(grasp_points.reshape(-1, 3)).reshape(B, n_f, 3, 3)

    grasp_mats = torch.cat([R, p_cross @ R], dim=-2)
    return torch.cat([grasp_mats[:, ii, :, :] for ii in range(n_f)], dim=-1)


def rot_from_vec(n_z):
    """
    Creates rotation matrix which maps the basis vector e_3 to a vector n_z.
    Gets poorly conditioned when n_z ≅ ±e_3.
    Args:
        n_z: Batch of normal dirs, shape [B, 3].
    """
    # Construct constants.
    n_z = n_z.reshape(-1, 3)
    I = torch.eye(3, device=n_z.device).reshape(1, 3, 3).expand(n_z.shape[0], 3, 3)
    e3 = I[:, :, 2]

    # Compute cross product to find axis of rotation.
    v = torch.cross(e3, n_z, dim=-1)
    theta = torch.arccos(torch.sum(e3 * n_z, dim=-1)).reshape(-1, 1, 1)
    K = skew(v)

    ans = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * K @ K

    return ans


def skew(v):
    """
    Returns the skew-symmetric form of a batch of 3D vectors v (shape [B, 3]).
    """
    v = v.reshape(-1, 3)

    K = torch.zeros(v.shape[0], 3, 3, device=v.device)

    K[:, 0, 1] = -v[:, 2]
    K[:, 0, 2] = v[:, 1]
    K[:, 1, 0] = v[:, 2]
    K[:, 1, 2] = -v[:, 0]
    K[:, 2, 0] = -v[:, 1]
    K[:, 2, 1] = v[:, 0]

    return K


def psv(grasp_points, normals):
    """
    Evaluates the minimum-singular-value grasp metric proposed in Li and Sastry '88.
    Args:
        grasp_points: a list of grasp points (torch.Tensors in 3D) at which to evaluate
             the grasp metric.
    Returns the product singular value of the grasp matrix formed by these points.
    """
    grasp_points = grasp_points - centroid
    G = grasp_matrix(grasp_points, normals)
    return torch.prod(torch.linalg.svdvals(G), dim=-1)


def msv(grasp_points, normals, centroid=0.):
    """
    Evaluates the minimum-singular-value grasp metric proposed in Li and Sastry '88.
    Args:
        grasp_points: a list of grasp points (torch.Tensors in 3D) at which to evaluate
             the grasp metric.
    Returns the minimum singular value of the grasp matrix formed by these points.
    """
    grasp_points = grasp_points - centroid
    G = grasp_matrix(grasp_points, normals)
    return torch.min(torch.linalg.svdvals(G), dim=-1)[0]


def ferrari_canny(grasp_points, normals):
    """Calculates Ferrari Canny L1 grasp metric"""
    G = grasp_matrix(grasp_points, normals)
    if len(G.size()) == 3:
        dists = []
        for g in G:
            dists.append(fc(g.cpu().numpy()))
        dist = torch.as_tensor(dists).float()
    else:
        dist = fc(G)
    return dist


def fc(G):
    hull = cvh.ConvexHull(G.T)
    min_norm_in_hull, v = min_norm_vector_in_facet(G)
    if len(hull.vertices) == 0:
        logging.debug("Convex hull could not be computed")
        return 0.0

    if min_norm_in_hull > 1e-3:
        logging.debug("Min norm in hull not negative")
        return 0.0

    if np.sum(v > 1e-4) <= G.shape[0] - 1:
        logging.debug("Zero not in interior of convex hull")
        return 0.0

    min_dist = sys.float_info.max
    closest_facet = None
    for v in hull.vertices:
        if (
            np.max(np.array(v)) < G.shape[1]
        ):  # because of some occasional odd behavior from pyhull
            facet = G[:, v]
            dist, _ = min_norm_vector_in_facet(facet)
            if dist < min_dist:
                min_dist = dist
                closest_facet = v
    del closest_facet
    return min_dist


def min_norm_vector_in_facet(facet, wrench_regularizer=1e-10):
    """Finds the minimum norm point in the convex hull of a given facet (aka simplex) by solving a QP.
    Parameters
    ----------
    facet : 6xN :obj:`numpy.ndarray`
        vectors forming the facet
    wrench_regularizer : float
        small float to make quadratic program positive semidefinite
    Returns
    -------
    float
        minimum norm of any point in the convex hull of the facet
    Nx1 :obj:`numpy.ndarray`
        vector of coefficients that achieves the minimum
    """
    dim = facet.shape[1]  # num vertices in facet

    # create alpha weights for vertices of facet
    G = facet.T @ facet
    grasp_matrix = G + wrench_regularizer * np.eye(G.shape[0])

    # Solve QP to minimize .5 x'Px + q'x subject to Gx <= h, Ax = b

    P = cvx.matrix(2 * grasp_matrix)  # quadratic cost for Euclidean dist
    q = cvx.matrix(np.zeros((dim, 1)))
    G = cvx.matrix(-np.eye(dim))  # greater than zero constraint
    h = cvx.matrix(np.zeros((dim, 1)))
    A = cvx.matrix(np.ones((1, dim)))  # sum constraint to enforce convex
    b = cvx.matrix(np.ones(1))  # combinations of vertices

    cvx.solvers.options["show_progress"] = False
    sol = cvx.solvers.qp(P, q, G, h, A, b)
    v = np.array(sol["x"])
    min_norm = np.sqrt(sol["primal objective"])

    return abs(min_norm), v


def l1_metric(grasp_points_t, normals_t, centroid=None, mu=1.0, num_edges=10):
    """L1 Grasp quality metric using PyFastGrasp. Assumes object center of mass is at origin"""
    import fastgrasp as fg

    device = normals_t.device
    grasp_points = grasp_points_t.detach().cpu().numpy().reshape(-1, 9)
    normals = normals_t.detach().cpu().numpy().reshape(-1, 9)
    centroid = np.zeros((len(grasp_points), 3))
    grasps = np.concatenate([grasp_points, normals, centroid], axis=1)
    result = np.zeros(len(grasps))
    _ = fg.getLowerBoundsPurgeQHull(grasps, mu, num_edges, result)
    return torch.tensor(result, device=device)


def optimize_cem(
    cost,
    mu_0,
    Sigma_0,
    num_iters=25,
    num_samples=250,
    elite_frac=0.1,
    constraint=None,
):
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
    device = mu_0.device
    cost_history = []
    for ii in range(num_iters):
        # Sample points from current distribution.
        if constraint:
            x = grasp_utils.rejection_sample(mu, Sigma, constraint, num_samples)
        else:
            x = (
                mu.reshape(1, n, 1)
                + torch.linalg.cholesky(Sigma).reshape(1, n, n)
                @ torch.randn(num_samples, n, 1, device=device)
            ).reshape(num_samples, n)

        # Evaluate costs of each point.
        cost_vals = cost(x)
        cost_history.append(cost_vals)
        print(torch.min(cost_vals), torch.mean(cost_vals))

        # Get elite indices.
        _, inds = torch.sort(cost_vals)
        elite_inds = inds[:num_elite]

        # Refit the sample distribution.
        mu = torch.mean(x[elite_inds, :], dim=0)
        residuals = x[elite_inds, :] - mu.reshape(1, n)
        Sigma = (1 / (num_elite - 1)) * torch.sum(
            torch.stack(
                [
                    residuals[ii, :][:, None] @ residuals[ii, :][None, :]
                    for ii in range(num_elite)
                ],
                dim=0,
            ),
            dim=0,
        ) + 1e-8 * torch.eye(n, device=device)

    return mu, Sigma, cost_history


def clip_loss(densities, lb=100, ub=200):
    """
    Helper function, provides "double hinge" loss to encourage fingers
    to lie in a desired density band [lb, ub].
    """
    return torch.mean(
        torch.maximum(
            torch.zeros_like(densities), torch.maximum(lb - densities, densities - ub)
        ),
        dim=-1,
    )


def grasp_cost(
    grasp_vars,
    n_f,
    model,
    num_grasps=10,
    residual_dirs=True,
    cost_fn="l1",
    cost_kwargs=dict(centroid=np.zeros((3, 1))),
    centroid=0.,
    risk_sensitivity=None,
):

    gps = grasp_vars.reshape(-1, n_f, 6)
    B = gps.shape[0]

    if centroid is not 0.:
        cost_kwargs['centroid'] = centroid

    if isinstance(model, trimesh.Trimesh):
        grasp_points, grad_ests, grasp_mask = mesh_utils.get_grasp_points(model, gps, residual_dirs)
        num_grasps = 1
        risk_sensitivity=None
    else:
        # TODO(pculbert): push centroid logic through code again.
        grasp_points, grad_ests, grasp_mask = grasp_utils.sample_grasps(
            gps, num_grasps, model, residual_dirs=residual_dirs
        )

    # Reshape grasp points and grads for msv evaluation.
    grasp_points = grasp_points.reshape(-1, n_f, 3)
    grad_ests = grad_ests.reshape(-1, n_f, 3)

    # Switch-case for cost function.
    if cost_fn == "psv":
        cost_fn = partial(psv, **cost_kwargs)
    elif cost_fn == "msv":
        cost_fn = partial(msv, **cost_kwargs)
    elif cost_fn == "fc":
        cost_fn = ferrari_canny
    elif cost_fn == "l1":
        cost_fn = partial(l1_metric, **cost_kwargs)

    g_cost = torch.mean(cost_fn(grasp_points, grad_ests).reshape(B, num_grasps), dim=-1)

    g_cost = torch.where(
        torch.all(grasp_mask, dim=-1), g_cost, -torch.ones_like(g_cost)
    )

    if not risk_sensitivity:
        return -g_cost
    else:
        return torch.exp(-risk_sensitivity * g_cost)

def get_points_cem(
    n_f,
    model,
    mu_scale=5e-3,
    sigma_scale=1e-3,
    mu_0=None,
    Sigma_0=None,
    num_samples=500,
    num_iters=10,
    constraint=None,
    cost_fn="msv",
    residual_dirs=True,
    device="cuda",
    centroid=0.,
    risk_sensitivity=1.,
):

    # grasp vars are 2 * 3 * number of fingers, since include both pos and direction
    if mu_0 is None:
        mu_0 = mu_scale * torch.randn(6 * n_f, device=device)

    mu_0 = mu_0.reshape(n_f, 6)
    mu_0[:, :3] = mu_0[:, :3] + centroid
    mu_0 = mu_0.reshape(-1)

    if Sigma_0 is None:
        Sigma_0 = sigma_scale * torch.eye(6 * n_f, device=device)

    cost = lambda x: grasp_cost(
        x, n_f, model, residual_dirs=residual_dirs, cost_fn=cost_fn, centroid=centroid, risk_sensitivity=risk_sensitivity
    )

    mu_f, Sigma_f, cost_history = optimize_cem(
        cost, mu_0, Sigma_0, num_iters=num_iters, num_samples=num_samples, constraint=constraint
    )

    return mu_f.reshape(n_f, 6)
