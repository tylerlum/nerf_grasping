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

from nerf_grasping import grasp_utils, mesh_utils, nerf_utils


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


def rot_from_vec(n_z, start_vec=None):
    """
    Creates rotation matrix which maps the basis vector e_3 to a vector n_z.
    Gets poorly conditioned when n_z ≅ ±e_3.
    Args:
        n_z: Batch of normal dirs, shape [B, 3].
    """
    # Construct constants.
    n_z = n_z.reshape(-1, 3)
    Identity = (
        torch.eye(3, device=n_z.device).reshape(1, 3, 3).expand(n_z.shape[0], 3, 3)
    )
    if start_vec is None:
        start_vec = Identity[:, :, 2]

    # Compute cross product to find axis of rotation.
    v = torch.cross(start_vec, n_z, dim=-1)
    theta = torch.arccos(torch.sum(start_vec * n_z, dim=-1)).reshape(-1, 1, 1)
    K = skew(v)

    ans = Identity + torch.sin(theta) * K + (1 - torch.cos(theta)) * K @ K

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


def psv(grasp_points, normals, centroid=0.0):
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


def msv(grasp_points, normals, centroid=0.0):
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


def l1_metric(
    grasp_points_t, normals_t, centroid=None, mu=1.0, num_edges=10, grasp_mask=None
):
    """L1 Grasp quality metric using PyFastGrasp. Assumes object center of mass is at origin"""
    import fastgrasp as fg

    B, _, _ = grasp_points_t.shape

    if grasp_mask is not None:
        valid_inds = torch.argwhere(grasp_mask.reshape(-1)).cpu().numpy().reshape(-1)
    else:
        valid_inds = torch.arange(B)

    device = normals_t.device
    grasp_points = grasp_points_t.detach().cpu().numpy().reshape(-1, 9)[valid_inds, :]
    normals = normals_t.detach().cpu().numpy().reshape(-1, 9)[valid_inds, :]
    if centroid is None:
        centroid = np.zeros((len(grasp_points), 3))
    else:
        centroid = centroid.cpu().detach().numpy()
        centroid = np.stack([centroid for _ in range(len(grasp_points))]).astype(
            "float64"
        )
    grasps = np.concatenate([grasp_points, normals, centroid], axis=1)
    result = np.zeros(len(grasps))
    _ = fg.getLowerBoundsPurgeQHull(grasps, mu, num_edges, result)

    result_full = np.zeros(B)
    result_full[valid_inds] = result
    return torch.tensor(result_full, device=device)


def optimize_cem(
    cost,
    mu_0,
    Sigma_0,
    num_iters=25,
    num_samples=250,
    elite_frac=0.1,
    projection=None,
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
    best_point = None
    best_cost = torch.inf
    for ii in range(num_iters):
        # Sample points from current distribution.
        x = (
            mu.reshape(1, n, 1)
            + torch.linalg.cholesky(Sigma).reshape(1, n, n)
            @ torch.randn(num_samples, n, 1, device=device)
        ).reshape(num_samples, n)

        if projection:
            x = projection(x)

        # Evaluate costs of each point.
        with torch.no_grad():
            cost_vals = cost(x)
        cost_history.append(cost_vals)
        print(
            "minimum cost_val:",
            torch.min(cost_vals),
            "mean cost_val:",
            torch.mean(cost_vals),
        )

        # Get elite indices.
        _, inds = torch.sort(cost_vals)
        elite_inds = inds[:num_elite]

        if cost_vals[inds[0]] < best_cost:
            best_point = x[inds[0]]

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

    return mu, Sigma, cost_history, best_point


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
    centroid=0.0,
    risk_sensitivity=None,
):

    gps = grasp_vars.reshape(-1, n_f, 6)
    B = gps.shape[0]

    if not (isinstance(centroid, float) and centroid == 0.0):
        cost_kwargs["centroid"] = centroid

    if isinstance(model, trimesh.Trimesh):
        grasp_points, grad_ests, grasp_mask = mesh_utils.get_grasp_points(
            model, gps, residual_dirs
        )
        grasp_mask = grasp_mask.all(-1, keepdim=True)
        num_grasps = 1
        risk_sensitivity = None
    else:
        grasp_points, grad_ests, grasp_mask = nerf_utils.sample_grasps(
            gps, num_grasps, model, residual_dirs=residual_dirs, centroid=centroid
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
        cost_kwargs["grasp_mask"] = grasp_mask.expand(B, num_grasps)
        cost_fn = partial(l1_metric, **cost_kwargs)

    g_cost = cost_fn(grasp_points, grad_ests).reshape(B, num_grasps)

    # dists = []
    # for i in range(B):
    #     dists.append(
    #         torch.triu(torch.pairwise_distance(grasp_points[i], grasp_points[i])).sum()
    #     )
    # dists = torch.stack(dists)
    # dists /= dists.max()  # scale from 0 to 0.2
    # g_cost += dists.reshape(B, 1)

    if risk_sensitivity:
        g_cost = torch.exp(-risk_sensitivity * g_cost)
    else:
        g_cost = -g_cost

    g_cost = g_cost.mean(-1)  # shape (B,)

    g_cost = torch.where(
        torch.all(grasp_mask, dim=-1), g_cost, 2.0 * torch.ones_like(g_cost)
    )

    if risk_sensitivity:
        g_cost = (1 / risk_sensitivity) * torch.log(g_cost)

    return g_cost


def get_points_cem(
    n_f,
    model,
    mu_scale=5e-3,
    sigma_scale=1e-3,
    mu_0=None,
    Sigma_0=None,
    num_samples=500,
    num_iters=10,
    projection=None,
    cost_fn="l1",
    residual_dirs=True,
    device="cuda",
    centroid=0.0,
    elite_frac=0.1,
    risk_sensitivity=5.0,
    return_cost_hist=False,
    return_dec_vars=False,
):

    # grasp vars are 2 * 3 * number of fingers, since include both pos and direction
    if mu_0 is None:
        mu_0 = mu_scale * torch.randn(6 * n_f, device=device)

        mu_0 = mu_0.reshape(n_f, 6)
        mu_0[:, :3] = mu_0[:, :3] + centroid
        mu_0 = mu_0.reshape(-1)

    if Sigma_0 is None:
        Sigma_0 = sigma_scale * torch.eye(6 * n_f, device=device)

    def cost(x):
        return grasp_cost(
            x,
            n_f,
            model,
            residual_dirs=residual_dirs,
            cost_fn=cost_fn,
            centroid=centroid,
            risk_sensitivity=risk_sensitivity,
        )

    mu_f, Sigma_f, cost_history, best_point = optimize_cem(
        cost,
        mu_0,
        Sigma_0,
        num_iters=num_iters,
        num_samples=num_samples,
        projection=projection,
        elite_frac=elite_frac,
    )
    ret = [best_point.reshape(n_f, 6)]
    if return_cost_hist:
        ret.append(cost_history)
    if return_dec_vars:
        ret.append(mu_f)
        ret.append(Sigma_f)
    return ret


def dice_the_grasp(
    model,
    cost_fn="psv",
    num_grasps=5000,
    mu=0.5,
    centroid=0.0,
    des_z_dist=0.025,
    projection=None,
):
    """Implements the sampling scheme proposed in Borst et al., '03."""

    face_inds = np.arange(model.triangles_center.shape[0])

    rays_o, rays_d = np.zeros((num_grasps, 3, 3)), np.zeros((num_grasps, 3, 3))
    num_sampled = 0

    lower_corner = np.array([oo[0] for oo in grasp_utils.OBJ_BOUNDS]).reshape(1, 1, 3)
    upper_corner = np.array([oo[1] for oo in grasp_utils.OBJ_BOUNDS]).reshape(1, 1, 3)

    while num_sampled < num_grasps:
        curr_inds = np.random.choice(face_inds, size=(num_grasps, 3), replace=True)
        curr_points = model.triangles_center[curr_inds, :]
        curr_normals = model.face_normals[curr_inds, :]

        # Correct so ray originas are off the mesh.
        curr_points = curr_points + des_z_dist * curr_normals

        # Mask out invalid points.
        grasp_mask = grasp_utils.dicing_rejection_heuristic(curr_normals, mu)
        grasp_mask = grasp_mask * np.all(curr_points >= lower_corner, axis=(-1, -2))
        grasp_mask = grasp_mask * np.all(curr_points <= upper_corner, axis=(-1, -2))

        valid_inds = np.argwhere(grasp_mask)[:, 0]
        num_added = min(num_grasps - num_sampled, valid_inds.shape[0])
        end_slice = num_sampled + num_added

        rays_o[num_sampled:end_slice] = curr_points[valid_inds[:num_added]]
        rays_d[num_sampled:end_slice] = -curr_normals[valid_inds[:num_added]]

        num_sampled += num_added

    # Stack into "grasp var" form.
    grasp_vars = np.concatenate([rays_o, rays_d], axis=-1)
    grasp_vars = torch.from_numpy(grasp_vars).cuda().float().reshape(num_grasps, -1)

    # Finally evaluate all with a desired grasp metric to find the best one.
    costs = grasp_cost(
        grasp_vars,
        3,
        model,
        residual_dirs=False,
        cost_fn=cost_fn,
        centroid=torch.from_numpy(centroid).float().cuda(),
    )

    best_grasp = np.argmin(costs.cpu())

    print(costs.cpu()[best_grasp])

    return rays_o[best_grasp], rays_d[best_grasp]
