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

from nerf_grasping import config, grasp_utils, mesh_utils, nerf_utils


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


def psv(grasp_points, normals):
    """
    Evaluates the minimum-singular-value grasp metric proposed in Li and Sastry '88.
    Args:
        grasp_points: a list of grasp points (torch.Tensors in 3D) at which to evaluate
             the grasp metric.
    Returns the product singular value of the grasp matrix formed by these points.
    """
    G = grasp_matrix(grasp_points, normals)
    return torch.prod(torch.linalg.svdvals(G), dim=-1)


def msv(grasp_points, normals):
    """
    Evaluates the minimum-singular-value grasp metric proposed in Li and Sastry '88.
    Args:
        grasp_points: a list of grasp points (torch.Tensors in 3D) at which to evaluate
             the grasp metric.
    Returns the minimum singular value of the grasp matrix formed by these points.
    """
    grasp_points = grasp_points
    G = grasp_matrix(grasp_points, normals)
    return torch.min(torch.linalg.svdvals(G), dim=-1)[0]


def poly_area(grasp_points, normals):
    """Computes the area of the triangle formed by the grasp points."""
    v1 = grasp_points[:, 1] - grasp_points[:, 0]
    v2 = grasp_points[:, 2] - grasp_points[:, 0]
    d1 = torch.norm(v1, dim=-1)
    d2 = torch.norm(v2, dim=-1)

    # bad_inds = torch.argwhere(torch.all(grasp_points[:, 0] == grasp_points[:, 1], dim=-1))
    # print('bad indices, inside_cost: ', bad_inds)

    # bad_inds = torch.argwhere(d1 == 0)
    # print(grasp_points[bad_inds, 1], grasp_points[bad_inds, 0])

    angles = torch.sqrt(1 - torch.square(torch.sum(v1 * v2, dim=-1) / (d1 * d2)))
    areas = 0.5 * d1 * d2 * torch.sin(angles)
    return areas


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


def get_cost_function(exp_config, model):
    """Factory for grasp cost function; generates grasp cost for CEM using config/model."""

    if isinstance(model.ig_centroid, torch.Tensor):
        centroid = model.ig_centroid
    else:
        centroid = torch.from_numpy(model.ig_centroid)

    def cost_function(grasp_vars):

        # Reshape grasp vars into something useful, get dims.
        n_f = exp_config.robot_config.num_fingers
        gps = grasp_vars.reshape(-1, n_f, 6)
        B = gps.shape[0]

        # If model is a triangle mesh, extract points/normals from it.
        if isinstance(model, trimesh.Trimesh):
            grasp_points, grad_ests, grasp_mask = mesh_utils.get_grasp_points(
                model, gps, not exp_config.dice_grasp
            )
            grasp_mask = grasp_mask.all(-1, keepdim=True)

            risk_sensitivity = None
            num_grasp_samples = 1

        # Otherwise, use fuzzy NeRF method for point/normals.
        else:

            risk_sensitivity = exp_config.risk_sensitivity
            if exp_config.model_config.expected_surface:
                num_grasp_samples = 1
            else:
                num_grasp_samples = exp_config.num_grasp_samples

            grasp_points, grad_ests, grasp_mask = nerf_utils.sample_grasps(
                gps, num_grasp_samples, model, exp_config.model_config
            )

            # print('grasp_points [B, n_f, num_grasps, 3]: ', grasp_points.shape)
            # print('grad_ests [B, n_f, num_grasps, 3]: ', grad_ests.shape)

        # Reshape grasp points and grads for cost evaluation.
        grasp_points = grasp_points.reshape(-1, n_f, 3)  # [B * num_grasps, n_f, 3]
        grad_ests = grad_ests.reshape(-1, n_f, 3)  # [B * num_grasps, n_f, 3]

        #         bad_inds = torch.argwhere(torch.all(grasp_points[:, 0] == grasp_points[:, 1], dim=-1))
        #         print('bad indices: ', bad_inds)
        #         print(torch.all(gps[torch.floor(bad_inds/10).long(), 0] == gps[torch.floor(bad_inds/10).long(), 1], dim=-1))

        #         print(grasp_points[bad_inds, 0]-grasp_points[bad_inds, 1])

        #         print(grasp_points.shape, gps.shape)

        # Center grasp_points around centroid.
        grasp_points_centered = grasp_points - centroid.reshape(1, 1, 3)

        # bad_inds = torch.argwhere(torch.all(grasp_points_centered[:, 0] == grasp_points_centered[:, 1], dim=-1))
        # print('bad indices, centered: ', bad_inds)

        # Switch-case for cost function.
        if exp_config.cost_function == config.CostType.PSV:
            grasp_metric = partial(psv)
        elif exp_config.cost_function == config.CostType.MSV:
            grasp_metric = partial(msv)
        elif exp_config.cost_function == config.CostType.POLY_AREA:
            grasp_metric = poly_area
        elif exp_config.cost_function == config.CostType.FC:
            grasp_metric = ferrari_canny
        elif exp_config.cost_function == config.CostType.L1:
            grasp_metric = partial(
                l1_metric, grasp_mask=grasp_mask.expand(B, num_grasp_samples)
            )

        raw_cost = grasp_metric(grasp_points_centered, grad_ests).reshape(
            B, num_grasp_samples
        )

        # Exponentiate cost if using risk sensitivity.
        if risk_sensitivity:
            g_cost = torch.exp(-risk_sensitivity * raw_cost)
        else:
            g_cost = -raw_cost

        # Take expectation along sample dim.
        g_cost = g_cost.mean(-1)  # shape (B,)

        # Set invalid grasp costs to an upper bound (here, 2.0).
        g_cost = torch.where(
            torch.all(grasp_mask, dim=-1), g_cost, 2.0 * torch.ones_like(g_cost)
        )

        if risk_sensitivity:
            g_cost = (1 / risk_sensitivity) * torch.log(g_cost)

        return g_cost

    return cost_function


def dice_the_grasp(
    model,
    cost_function,
    exp_config,
    projection=None,
):
    """Implements the sampling scheme proposed in Borst et al., '03."""

    num_grasps = exp_config.cem_num_iters * exp_config.cem_num_samples

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
        curr_points = curr_points + exp_config.model_config.des_z_dist * curr_normals

        # Mask out invalid points.
        grasp_mask = grasp_utils.dicing_rejection_heuristic(
            curr_normals, exp_config.dice_mu
        )
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
    grasp_vars = (
        torch.from_numpy(grasp_vars)
        .cuda()
        .float()
        .reshape(num_grasps, exp_config.robot_config.num_fingers, -1)
    )

    # Finally evaluate all with a desired grasp metric to find the best one.
    costs = cost_function(grasp_vars)

    best_grasp = np.argmin(costs.cpu())

    return rays_o[best_grasp], rays_d[best_grasp]
