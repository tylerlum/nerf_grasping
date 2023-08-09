import logging
import numpy as np
import torch

OBJ_BOUNDS = [(-0.0625, 0.0625), (0.01, 0.1), (-0.0625, 0.0625)]


def cos_similarity(a, b):
    """
    Returns the cosine similarity of two batches of vectors, a and b.
    """
    return torch.sum(a * b, dim=-1) / (torch.norm(a, dim=-1) * torch.norm(b, dim=-1))


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
        points = points.detach().cpu().numpy()
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


def res_to_true_dirs(rays_o, rays_d, centroid=0.0):
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


def box_projection(x, object_bounds=OBJ_BOUNDS):
    B, _ = x.shape
    x = x.reshape(B, 3, -1)

    lower = torch.tensor([oo[0] for oo in object_bounds]).to(x)
    upper = torch.tensor([oo[1] for oo in object_bounds]).to(x)

    x[..., :3] = x[..., :3].clamp(lower, upper)

    return x.reshape(B, -1)


def dicing_rejection_heuristic(grasp_normals, mu=0.5):
    """Implements the disturbance rejection heuristic from Borst et al., '03.

    Args:
        grasp_normals: set of grasp normals [B, n_f, 3] at contact points.
        mu: (optional) coefficient of friction, default 0.5.

    Returns a boolean mask [B] defining which grasps satisfy the rejection heuristic.
    """
    # Compute heuristic disturbance force proposed in paper -- opposite of mean contact normal.
    F_ext = -np.sum(grasp_normals, axis=1) / grasp_normals.shape[1]

    # Make sure F_ext, grasp_normals are unit vectors.
    F_ext = F_ext / np.linalg.norm(F_ext, axis=-1, keepdims=True)
    grasp_normals = grasp_normals / np.linalg.norm(
        grasp_normals, axis=-1, keepdims=True
    )

    # Reject grasps whose angles between the disturbance direction + all normals is > 90deg + atan(mu).
    valid_fingers = np.arccos(
        np.sum(np.expand_dims(F_ext, 1) * grasp_normals, axis=-1)
    ) > np.pi / 2 + np.arctan(mu)

    return np.any(valid_fingers, axis=-1)
