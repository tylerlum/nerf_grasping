"""
Module implementing utils for grasping,
including normal estimation and surface detection.
"""
import logging

# import lietorch
import numpy as np
import scipy
import torch

from nerfstudio.cameras.rays import RayBundle, RaySamples

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


# Grid of points in grasp frame (x, y, z)
GRASP_DEPTH_MM = 20
FINGER_WIDTH_MM = 10
FINGER_HEIGHT_MM = 15

# Want points equally spread out in space
DIST_BTWN_PTS_MM = 0.5

# +1 to include both end points
NUM_PTS_X = int(FINGER_WIDTH_MM / DIST_BTWN_PTS_MM) + 1
NUM_PTS_Y = int(FINGER_HEIGHT_MM / DIST_BTWN_PTS_MM) + 1
NUM_PTS_Z = int(GRASP_DEPTH_MM / DIST_BTWN_PTS_MM) + 1

NUM_FINGERS = 4


def get_ray_origins_finger_frame_helper(
    num_pts_x: int,
    num_pts_y: int,
    grasp_depth_mm: float,
    finger_width_mm: float,
    finger_height_mm: float,
) -> np.ndarray:
    grasp_depth_m = grasp_depth_mm / 1000.0
    gripper_finger_width_m = finger_width_mm / 1000.0
    gripper_finger_height_m = finger_height_mm / 1000.0

    # Create grid of grasp origins in finger frame with shape (num_pts_x, num_pts_y, 3)
    # So that grid_of_points[2, 3] = [x, y, z], where x, y, z are the coordinates of the '
    # ray origin for the [2, 3] "pixel" in the finger frame.
    # Origin of transform is at center of xy at 1/4 of the way into the depth z
    # x is width, y is height, z is depth
    x_coords = np.linspace(
        -gripper_finger_width_m / 2, gripper_finger_width_m / 2, num_pts_x
    )
    y_coords = np.linspace(
        -gripper_finger_height_m / 2, gripper_finger_height_m / 2, num_pts_y
    )
    # z_coords = np.linspace(-grasp_depth_m / 4, 3 * grasp_depth_m / 4, num_pts_z)

    xx, yy = np.meshgrid(x_coords, y_coords, indexing="ij")
    zz = -grasp_depth_m / 4 * np.ones_like(xx)

    assert xx.shape == yy.shape == zz.shape == (num_pts_x, num_pts_y)
    ray_origins = np.stack([xx, yy, zz], axis=-1)

    assert ray_origins.shape == (num_pts_x, num_pts_y, 3)

    return ray_origins


def get_ray_origins_finger_frame() -> np.ndarray:
    ray_origins_finger_frame = get_ray_origins_finger_frame_helper(
        num_pts_x=NUM_PTS_X,
        num_pts_y=NUM_PTS_Y,
        grasp_depth_mm=GRASP_DEPTH_MM,
        finger_width_mm=FINGER_WIDTH_MM,
        finger_height_mm=FINGER_HEIGHT_MM,
    )
    return ray_origins_finger_frame


def get_transformed_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    n_points = points.shape[0]
    assert points.shape == (n_points, 3), f"{points.shape}"
    assert transform.shape == (4, 4), f"{transform.shape}"

    extra_ones = np.ones((n_points, 1))
    points_homogeneous = np.concatenate([points, extra_ones], axis=1)

    # First (4, 4) @ (4, N) = (4, N)
    # Then transpose to get (N, 4)
    transformed_points = np.matmul(transform, points_homogeneous.T).T

    transformed_points = transformed_points[:, :3]
    assert transformed_points.shape == (n_points, 3), f"{transformed_points.shape}"
    return transformed_points


def get_transformed_dirs(dirs: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Transforms direction vectors (i.e., doesn't apply the translation of a homologous tranform).
    """
    n_dirs = dirs.shape[0]
    assert dirs.shape == (n_dirs, 3), f"{dirs.shape}"
    assert transform.shape == (4, 4), f"{transform.shape}"

    transformed_dirs = np.matmul(transform[:3, :3], dirs.T).T  # only rotate directions.

    return transformed_dirs


def get_ray_samples(
    ray_origins_finger_frame: np.ndarray,
    transform: np.ndarray,
    num_pts_z=NUM_PTS_Z,
    grasp_depth_mm: float = float(GRASP_DEPTH_MM),
) -> RaySamples:
    grasp_depth_m = grasp_depth_mm / 1000.0

    num_pts_x, num_pts_y = ray_origins_finger_frame.shape[:2]

    assert ray_origins_finger_frame.shape == (num_pts_x, num_pts_y, 3)

    # Collapse batch dimensions + apply transform.
    ray_origins_world_frame = get_transformed_points(
        ray_origins_finger_frame.reshape(-1, 3), transform
    )
    ray_origins_world_frame = torch.tensor(ray_origins_world_frame).float().contiguous()

    ray_dirs_finger_frame = np.array([0.0, 0.0, 1.0]).reshape(
        1, 3
    )  # Ray dirs are along +z axis.
    ray_dirs_world_frame = get_transformed_dirs(ray_dirs_finger_frame, transform)

    # Cast to Tensor + expand to match origins shape.
    ray_dirs_world_frame = (
        torch.tensor(ray_dirs_world_frame)
        .expand(ray_origins_world_frame.shape)
        .float()
        .contiguous()
    )

    # Create dummy pixel areas object.
    pixel_area = (
        torch.ones_like(ray_dirs_world_frame[..., 0]).unsqueeze(-1).float().contiguous()
    )

    ray_bundle = RayBundle(ray_origins_world_frame, ray_dirs_world_frame, pixel_area)

    # Work out sample lengths.
    sample_dists = torch.linspace(0.0, grasp_depth_m, steps=num_pts_z)

    sample_dists = sample_dists.reshape(1, num_pts_z, 1).expand(
        ray_origins_world_frame.shape[0], -1, -1
    )

    # Pull ray samples -- note these are degenerate, i.e., the deltas field is meaningless.
    return ray_bundle.get_ray_samples(sample_dists, sample_dists)
