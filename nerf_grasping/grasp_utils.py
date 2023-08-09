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
