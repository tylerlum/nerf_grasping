"""
Module implementing utils for grasping,
including normal estimation and surface detection.
"""
from __future__ import annotations
import nerfstudio
import numpy as np
import pathlib
import pypose as pp
import torch
import transforms3d

from typing import List, Tuple
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.utils import eval_utils
from nerf_grasping.config.fingertip_config import BaseFingertipConfig

from rich.progress import Progress, BarColumn, SpinnerColumn, TextColumn

DEXGRASPNET_TRANS_NAMES = ["WRJTx", "WRJTy", "WRJTz"]
DEXGRASPNET_ROT_NAMES = ["WRJRx", "WRJRy", "WRJRz"]
ALLEGRO_JOINT_NAMES = [
    "joint_0.0",
    "joint_1.0",
    "joint_2.0",
    "joint_3.0",
    "joint_4.0",
    "joint_5.0",
    "joint_6.0",
    "joint_7.0",
    "joint_8.0",
    "joint_9.0",
    "joint_10.0",
    "joint_11.0",
    "joint_12.0",
    "joint_13.0",
    "joint_14.0",
    "joint_15.0",
]


def get_ray_origins_finger_frame_helper(
    num_pts_x: int,
    num_pts_y: int,
    finger_width_mm: float,
    finger_height_mm: float,
    z_offset_mm: float = 0.0,
) -> torch.tensor:
    gripper_finger_width_m = finger_width_mm / 1000.0
    gripper_finger_height_m = finger_height_mm / 1000.0
    z_offset_m = z_offset_mm / 1000.0

    # Create grid of grasp origins in finger frame with shape (num_pts_x, num_pts_y, 3)
    # So that grid_of_points[2, 3] = [x, y, z], where x, y, z are the coordinates of the
    # ray origin for the [2, 3] "pixel" in the finger frame.
    # Origin of transform is at center of xy at z=z_offset_mm
    # x is width, y is height, z is depth
    x_coords = torch.linspace(
        -gripper_finger_width_m / 2, gripper_finger_width_m / 2, num_pts_x
    )
    y_coords = torch.linspace(
        -gripper_finger_height_m / 2, gripper_finger_height_m / 2, num_pts_y
    )

    xx, yy = torch.meshgrid(x_coords, y_coords, indexing="ij")
    zz = torch.zeros_like(xx) + z_offset_m

    assert xx.shape == yy.shape == zz.shape == (num_pts_x, num_pts_y)
    ray_origins = torch.stack([xx, yy, zz], axis=-1)

    assert ray_origins.shape == (num_pts_x, num_pts_y, 3)

    return ray_origins


def get_ray_origins_finger_frame(fingertip_config: BaseFingertipConfig) -> torch.tensor:
    ray_origins_finger_frame = get_ray_origins_finger_frame_helper(
        num_pts_x=fingertip_config.num_pts_x,
        num_pts_y=fingertip_config.num_pts_y,
        finger_width_mm=fingertip_config.finger_width_mm,
        finger_height_mm=fingertip_config.finger_height_mm,
    )
    return ray_origins_finger_frame


def get_ray_bundles(
    ray_origins_finger_frame: torch.tensor,
    transform: pp.LieTensor,
) -> RaySamples:
    num_pts_x, num_pts_y = ray_origins_finger_frame.shape[:2]
    assert ray_origins_finger_frame.shape == (num_pts_x, num_pts_y, 3)
    batch_dims = transform.lshape

    # Device / dtype cast for the transform.
    ray_origins_finger_frame = ray_origins_finger_frame.to(
        device=transform.device, dtype=transform.dtype
    )

    # Add batch dims for transform.
    for _ in range(len(batch_dims)):
        ray_origins_finger_frame = ray_origins_finger_frame.unsqueeze(0)

    # Add batch dims for finger-frame points.
    transform = transform.unsqueeze(-2).unsqueeze(-2)
    assert transform.lshape == (*batch_dims, 1, 1)

    # Apply transform.
    ray_origins_world_frame = (
        transform @ ray_origins_finger_frame
    )  # shape [*batch_dims, num_pts_x, num_pts_y, 3]
    assert ray_origins_world_frame.shape == (*batch_dims, num_pts_x, num_pts_y, 3)

    ray_dirs_finger_frame = torch.tensor(
        [0.0, 0.0, 1.0], device=transform.device, dtype=transform.dtype
    )

    # Expand ray_dirs to add the batch dims.
    for _ in range(len(batch_dims)):
        ray_dirs_finger_frame = ray_dirs_finger_frame.unsqueeze(0)

    # Rotate ray directions (hence SO3 cast).
    ray_dirs_world_frame = (
        transform.rotation() @ ray_dirs_finger_frame
    )  # [*batch_dims, num_pts_x,  num_pts_y, 3]
    assert ray_dirs_world_frame.shape == (*batch_dims, 1, 1, 3)

    # Create dummy pixel areas object.
    pixel_area = (
        torch.ones_like(ray_dirs_world_frame[..., 0]).unsqueeze(-1).float().contiguous()
    )  # [*batch_dims, num_pts_x, num_pts_y, 1]

    ray_bundle = RayBundle(ray_origins_world_frame, ray_dirs_world_frame, pixel_area)

    return ray_bundle


def get_ray_samples(
    ray_origins_finger_frame: torch.tensor,
    transform: pp.LieTensor,
    fingertip_config: BaseFingertipConfig,
):
    ray_bundles = get_ray_bundles(ray_origins_finger_frame, transform)
    grasp_depth_m = fingertip_config.grasp_depth_mm / 1000.0

    # Work out sample lengths.
    sample_dists = torch.linspace(
        0.0,
        grasp_depth_m,
        steps=fingertip_config.num_pts_z,
        dtype=transform.dtype,
        device=transform.device,
    )  # [num_pts_z]

    for _ in range(len(transform.lshape)):
        sample_dists = sample_dists.unsqueeze(0)

    sample_dists = sample_dists.expand(
        *ray_origins_finger_frame.shape[:-1], fingertip_config.num_pts_z
    ).unsqueeze(
        -1
    )  # [*batch_dims, num_pts_x, num_pts_y, num_pts_z, 1]

    # Pull ray samples -- note these are degenerate, i.e., the deltas field is meaningless.
    return ray_bundles.get_ray_samples(sample_dists, sample_dists)


def get_nerf_configs(nerf_checkpoints_path: str) -> List[str]:
    return list(pathlib.Path(nerf_checkpoints_path).rglob("config.yml"))


def load_nerf(cfg_path: pathlib.Path) -> nerfstudio.models.base_model.Model:
    _, pipeline, _, _ = eval_utils.eval_setup(cfg_path, test_mode="inference")

    return pipeline.model.field


def get_hand_config_from_hand_config_dict(
    hand_config_dict: dict,
) -> Tuple[pp.LieTensor, torch.Tensor]:
    qpos = hand_config_dict["qpos"]

    # Get wrist pose.
    wrist_translation = torch.tensor([qpos[tn] for tn in DEXGRASPNET_TRANS_NAMES])
    assert wrist_translation.shape == (3,)

    euler_angles = torch.tensor([qpos[rn] for rn in DEXGRASPNET_ROT_NAMES])
    wrist_quat = torch.tensor(transforms3d.euler.euler2quat(*euler_angles, axes="sxyz"))
    wrist_quat = wrist_quat[[1, 2, 3, 0]]  # Convert (w, x, y, z) -> (x, y, z, w)
    assert wrist_quat.shape == (4,)

    wrist_pose = pp.SE3(torch.cat([wrist_translation, wrist_quat], dim=0))

    # Get joint angles.
    joint_angles = torch.tensor([qpos[jn] for jn in ALLEGRO_JOINT_NAMES])
    assert joint_angles.shape == (16,)

    return wrist_pose, joint_angles


def get_hand_config_dict_from_hand_config(
    wrist_pose: pp.LieTensor,
    joint_angles: torch.Tensor,
) -> dict:
    assert joint_angles.shape == (16,)
    qpos = {}
    for ii, jn in enumerate(ALLEGRO_JOINT_NAMES):
        qpos[jn] = joint_angles[ii].item()

    assert wrist_pose.shape == (7,)
    wrist_translation = wrist_pose.translation().detach().cpu().numpy()
    wrist_quat = wrist_pose.rotation().detach().cpu().numpy()
    assert wrist_translation.shape == (3,)
    assert wrist_quat.shape == (4,)

    wrist_quat = wrist_quat[[3, 0, 1, 2]]  # Convert (x, y, z, w) -> (w, x, y, z)
    euler_angles = transforms3d.euler.quat2euler(wrist_quat, axes="sxyz")
    assert len(euler_angles) == 3
    for ii, rn in enumerate(DEXGRASPNET_ROT_NAMES):
        qpos[rn] = euler_angles[ii]

    for ii, tn in enumerate(DEXGRASPNET_TRANS_NAMES):
        qpos[tn] = wrist_translation[ii]

    return {"qpos": qpos}


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)
