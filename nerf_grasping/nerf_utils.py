"""
Utils for rendering depth / uncertainty images from the NeRF.
"""
from nerf_grasping.config.camera_config import CameraConfig
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.models.base_model import Model
from nerfstudio.data.scene_box import SceneBox
from collections import defaultdict
import numpy as np
import pypose as pp
import torch
from typing import Literal, Dict, Tuple

GRASP_TO_OPENCV = pp.euler2SO3([np.pi, 0, 0]).unsqueeze(0)


def get_cameras(
    grasp_transforms: pp.LieTensor,
    camera_config: CameraConfig,
) -> Cameras:
    """
    Get cameras from the grasp transforms and camera config.
    """
    # Flip rotations so -z points along the grasp dir.
    c2w_rotations = grasp_transforms.rotation() @ GRASP_TO_OPENCV
    # c2w_rotations = grasp_transforms.rotation()
    c2w_translations = grasp_transforms.translation()

    cameras = Cameras(
        camera_to_worlds=torch.cat(
            (c2w_rotations.matrix(), c2w_translations.unsqueeze(-1)), dim=-1
        ),
        camera_type=CameraType.PERSPECTIVE,
        fx=camera_config.fx,
        fy=camera_config.fy,
        cx=camera_config.cx,
        cy=camera_config.cy,
        width=camera_config.W,
        height=camera_config.H,
    )

    return cameras


def render(
    cameras: Cameras,
    nerf_model: Model,
    depth_mode: Literal["median", "expected"] = "expected",
    near_plane: float = 1e-3,
    far_plane: float = 1e-1,
):
    assert len(cameras.shape) == 1
    # TODO: make sure we have enough VRAM to render all the cameras at once.

    ray_bundle = cameras.generate_rays(
        torch.arange(
            cameras.camera_to_worlds.shape[0], device=cameras.camera_to_worlds.device
        ).unsqueeze(-1),
    )
    ray_bundle.nears = torch.ones_like(ray_bundle.pixel_area) * near_plane
    ray_bundle.fars = torch.ones_like(ray_bundle.pixel_area) * far_plane

    return _render_depth_and_uncertainty_for_camera_ray_bundle(
        nerf_model, ray_bundle, depth_mode
    )


def get_ray_samples(
    cameras: Cameras,
    nerf_model: Model,
    near_plane: float = 1e-3,
    far_plane: float = 1e-1,
):
    assert len(cameras.shape) == 1

    ray_bundle = cameras.generate_rays(
        torch.arange(
            cameras.camera_to_worlds.shape[0], device=cameras.camera_to_worlds.device
        ).unsqueeze(-1),
    )

    print(cameras.shape)
    print(ray_bundle.shape)
    ray_bundle = ray_bundle.flatten()
    print(ray_bundle.shape)

    ray_bundle.nears = torch.ones_like(ray_bundle.pixel_area) * near_plane
    ray_bundle.fars = torch.ones_like(ray_bundle.pixel_area) * far_plane

    # Query proposal sampler.
    ray_samples, _, _ = nerf_model.proposal_sampler(
        ray_bundle, density_fns=nerf_model.density_fns
    )

    return ray_samples


def _render_depth_and_uncertainty_for_camera_ray_bundle(
    nerf_model, camera_ray_bundle: RayBundle, depth_mode: Literal["median", "expected"]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Takes in camera parameters and computes the output of the model.

    Args:
        camera_ray_bundle: ray bundle to calculate outputs over
    """
    num_rays_per_chunk = nerf_model.config.eval_num_rays_per_chunk
    image_height, image_width = camera_ray_bundle.origins.shape[:2]
    num_rays = len(camera_ray_bundle)
    outputs_lists = defaultdict(list)
    for i in range(0, num_rays, num_rays_per_chunk):
        start_idx = i
        end_idx = i + num_rays_per_chunk
        ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(
            start_idx, end_idx
        )
        depth, uncertainty = _render_depth_and_uncertainty_for_single_ray_bundle(
            nerf_model, ray_bundle, depth_mode
        )
        outputs_lists["depth"].append(depth)
        outputs_lists["uncertainty"].append(uncertainty)

    return torch.cat(outputs_lists["depth"]).view(
        image_height, image_width, -1
    ), torch.cat(outputs_lists["uncertainty"]).view(image_height, image_width, -1)


def _render_depth_and_uncertainty_for_single_ray_bundle(
    nerf_model: Model, ray_bundle: RayBundle, depth_mode: Literal["median", "expected"]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Slight variation on typical nerf rendering that doesn't render RGB but *does*
    compute the uncertainty along the ray.
    """

    # Query proposal sampler.
    ray_samples, _, _ = nerf_model.proposal_sampler(
        ray_bundle, density_fns=nerf_model.density_fns
    )

    # Query field for density.
    density, _ = nerf_model.field.get_density(ray_samples)

    weights = ray_samples.get_weights(density)

    # Compute the depth image.

    # Idea: compute IQR of depth values, and use that as the uncertainty.

    # Compute the uncertainty variance.
    steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2

    # Render expected depth map (need regardless to compute variance).

    normalized_weights = weights / weights.sum(dim=-2, keepdim=True)
    expected_depth = (normalized_weights * steps).sum(dim=-2)
    expected_depth = torch.clip(expected_depth, steps.min(), steps.max())

    # expected_depth = nerf_model.renderer_expected_depth(
    #     weights=weights, ray_samples=ray_samples
    # )

    depth_variance = (
        normalized_weights * torch.square(steps - expected_depth.unsqueeze(-2))
    ).sum(-2)

    if depth_mode == "median":
        median_depth = nerf_model.renderer_depth(
            weights=weights, ray_samples=ray_samples
        )
        depth = median_depth
    elif depth_mode == "expected":
        depth = expected_depth
    else:
        raise ValueError(f"Invalid depth mode {depth_mode}")

    return depth, depth_variance