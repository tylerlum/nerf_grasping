"""
Utils for rendering depth / uncertainty images from the NeRF.
"""
from nerf_grasping.config.camera_config import CameraConfig
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.models.base_model import Model
import pypose as pp
import torch
from typing import Literal


def get_cameras(grasp_transforms: pp.LieTensor, camera_config: CameraConfig) -> Cameras:
    """
    Get cameras from the grasp transforms and camera config.
    """
    return Cameras(
        camera_to_worlds=grasp_transforms.matrix()[..., :3, :4],
        camera_type=CameraType.PERSPECTIVE,
        fx=camera_config.fx,
        fy=camera_config.fy,
        cx=camera_config.cx,
        cy=camera_config.cy,
        width=camera_config.W,
        height=camera_config.H,
    )


def render(
    cameras: Cameras,
    nerf_model: Model,
    depth_mode: Literal["median", "expected"] = "median",
):
    breakpoint()
    # TODO: make sure we have enough VRAM to render all the cameras at once.
    ray_bundle = cameras.generate_rays(
        torch.arange(
            cameras.camera_to_worlds.shape[0], device=cameras.camera_to_worlds.device
        )
    )

    return _render_depth_and_uncertainty_images(nerf_model, ray_bundle, depth_mode)


def _render_depth_and_uncertainty_images(
    nerf_model: Model, ray_bundle: RayBundle, depth_mode: Literal["median", "expected"]
):
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
    if depth_mode == "median":
        median_depth = nerf_model.renderer_depth(
            weights=weights, ray_samples=ray_samples
        )

    # Render expected depth map (need regardless to compute variance).
    expected_depth = nerf_model.renderer_expected_depth(
        weights=weights, ray_samples=ray_samples
    )

    # Idea: compute IQR of depth values, and use that as the uncertainty.

    # Compute the uncertainty variance.
    steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2

    depth_variance = weights * (steps - expected_depth) ** 2

    depth = median_depth if depth_mode == "median" else expected_depth

    return depth, depth_variance
