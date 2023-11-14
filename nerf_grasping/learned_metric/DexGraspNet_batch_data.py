from __future__ import annotations
import functools
from dataclasses import dataclass
import pypose as pp
import torch
from typing import List, Optional, Union
import numpy as np
from nerf_grasping.grasp_utils import (
    get_ray_samples,
)
from nerf_grasping.grasp_utils import (
    get_ray_origins_finger_frame,
)
from nerf_grasping.config.fingertip_config import (
    BaseFingertipConfig,
    EvenlySpacedFingertipConfig,
)

NUM_XYZ = 3


@functools.lru_cache()
def get_ray_origins_finger_frame_cached(cfg: BaseFingertipConfig) -> torch.Tensor:
    ray_origins_finger_frame = get_ray_origins_finger_frame(cfg)
    return ray_origins_finger_frame


@dataclass
class BatchDataInput:
    nerf_densities: torch.Tensor
    grasp_transforms: pp.LieTensor
    fingertip_config: BaseFingertipConfig  # have to take this because all these shape checks used to use hardcoded constants.
    random_rotate_transform: Optional[pp.LieTensor] = None
    conditioning_var: Optional[
        torch.Tensor
    ] = None  # Optional conditioning var for the classifier. This will get passed if not None, otherwise pass grasp_transforms.
    nerf_density_threshold_value: Optional[float] = None

    def to(self, device) -> BatchDataInput:
        self.nerf_densities = self.nerf_densities.to(device)
        self.grasp_transforms = self.grasp_transforms.to(device)
        self.random_rotate_transform = (
            self.random_rotate_transform.to(device=device)
            if self.random_rotate_transform is not None
            else None
        )
        if self.conditioning_var is not None:
            self.conditioning_var = self.conditioning_var.to(device)
        return self

    @property
    def nerf_alphas(self) -> torch.Tensor:
        # alpha = 1 - exp(-delta * sigma)
        #       = probability of collision within this segment starting from beginning of segment
        delta = (
            self.fingertip_config.grasp_depth_mm
            / (self.fingertip_config.num_pts_z - 1)
            / 1000
        )
        if isinstance(self.fingertip_config, EvenlySpacedFingertipConfig):
            assert delta == self.fingertip_config.distance_between_pts_mm / 1000
        alphas = 1.0 - torch.exp(-delta * self.nerf_densities)

        if self.nerf_density_threshold_value is not None:
            alphas = torch.where(
                self.nerf_densities > self.nerf_density_threshold_value,
                torch.ones_like(alphas),
                torch.zeros_like(alphas),
            )

        return alphas

    @property
    def coords(self) -> torch.Tensor:
        return self._coords_helper(self.grasp_transforms)

    @property
    def augmented_coords(self) -> torch.Tensor:
        return self._coords_helper(self.augmented_grasp_transforms)

    @property
    def nerf_alphas_with_coords(self) -> torch.Tensor:
        return self._nerf_alphas_with_coords_helper(self.coords)

    @property
    def nerf_alphas_with_augmented_coords(self) -> torch.Tensor:
        return self._nerf_alphas_with_coords_helper(self.augmented_coords)

    @property
    def augmented_grasp_transforms(self) -> torch.Tensor:
        if self.random_rotate_transform is None:
            return self.grasp_transforms

        # Unsqueeze because we're applying the same (single) random rotation to all fingers.
        return_value = (
            self.random_rotate_transform.unsqueeze(dim=1) @ self.grasp_transforms
        )
        assert (
            return_value.lshape
            == self.grasp_transforms.lshape
            == (self.batch_size, self.fingertip_config.n_fingers)
        )
        return return_value

    @property
    def batch_size(self) -> int:
        return self.nerf_densities.shape[0]

    @property
    def device(self) -> torch.device:
        return self.nerf_densities.device

    def _coords_helper(self, grasp_transforms: pp.LieTensor) -> torch.Tensor:
        assert grasp_transforms.lshape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
        )
        ray_origins_finger_frame = get_ray_origins_finger_frame_cached(
            self.fingertip_config
        )

        all_query_points = get_ray_samples(
            ray_origins_finger_frame.to(
                device=grasp_transforms.device, dtype=grasp_transforms.dtype
            ),
            grasp_transforms,
            self.fingertip_config,
        ).frustums.get_positions()

        assert all_query_points.shape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
            NUM_XYZ,
        )
        all_query_points = all_query_points.permute(0, 1, 5, 2, 3, 4)
        assert all_query_points.shape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
            NUM_XYZ,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )
        return all_query_points

    def _nerf_alphas_with_coords_helper(self, coords: torch.Tensor) -> torch.Tensor:
        assert coords.shape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
            NUM_XYZ,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )
        reshaped_nerf_alphas = self.nerf_alphas.reshape(
            self.batch_size,
            self.fingertip_config.n_fingers,
            1,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )
        return_value = torch.cat(
            [
                reshaped_nerf_alphas,
                coords,
            ],
            dim=2,
        )
        assert return_value.shape == (
            self.batch_size,
            self.fingertip_config.n_fingers,
            NUM_XYZ + 1,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )
        return return_value

@dataclass
class DepthImageBatchDataInput:
    depth_uncertainty_images: torch.Tensor
    grasp_transforms: pp.LieTensor
    fingertip_config: BaseFingertipConfig  # have to take this because all these shape checks used to use hardcoded constants.
    random_rotate_transform: Optional[pp.LieTensor] = None
    conditioning_var: Optional[
        torch.Tensor
    ] = None  # Optional conditioning var for the classifier. This will get passed if not None, otherwise pass grasp_transforms.
    nerf_density_threshold_value: Optional[float] = None

    def to(self, device) -> BatchDataInput:
        self.depth_uncertainty_images = self.depth_uncertainty_images.to(device)
        self.grasp_transforms = self.grasp_transforms.to(device)
        self.random_rotate_transform = (
            self.random_rotate_transform.to(device=device)
            if self.random_rotate_transform is not None
            else None
        )
        if self.conditioning_var is not None:
            self.conditioning_var = self.conditioning_var.to(device)
        return self

    @property
    def augmented_grasp_transforms(self) -> torch.Tensor:
        if self.random_rotate_transform is None:
            return self.grasp_transforms

        # Unsqueeze because we're applying the same (single) random rotation to all fingers.
        return_value = (
            self.random_rotate_transform.unsqueeze(dim=1) @ self.grasp_transforms
        )
        assert (
            return_value.lshape
            == self.grasp_transforms.lshape
            == (self.batch_size, self.fingertip_config.n_fingers)
        )
        return return_value

    @property
    def batch_size(self) -> int:
        return self.depth_uncertainty_images.shape[0]

    @property
    def device(self) -> torch.device:
        return self.depth_uncertainty_images.device


@dataclass
class BatchDataOutput:
    passed_simulation: torch.Tensor
    passed_penetration_threshold: torch.Tensor
    passed_eval: torch.Tensor

    def to(self, device) -> BatchDataOutput:
        self.passed_simulation = self.passed_simulation.to(device)
        self.passed_penetration_threshold = self.passed_penetration_threshold.to(device)
        self.passed_eval = self.passed_eval.to(device)
        return self

    @property
    def batch_size(self) -> int:
        return self.passed_eval.shape[0]

    @property
    def device(self) -> torch.device:
        return self.passed_eval.device


@dataclass
class BatchData:
    input: Union[BatchDataInput, DepthImageBatchDataInput]
    output: BatchDataOutput
    nerf_config: List[str]

    def to(self, device) -> BatchData:
        self.input = self.input.to(device)
        self.output = self.output.to(device)
        return self

    @property
    def batch_size(self) -> int:
        return self.output.batch_size

    @property
    def device(self) -> torch.device:
        return self.output.device
