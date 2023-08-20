from __future__ import annotations
import functools
from dataclasses import dataclass
import pypose as pp
import torch
from typing import List, Optional
import numpy as np
from nerf_grasping.dataset.DexGraspNet_NeRF_Grasps_utils import (
    get_ray_samples,
)
from nerf_grasping.grasp_utils import (
    DIST_BTWN_PTS_MM,
    GRASP_DEPTH_MM,
    NUM_PTS_X,
    NUM_PTS_Y,
    NUM_PTS_Z,
    NUM_FINGERS,
    get_ray_origins_finger_frame,
)

NUM_XYZ = 3


@functools.lru_cache()
def get_ray_origins_finger_frame_cached():
    ray_origins_finger_frame = get_ray_origins_finger_frame()
    return ray_origins_finger_frame


@dataclass
class BatchData:
    nerf_densities: torch.Tensor
    grasp_success: torch.Tensor
    grasp_transforms: pp.LieTensor
    nerf_config: List[str]
    random_rotate_transform: Optional[pp.LieTensor] = None

    def to(self, device) -> BatchData:
        self.nerf_densities = self.nerf_densities.to(device)
        self.grasp_success = self.grasp_success.to(device)
        self.grasp_transforms = self.grasp_transforms.to(device)
        self.random_rotate_transform = (
            self.random_rotate_transform.to(device=device)
            if self.random_rotate_transform is not None
            else None
        )
        return self

    @classmethod
    def create_input_only_batch(
        cls, nerf_densities: torch.Tensor, grasp_transforms: pp.LieTensor
    ) -> BatchData:
        # This is a hacky way to create a BatchData object with only the input data
        # This way, won't have to handle Optional/None properties
        # Better solution: create a BatchDataInput class that BatchData stores
        batch_size = nerf_densities.shape[0]
        dummy_grasp_success = torch.zeros(
            batch_size, dtype=nerf_densities.dtype, device=nerf_densities.device
        )
        dummy_nerf_config = ["dummy" for _ in range(batch_size)]
        batch_data = BatchData(
            nerf_densities=nerf_densities,
            grasp_success=dummy_grasp_success,
            grasp_transforms=grasp_transforms,
            nerf_config=dummy_nerf_config,
        ).to(nerf_densities.device)
        return batch_data

    @property
    def nerf_alphas(self) -> torch.Tensor:
        # alpha = 1 - exp(-delta * sigma)
        #       = probability of collision within this segment starting from beginning of segment
        DELTA = DIST_BTWN_PTS_MM / 1000
        OTHER_DELTA = GRASP_DEPTH_MM / (NUM_PTS_Z - 1) / 1000
        assert np.isclose(DELTA, OTHER_DELTA), f"{DELTA} != {OTHER_DELTA}"
        return 1.0 - torch.exp(-DELTA * self.nerf_densities)

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
            == (self.batch_size, NUM_FINGERS)
        )
        return return_value

    @property
    def batch_size(self) -> int:
        return self.grasp_success.shape[0]

    @property
    def device(self) -> torch.device:
        return self.grasp_success.device

    def _coords_helper(self, grasp_transforms: pp.LieTensor) -> torch.Tensor:
        assert grasp_transforms.lshape == (self.batch_size, NUM_FINGERS)
        ray_origins_finger_frame = get_ray_origins_finger_frame_cached()

        all_query_points = get_ray_samples(
            ray_origins_finger_frame.to(
                device=grasp_transforms.device, dtype=grasp_transforms.dtype
            ),
            grasp_transforms,
        ).frustums.get_positions()

        assert all_query_points.shape == (
            self.batch_size,
            NUM_FINGERS,
            NUM_PTS_X,
            NUM_PTS_Y,
            NUM_PTS_Z,
            NUM_XYZ,
        )
        all_query_points = all_query_points.permute(0, 1, 5, 2, 3, 4)
        assert all_query_points.shape == (
            self.batch_size,
            NUM_FINGERS,
            NUM_XYZ,
            NUM_PTS_X,
            NUM_PTS_Y,
            NUM_PTS_Z,
        )
        return all_query_points

    def _nerf_alphas_with_coords_helper(self, coords: torch.Tensor) -> torch.Tensor:
        assert coords.shape == (
            self.batch_size,
            NUM_FINGERS,
            NUM_XYZ,
            NUM_PTS_X,
            NUM_PTS_Y,
            NUM_PTS_Z,
        )
        reshaped_nerf_alphas = self.nerf_alphas.reshape(
            self.batch_size, NUM_FINGERS, 1, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z
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
            NUM_FINGERS,
            NUM_XYZ + 1,
            NUM_PTS_X,
            NUM_PTS_Y,
            NUM_PTS_Z,
        )
        return return_value
