from __future__ import annotations
import functools
from dataclasses import dataclass
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
    grasp_transforms: torch.Tensor
    nerf_config: List[str]
    random_rotate_transform: Optional[torch.Tensor] = None

    def to(self, device) -> BatchData:
        self.nerf_densities = self.nerf_densities.to(device)
        self.grasp_success = self.grasp_success.to(device)
        self.grasp_transforms = self.grasp_transforms.to(device)
        self.random_rotate_transform = (
            self.random_rotate_transform.to(device)
            if self.random_rotate_transform is not None
            else None
        )
        return self

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

    def _coords_helper(self, grasp_transforms: torch.Tensor) -> torch.Tensor:
        assert grasp_transforms.shape == (self.batch_size, NUM_FINGERS, 4, 4)
        ray_origins_finger_frame = get_ray_origins_finger_frame_cached()

        # TODO: Change this to not be np and be vectorized
        all_query_points = []
        for i in range(self.batch_size):
            transforms = grasp_transforms[i]
            intermediate_query_points_list = []

            # TODO: SLOW?
            for j in range(NUM_FINGERS):
                transform = transforms[j]
                ray_samples = get_ray_samples(
                    ray_origins_finger_frame, transform.cpu().numpy()
                )
                query_points = np.copy(
                    ray_samples.frustums.get_positions().cpu().numpy().reshape(-1, 3)
                )
                intermediate_query_points_list.append(query_points)
            intermediate_query_points = np.stack(intermediate_query_points_list, axis=0)

            all_query_points.append(torch.from_numpy(intermediate_query_points))
        all_query_points = torch.stack(all_query_points, dim=0).float().to(self.device)
        assert all_query_points.shape == (
            self.batch_size,
            NUM_FINGERS,
            NUM_PTS_X * NUM_PTS_Y * NUM_PTS_Z,
            NUM_XYZ,
        )
        all_query_points = all_query_points.reshape(
            self.batch_size, NUM_FINGERS, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z, NUM_XYZ
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

    @property
    def nerf_alphas_with_coords(self) -> torch.Tensor:
        return self._nerf_alphas_with_coords_helper(self.coords)

    @property
    def nerf_alphas_with_augmented_coords(self) -> torch.Tensor:
        return self._nerf_alphas_with_coords_helper(self.augmented_coords)

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

    @property
    def augmented_grasp_transforms(self) -> torch.Tensor:
        if self.random_rotate_transform is None:
            return self.grasp_transforms

        return_value = torch.matmul(
            self.random_rotate_transform.unsqueeze(dim=1),
            self.grasp_transforms,
        )
        assert (
            return_value.shape
            == self.grasp_transforms.shape
            == (self.batch_size, NUM_FINGERS, 4, 4)
        )
        return return_value

    @property
    def batch_size(self) -> int:
        return self.grasp_success.shape[0]

    @property
    def device(self) -> torch.device:
        return self.grasp_success.device
