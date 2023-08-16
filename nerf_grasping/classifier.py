import pypose as pp
import torch
import torch.nn as nn
from nerf_grasping.grasp_utils import (
    NUM_PTS_X,
    NUM_PTS_Y,
    NUM_PTS_Z,
    NUM_FINGERS,
)


class Classifier(nn.Module):
    def __init__(self):
        pass

    def forward(
        self, nerf_densities: torch.Tensor, grasp_transforms: pp.LieTensor
    ) -> torch.Tensor:
        batch_size = nerf_densities.shape[0]
        assert nerf_densities.shape == (
            batch_size,
            NUM_FINGERS,
            NUM_PTS_X,
            NUM_PTS_Y,
            NUM_PTS_Z,
        )
        assert grasp_transforms.ltype == pp.SE3_type
        assert grasp_transforms.lshape == (batch_size, NUM_FINGERS)

        hardcoded_output = torch.zeros(
            batch_size, dtype=nerf_densities.dtype, device=nerf_densities.device
        )
        assert hardcoded_output.shape == (batch_size,)
        return hardcoded_output
