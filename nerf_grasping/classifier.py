import pypose as pp
import torch
import torch.nn as nn
from nerf_grasping.grasp_utils import (
    NUM_PTS_X,
    NUM_PTS_Y,
    NUM_PTS_Z,
    NUM_FINGERS,
)
import pathlib
from nerf_grasping.models.dexgraspnet_models import CNN_3D_Classifier as CNN_3D_Model


class Classifier(nn.Module):
    def __init__(self, config: pathlib.Path):
        super().__init__()
        self.config = config

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

class CNN_3D_Classifier(Classifier):
    def __init__(self, config: pathlib.Path) -> None:
        super().__init__(config)
        self.model = self.load_model(config)

    def load_model(self, config: pathlib.Path) -> CNN_3D_Model:
        NUM_XYZ = 3
        model = CNN_3D_Model(
            input_shape=(NUM_XYZ + 1, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z),
            n_fingers=NUM_FINGERS,
        )
        model.load_state_dict(torch.load(config))
        return model

    def forward(
        self, nerf_densities: torch.Tensor, grasp_transforms: pp.LieTensor
    ) -> torch.Tensor:
        batch_size = nerf_densities.shape[0]