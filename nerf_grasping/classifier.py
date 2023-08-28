import pypose as pp
import torch
import torch.nn as nn
import nerf_grasping
from nerf_grasping.grasp_utils import (
    NUM_PTS_X,
    NUM_PTS_Y,
    NUM_PTS_Z,
    NUM_FINGERS,
)
import pathlib
from nerf_grasping.models.dexgraspnet_models import CNN_3D_Classifier as CNN_3D_Model
from nerf_grasping.learned_metric.DexGraspNet_batch_data import BatchDataInput
from typing import List
from dataclasses import dataclass


@dataclass
class Classifier(nn.Module):
    input_shape: List[int]

    def __init__(self, input_shape: List[int]) -> None:
        super().__init__()
        self.input_shape = input_shape

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


@dataclass
class CNN_3D_XYZ_Classifier(Classifier):
    conv_channels: List[int]
    mlp_hidden_layers: List[int]
    n_fingers: int = 4

    def __init__(
        self,
        input_shape: List[int],
        conv_channels: List[int],
        mlp_hidden_layers: List[int],
        n_fingers,
    ) -> CNN_3D_Model:
        # TODO: Later will need to find another way of using config to get model architecture
        # Currently has many hardcoded/default values
        super().__init__(input_shape=input_shape)
        self.conv_channels = conv_channels
        self.mlp_hidden_layers = mlp_hidden_layers
        self.n_fingers = n_fingers

        self.model = CNN_3D_Model(
            input_shape=input_shape,
            conv_channels=conv_channels,
            mlp_hidden_layers=mlp_hidden_layers,
            n_fingers=n_fingers,
        )

    def forward(
        self, nerf_densities: torch.Tensor, grasp_transforms: pp.LieTensor
    ) -> torch.Tensor:
        batch_size = nerf_densities.shape[0]

        # Prepare data
        batch_data_input = BatchDataInput(
            nerf_densities=nerf_densities,
            grasp_transforms=grasp_transforms,
        ).to(nerf_densities.device)

        # Run model
        logits = self.model(batch_data_input.nerf_alphas_with_augmented_coords)

        N_CLASSES = 2
        assert logits.shape == (batch_size, N_CLASSES)

        # REMOVE, using to ensure gradients are non-zero
        # for overfit classifier.
        PROB_SCALING = 1e0

        # Return failure probabilities (as loss).
        return nn.functional.softmax(PROB_SCALING * logits, dim=-1)[:, 0]


def main() -> None:
    # Prepare inputs
    CONFIG = (
        pathlib.Path(nerf_grasping.get_package_root())
        / "learned_metric"
        / "Train_DexGraspNet_NeRF_Grasp_Metric_workspaces"
        / "2023-08-20_17-18-07"
        / "checkpoint_1000.pt"
    )
    assert CONFIG.exists(), f"CONFIG: {CONFIG} does not exist"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    cnn_3d_classifier = CNN_3D_Classifier(config=CONFIG).to(DEVICE)

    # Example input
    batch_size = 2
    nerf_densities = torch.rand(
        batch_size, NUM_FINGERS, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z
    ).to(DEVICE)
    grasp_transforms = pp.identity_SE3(batch_size, NUM_FINGERS).to(DEVICE)

    # Run model
    scores = cnn_3d_classifier(nerf_densities, grasp_transforms)
    print(f"scores: {scores}")
    print(f"scores.shape: {scores.shape}")


if __name__ == "__main__":
    main()
