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
from nerf_grasping.models.dexgraspnet_models import (
    CNN_3D_Model,
    CNN_2D_1D_Model,
)
from nerf_grasping.learned_metric.DexGraspNet_batch_data import BatchDataInput
from typing import Iterable, Tuple


class Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch_data_input: BatchDataInput) -> torch.Tensor:
        nerf_densities = batch_data_input.nerf_densities
        batch_size = nerf_densities.shape[0]
        assert nerf_densities.shape == (
            batch_size,
            NUM_FINGERS,
            NUM_PTS_X,
            NUM_PTS_Y,
            NUM_PTS_Z,
        )

        grasp_transforms = batch_data_input.grasp_transforms
        assert grasp_transforms.ltype == pp.SE3_type
        assert grasp_transforms.lshape == (batch_size, NUM_FINGERS)

        hardcoded_output = torch.zeros(
            batch_size, dtype=nerf_densities.dtype, device=nerf_densities.device
        )
        assert hardcoded_output.shape == (batch_size,)
        raise NotImplementedError
        return hardcoded_output

    def get_success_logits(self, batch_data_input: BatchDataInput) -> torch.Tensor:
        return self.forward(batch_data_input)

    def get_failure_probability(self, batch_data_input: BatchDataInput) -> torch.Tensor:
        logits = self.get_success_logits(batch_data_input)
        N_CLASSES = 2
        assert logits.shape == (batch_data_input.batch_size, N_CLASSES)

        # REMOVE, using to ensure gradients are non-zero
        # for overfit classifier.
        PROB_SCALING = 1e0

        # Return failure probabilities (as loss).
        return nn.functional.softmax(PROB_SCALING * logits, dim=-1)[:, 0]


class CNN_3D_XYZ_Classifier(Classifier):
    def __init__(
        self,
        input_shape: Iterable[int],
        conv_channels: Iterable[int],
        mlp_hidden_layers: Iterable[int],
        n_fingers: int,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.conv_channels = conv_channels
        self.mlp_hidden_layers = mlp_hidden_layers
        self.n_fingers = n_fingers

        self.model = CNN_3D_Model(
            input_shape=input_shape,
            conv_channels=conv_channels,
            mlp_hidden_layers=mlp_hidden_layers,
            n_fingers=n_fingers,
        )

    def forward(self, batch_data_input: BatchDataInput) -> torch.Tensor:
        # Run model
        logits = self.model.get_success_logits(batch_data_input.nerf_alphas_with_augmented_coords)
        return logits


class CNN_2D_1D_Classifier(Classifier):
    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        n_fingers: int,
        conditioning_dim: int,
        conv_2d_film_hidden_layers: Tuple[int, ...],
        mlp_hidden_layers: Tuple[int, ...],
    ) -> None:
        super().__init__()
        self.model = CNN_2D_1D_Model(
            grid_shape=grid_shape,
            n_fingers=n_fingers,
            conditioning_dim=conditioning_dim,
            conv_2d_film_hidden_layers=conv_2d_film_hidden_layers,
            mlp_hidden_layers=mlp_hidden_layers,
        )

    def forward(self, batch_data_input: BatchDataInput) -> torch.Tensor:
        # Run model
        logits = self.model.get_success_logits(batch_data_input.nerf_alphas, batch_data_input.grasp_transforms)
        return logits


def main() -> None:
    # Prepare inputs
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example input
    batch_size = 2
    nerf_densities = torch.rand(
        batch_size, NUM_FINGERS, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z
    ).to(DEVICE)
    grasp_transforms = pp.identity_SE3(batch_size, NUM_FINGERS).to(DEVICE)

    batch_data_input = BatchDataInput(
        nerf_densities=nerf_densities,
        grasp_transforms=grasp_transforms,
    ).to(DEVICE)

    # Create model
    cnn_3d_classifier = CNN_3D_XYZ_Classifier(
        input_shape=batch_data_input.nerf_alphas_with_augmented_coords.shape[-4:],
        conv_channels=[32, 64, 128],
        mlp_hidden_layers=[256, 256],
        n_fingers=NUM_FINGERS,
    ).to(DEVICE)
    cnn_2d_1d_classifier = CNN_2D_1D_Classifier(
        grid_shape=(NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z),
        n_fingers=NUM_FINGERS,
        conditioning_dim=7,
        conv_2d_film_hidden_layers=(256, 256),
        mlp_hidden_layers=(256, 256),
    ).to(DEVICE)

    # Run model
    cnn_3d_scores = cnn_3d_classifier.get_failure_probability(batch_data_input)
    print(f"cnn_3d_scores: {cnn_3d_scores}")
    print(f"cnn_3d_scores.shape: {cnn_3d_scores.shape}")
    cnn_2d_1d_scores = cnn_2d_1d_classifier.get_failure_probability(batch_data_input)
    print(f"cnn_2d_1d_scores: {cnn_2d_1d_scores}")
    print(f"cnn_2d_1d_scores.shape: {cnn_2d_1d_scores.shape}")


if __name__ == "__main__":
    main()
