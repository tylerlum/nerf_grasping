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
from nerf_grasping.learned_metric.DexGraspNet_batch_data import BatchData


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
        # TODO: Later will need to find another way of using config to get model architecture
        # Currently has many hardcoded/default values
        NUM_XYZ = 3
        model = CNN_3D_Model(
            input_shape=(NUM_XYZ + 1, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z),
            n_fingers=NUM_FINGERS,
        )
        state_dict = torch.load(config)["nerf_to_grasp_success_model"]
        model.load_state_dict(state_dict)
        return model

    def forward(
        self, nerf_densities: torch.Tensor, grasp_transforms: pp.LieTensor
    ) -> torch.Tensor:
        # Prepare data
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

        # Run model
        scores = self.model(batch_data.nerf_alphas_with_augmented_coords)

        N_CLASSES = 2
        assert scores.shape == (batch_size, N_CLASSES)

        scores = scores[:, 1]  # take the "grasp success" score

        return scores


def main() -> None:
    # Prepare inputs
    CONFIG = (
        pathlib.Path(nerf_grasping.get_package_root())
        / "learned_metric"
        / "Train_DexGraspNet_NeRF_Grasp_Metric_workspaces"
        / "2023-08-19_16-26-29"
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
