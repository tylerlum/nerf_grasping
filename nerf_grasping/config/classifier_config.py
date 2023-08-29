from dataclasses import dataclass, field
from typing import Optional, Literal, Tuple, List, Union
from nerf_grasping.config.fingertip_config import UnionFingertipConfig
from nerf_grasping.classifier import (
    CNN_3D_XYZ_Classifier,
    CNN_2D_1D_Classifier,
    Classifier,
)
from nerf_grasping.config.base import WandbConfig, CONFIG_DATETIME_STR
from nerf_grasping.config.nerfdata_config import NerfDataConfig
import tyro
import pathlib


@dataclass(frozen=True)
class ClassifierDataConfig:
    """Parameters for dataset loading."""

    frac_val: float = 0.1
    frac_test: float = 0.1
    frac_train: float = 1 - frac_val - frac_test

    max_num_data_points: Optional[int] = None
    """Maximum number of data points to use from the dataset. If None, use all."""

    use_random_rotations: bool = True
    """Flag to add random rotations to augment the dataset."""


@dataclass(frozen=True)
class ClassifierDataLoaderConfig:
    """Parameters for dataloader."""

    batch_size: int = 1

    num_workers: int = 8
    """Number of workers for the dataloader."""

    pin_memory: bool = True
    """Flag to pin memory for the dataloader."""

    load_nerf_grid_inputs_in_ram: bool = False
    """Flag to load the nerf grid inputs in RAM -- otherwise load on the fly."""

    load_grasp_successes_in_ram: bool = False
    """Flag to load the grasp successes in RAM -- otherwise load on the fly."""

    load_grasp_transforms_in_ram: bool = False
    """Flag to load the grasp transforms in RAM -- otherwise load on the fly."""

    load_nerf_configs_in_ram: bool = False
    """Flag to load the nerf configs in RAM -- otherwise load on the fly."""


@dataclass(frozen=True)
class ClassifierTrainingConfig:
    """Hyperparameters for training."""

    grad_clip_val: float = 1.0
    """Maximimum value of the gradient norm."""

    lr: float = 1e-4
    """Learning rate."""

    weight_decay: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    """Adam optimizer parameters."""

    label_smoothing: float = 0.0

    lr_scheduler_name: Literal[
        "constant",
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant_with_warmup",
        "piecewise_constant",
    ] = "constant"
    """Strategy for learning rate scheduling."""

    lr_scheduler_num_warmup_steps: int = 0
    """(if applicable) number of warmup steps for learning rate scheduling."""

    n_epochs: int = 1000
    """Number of epochs to train for."""

    val_freq: int = 5
    """Number of iterations between validation steps."""

    val_on_epoch_0: bool = False
    """Flag to run validation on epoch 0."""

    save_checkpoint_freq: int = 5
    """Number of iterations between saving checkpoints."""

    save_checkpoint_on_epoch_0: bool = False
    """Flag to save checkpoint on epoch 0."""


@dataclass(frozen=True)
class CheckpointWorkspaceConfig:
    """Parameters for paths to checkpoints."""

    root_dir: str = "Train_DexGraspNet_NeRF_Grasp_Metric_workspaces"
    """Root directory for checkpoints."""

    leaf_dir: str = field(default_factory=lambda: CONFIG_DATETIME_STR)
    """Leaf directory for checkpoints."""

    force_no_resume: bool = True
    """Flag to force no resume, even if a checkpoint exists."""


@dataclass(frozen=True)
class ClassifierModelConfig:
    """Default (abstract) parameters for the classifier."""

    def get_classifier_from_fingertip_config(
        self, fingertip_config: UnionFingertipConfig
    ) -> Classifier:
        """Helper method to return the correct classifier from config."""
        raise NotImplementedError("Implement in subclass.")


@dataclass(frozen=True)
class CNN_3D_XYZ_ModelConfig(ClassifierModelConfig):
    """Parameters for the CNN_3D_XYZ_Classifier."""

    conv_channels: List[int]
    """List of channels for each convolutional layer. Length specifies number of layers."""

    mlp_hidden_layers: List[int]
    """List of hidden layer sizes for the MLP. Length specifies number of layers."""

    n_fingers: int = 4
    """Number of fingers."""

    def input_shape_from_fingertip_config(self, fingertip_config: UnionFingertipConfig):
        return [
            4,
            fingertip_config.num_pts_x,
            fingertip_config.num_pts_y,
            fingertip_config.num_pts_z,
        ]

    def get_classifier_from_fingertip_config(
        self, fingertip_config: UnionFingertipConfig
    ) -> Classifier:
        """Helper method to return the correct classifier from config."""

        input_shape = self.input_shape_from_fingertip_config(fingertip_config)
        return CNN_3D_XYZ_Classifier(
            input_shape=input_shape,
            n_fingers=fingertip_config.n_fingers,
            conv_channels=self.conv_channels,
            mlp_hidden_layers=self.mlp_hidden_layers,
        )


# TODO(pculbert): fix.
# UnionClassifierModelConfig = Union[
#     CNN_3D_XYZ_ModelConfig, ClassifierModelConfig
# ]  # Passing none here so union is valid.


@dataclass(frozen=True)
class CNN_2D_1D_ModelConfig(ClassifierModelConfig):
    """Parameters for the CNN_2D_1D_Classifier."""

    conv_2d_film_hidden_layers: List[int]
    mlp_hidden_layers: List[int]
    conditioning_dim: int = 7
    n_fingers: int = 4

    @classmethod
    def grid_shape_from_fingertip_config(
        self, fingertip_config: UnionFingertipConfig
    ) -> List[int]:
        return [
            fingertip_config.num_pts_x,
            fingertip_config.num_pts_y,
            fingertip_config.num_pts_z,
        ]

    def get_classifier_from_fingertip_config(
        self, fingertip_config: UnionFingertipConfig
    ) -> Classifier:
        """Helper method to return the correct classifier from config."""

        return CNN_2D_1D_Classifier(
            grid_shape=self.grid_shape_from_fingertip_config(fingertip_config),
            n_fingers=self.n_fingers,
            conditioning_dim=self.conditioning_dim,
            conv_2d_film_hidden_layers=self.conv_2d_film_hidden_layers,
            mlp_hidden_layers=self.mlp_hidden_layers,
        )


UnionClassifierModelConfig = tyro.extras.subcommand_type_from_defaults(
    {
        "cnn_3d_xyz": CNN_3D_XYZ_ModelConfig(
            conv_channels=[32, 64, 128], mlp_hidden_layers=[256, 256]
        ),
        "cnn_2d_1d": CNN_2D_1D_ModelConfig(
            conv_2d_film_hidden_layers=[256, 256], mlp_hidden_layers=[256, 256]
        ),
    }
)


@dataclass
class ClassifierConfig:
    model_config: UnionClassifierModelConfig
    data: ClassifierDataConfig
    nerfdata_config: NerfDataConfig
    dataloader: ClassifierDataLoaderConfig
    training: ClassifierTrainingConfig
    checkpoint_workspace: CheckpointWorkspaceConfig
    nerfdata_cfg_path: Optional[pathlib.Path]

    wandb: WandbConfig = field(
        default_factory=lambda: WandbConfig(
            project="learned_metric", name=CONFIG_DATETIME_STR
        )
    )

    dry_run: bool = False
    random_seed: int = 42

    def __post_init__(self):
        """
        If a nerfdata config path was passed, load that config object.
        Otherwise use defaults.

        Then load the correct model config based on the nerfdata config.
        """
        if self.nerfdata_cfg_path is not None:
            print(f"Loading nerfdata config from {self.nerfdata_cfg_path}")
            self.nerfdata_config = tyro.extras.from_yaml(
                NerfDataConfig, self.nerfdata_cfg_path.open()
            )
        elif self.nerfdata_config.config_filepath.exists():
            print(
                f"Loading nerfdata config from {self.nerfdata_config.config_filepath}"
            )
            self.nerfdata_config = tyro.extras.from_yaml(
                NerfDataConfig, self.nerfdata_config.config_filepath.open()
            )
        else:
            print("Loading default nerfdata config")


if __name__ == "__main__":
    cfg = tyro.cli(ClassifierConfig)
    print(cfg)
