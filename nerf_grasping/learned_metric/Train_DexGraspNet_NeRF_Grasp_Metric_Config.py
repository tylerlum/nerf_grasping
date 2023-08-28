from dataclasses import dataclass, field
from typing import Optional, Tuple, Iterable, Union
from nerf_grasping.classifier import Classifier, CNN_3D_XYZ_Classifier
from datetime import datetime

# TODO(pculbert): refactor grasp_utils to make these configurable.
from nerf_grasping.grasp_utils import (
    NUM_PTS_X,
    NUM_PTS_Y,
    NUM_PTS_Z,
    NUM_FINGERS,
)

DATETIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
NUM_XYZ = 3
DEFAULT_INPUT_SHAPE = [NUM_XYZ + 1, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z]


@dataclass(frozen=True)
class WandbConfig:
    entity: str = "tylerlum"
    project: str = "NeRF_Grasp_Metric_V2"
    name: str = field(default_factory=lambda: DATETIME_STR)
    group: str = ""
    job_type: str = ""


@dataclass(frozen=True)
class DataConfig:
    frac_val: float = 0.1
    frac_test: float = 0.1
    frac_train: float = 1 - frac_val - frac_test

    input_dataset_root_dir: str = (
        "data/2023-08-26_evaled_overfit_grasp_config_dicts_learned_metric_dataset"
    )
    input_dataset_path: str = "2023-08-26_11-24-44_learned_metric_dataset.h5"
    max_num_data_points: Optional[int] = None

    use_random_rotations: bool = True


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int = 64
    num_workers: int = 8
    pin_memory: bool = True

    load_nerf_grid_inputs_in_ram: bool = False
    load_grasp_successes_in_ram: bool = False
    load_grasp_transforms_in_ram: bool = False
    load_nerf_configs_in_ram: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    grad_clip_val: float = 1.0
    lr: float = 1e-4
    weight_decay: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    label_smoothing: float = 0.0
    lr_scheduler_name: str = "constant"
    lr_scheduler_num_warmup_steps: int = 0
    n_epochs: int = 1000
    val_freq: int = 5
    val_on_epoch_0: bool = False
    save_checkpoint_freq: int = 5
    save_checkpoint_on_epoch_0: bool = False


@dataclass(frozen=True)
class CheckpointWorkspaceConfig:
    root_dir: str = "Train_DexGraspNet_NeRF_Grasp_Metric_workspaces"
    leaf_dir: str = field(default_factory=lambda: DATETIME_STR)
    force_no_resume: bool = True


@dataclass(frozen=True)
class ClassifierConfig:
    input_shape: Iterable[int] = field(default_factory=lambda: DEFAULT_INPUT_SHAPE)


@dataclass(frozen=True)
class CNN_3D_XYZ_ClassifierConfig(ClassifierConfig):
    n_fingers: int = 4
    conv_channels: Iterable[int] = (32, 64, 128)
    mlp_hidden_layers: Iterable[int] = (256, 256)

    def get_classifier(self):
        return CNN_3D_XYZ_Classifier(
            input_shape=self.input_shape,
            n_fingers=self.n_fingers,
            conv_channels=self.conv_channels,
            mlp_hidden_layers=self.mlp_hidden_layers,
        )


@dataclass
class Config:
    data: DataConfig
    dataloader: DataLoaderConfig
    wandb: WandbConfig
    training: TrainingConfig
    checkpoint_workspace: CheckpointWorkspaceConfig
    classifier: ClassifierConfig = CNN_3D_XYZ_ClassifierConfig()
    random_seed: int = 42
    dry_run: bool = False
