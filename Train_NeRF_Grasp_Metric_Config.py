from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Any, Dict, List, Optional, Tuple, Type
from enum import Enum, auto
from tyler_new_models import ClassifierConfig


@dataclass
class WandbConfig:
    entity: str = MISSING
    project: str = MISSING
    name: str = MISSING
    group: str = MISSING
    job_type: str = MISSING


class PreprocessDensityType(Enum):
    DENSITY = auto()
    ALPHA = auto()
    WEIGHT = auto()


@dataclass
class DataConfig:
    frac_val: float = MISSING
    frac_test: float = MISSING
    frac_train: float = MISSING

    input_dataset_root_dir: str = MISSING
    input_dataset_path: str = MISSING
    max_num_data_points: Optional[int] = MISSING


@dataclass
class DataLoaderConfig:
    batch_size: int = MISSING
    num_workers: int = MISSING
    pin_memory: bool = MISSING

    load_nerf_grid_inputs_in_ram: bool = MISSING
    load_grasp_successes_in_ram: bool = MISSING
    downsample_factor_x: int = MISSING
    downsample_factor_y: int = MISSING
    downsample_factor_z: int = MISSING


@dataclass
class PreprocessConfig:
    flip_left_right_randomly: bool = MISSING
    density_type: PreprocessDensityType = MISSING
    add_invariance_transformations: bool = MISSING
    rotate_polar_angle: bool = MISSING
    reflect_around_xz_plane_randomly: bool = MISSING
    reflect_around_xy_plane_randomly: bool = MISSING
    remove_y_axis: bool = MISSING


@dataclass
class TrainingConfig:
    grad_clip_val: float = MISSING
    lr: float = MISSING
    weight_decay: float = MISSING
    betas: Tuple[float, float] = MISSING
    label_smoothing: float = MISSING
    lr_scheduler_name: str = MISSING
    lr_scheduler_num_warmup_steps: int = MISSING
    n_epochs: int = MISSING
    log_grad_freq: int = MISSING
    log_grad_on_epoch_0: bool = MISSING
    val_freq: int = MISSING
    val_on_epoch_0: bool = MISSING
    save_checkpoint_freq: int = MISSING
    save_checkpoint_on_epoch_0: bool = MISSING
    confusion_matrix_freq: int = MISSING
    save_confusion_matrix_on_epoch_0: bool = MISSING
    use_dataloader_subset: bool = MISSING


@dataclass
class CheckpointWorkspaceConfig:
    root_dir: str = MISSING
    leaf_dir: str = MISSING
    force_no_resume: bool = MISSING


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    checkpoint_workspace: CheckpointWorkspaceConfig = field(default_factory=CheckpointWorkspaceConfig)
    random_seed: int = MISSING
    visualize_data: bool = MISSING
    dry_run: bool = MISSING