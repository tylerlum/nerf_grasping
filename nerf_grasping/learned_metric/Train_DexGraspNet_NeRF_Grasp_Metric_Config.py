from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Any, Dict, List, Optional, Tuple, Type
from nerf_grasping.classifier import Classifier, CNN_3D_XYZ_Classifier

from hydra.core.config_store import ConfigStore


@dataclass
class WandbConfig:
    entity: str = MISSING
    project: str = MISSING
    name: str = MISSING
    group: str = MISSING
    job_type: str = MISSING


@dataclass
class DataConfig:
    frac_val: float = MISSING
    frac_test: float = MISSING
    frac_train: float = MISSING

    input_dataset_root_dir: str = MISSING
    input_dataset_path: str = MISSING
    max_num_data_points: Optional[int] = MISSING

    use_random_rotations: bool = MISSING


@dataclass
class DataLoaderConfig:
    batch_size: int = MISSING
    num_workers: int = MISSING
    pin_memory: bool = MISSING

    load_nerf_grid_inputs_in_ram: bool = MISSING
    load_grasp_successes_in_ram: bool = MISSING
    load_grasp_transforms_in_ram: bool = MISSING
    load_nerf_configs_in_ram: bool = MISSING


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
    val_freq: int = MISSING
    val_on_epoch_0: bool = MISSING
    save_checkpoint_freq: int = MISSING
    save_checkpoint_on_epoch_0: bool = MISSING


@dataclass
class CheckpointWorkspaceConfig:
    root_dir: str = MISSING
    leaf_dir: str = MISSING
    force_no_resume: bool = MISSING


# @dataclass
# class ClassifierConfig:


# @dataclass
# class CNN_3D_XYZ_ClassifierConfig(ClassifierConfig):


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint_workspace: CheckpointWorkspaceConfig = field(
        default_factory=CheckpointWorkspaceConfig
    )
    classifier: Classifier = MISSING
    random_seed: int = MISSING
    dry_run: bool = MISSING


# Register ClassifierConfig subclasses.
cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="classifier", name="CNN_3D_XYZ_Classifier", node=CNN_3D_XYZ_Classifier)
