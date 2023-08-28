from dataclasses import dataclass, field
from typing import Optional, Tuple
from nerf_grasping.classifier import Classifier, CNN_3D_XYZ_Classifier
from datetime import datetime

DATETIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


@dataclass
class WandbConfig:
    entity: str = "tylerlum"
    project: str = "NeRF_Grasp_Metric_V2"
    name: str = DATETIME_STR
    group: str = ""
    job_type: str = ""


@dataclass
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


@dataclass
class DataLoaderConfig:
    batch_size: int = 64
    num_workers: int = 8
    pin_memory: bool = True

    load_nerf_grid_inputs_in_ram: bool = False
    load_grasp_successes_in_ram: bool = False
    load_grasp_transforms_in_ram: bool = False
    load_nerf_configs_in_ram: bool = False


@dataclass
class TrainingConfig:
    grad_clip_val: float = 1.0
    lr: float = 1e-4
    weight_decay: float = 1e-3
    betas: Tuple[float, float] = [0.9, 0.999]
    label_smoothing: float = 0.0
    lr_scheduler_name: str = "constant"
    lr_scheduler_num_warmup_steps: int = 0
    n_epochs: int = 1000
    val_freq: int = 5
    val_on_epoch_0: bool = False
    save_checkpoint_freq: int = 5
    save_checkpoint_on_epoch_0: bool = False


@dataclass
class CheckpointWorkspaceConfig:
    root_dir: str = "Train_DexGraspNet_NeRF_Grasp_Metric_workspaces"
    leaf_dir: str = DATETIME_STR
    force_no_resume: bool = True


@dataclass
class Config:
    data: DataConfig
    dataloader: DataLoaderConfig
    wandb: WandbConfig
    training: TrainingConfig
    checkpoint_workspace: CheckpointWorkspaceConfig
    classifier: Classifier = CNN_3D_XYZ_Classifier()
    random_seed: int = 42
    dry_run: bool = False
