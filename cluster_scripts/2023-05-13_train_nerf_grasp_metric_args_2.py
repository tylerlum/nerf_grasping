import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, List, Optional
from omegaconf import MISSING
from Train_NeRF_Grasp_Metric_Config import Config
from tyler_new_models import Encoder1DType
from enum import Enum

base_command = "python Train_NeRF_Grasp_Metric.py"

cfg = Config()

builtins = (int, float, str, bool, list, tuple, dict, Enum)


def get_non_missing_keys(
    cfg: Any, path: str = "", non_missing_keys: Optional[List[str]] = None
) -> List[str]:
    if non_missing_keys is None:
        non_missing_keys = []

    if not hasattr(cfg, "__dict__"):
        return non_missing_keys

    for field_name, field_value in cfg.__dict__.items():
        field_path = f"{path}.{field_name}" if path else field_name

        if field_value != MISSING and isinstance(field_value, builtins):
            non_missing_keys.append(field_path)
        elif field_value == MISSING and isinstance(field_value, builtins):
            continue
        else:
            get_non_missing_keys(
                field_value, path=field_path, non_missing_keys=non_missing_keys
            )

    return non_missing_keys


cfg.wandb.name = "cnn2d_cnn1d_66categories_lr1e-4_invariance_fromscratch"
cfg.data.input_dataset_path = "nerf_acronym_grasp_success_dataset_66_categories_v2.h5"
cfg.training.use_dataloader_subset = False
cfg.training.lr = 1e-4
cfg.dataloader.batch_size = 8
cfg.classifier.encoder_1d_type = Encoder1DType.CONV
cfg.preprocess.flip_left_right_randomly = True
cfg.preprocess.add_invariance_transformations = True
cfg.preprocess.rotate_polar_angle = True
cfg.preprocess.reflect_around_xz_plane_randomly = True
cfg.preprocess.reflect_around_xy_plane_randomly = True
cfg.preprocess.remove_y_axis = True
cfg.classifier.conv_encoder_2d_config.use_pretrained = False

get_non_missing_keys(cfg)
command = base_command

non_missing_keys = sorted(get_non_missing_keys(cfg))
for non_missing_key in non_missing_keys:
    value = eval(f"cfg.{non_missing_key}")
    command += f" {non_missing_key}={value}"

print(command)
