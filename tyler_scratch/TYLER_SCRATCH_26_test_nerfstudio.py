# %%
from nerfstudio.scripts.train import _set_random_seed
from nerfstudio.engine.trainer import TrainerConfig
from dataclasses import dataclass
import pathlib


from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig, Trainer
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig


# %%
def train_loop(
    local_rank: int, world_size: int, config: TrainerConfig, global_rank: int = 0
) -> Trainer:
    """Main training function that sets up and runs the trainer per process

    Args:
        local_rank: current rank of process
        world_size: total number of gpus available
        config: config file specifying training regimen
    """
    _set_random_seed(config.machine.seed + global_rank)
    trainer = config.setup(local_rank=local_rank, world_size=world_size)
    trainer.setup()
    trainer.train()
    return trainer


# %%
@dataclass
class Args:
    nerfdata_folder: pathlib.Path
    nerfcheckpoints_folder: pathlib.Path
    max_num_iterations: int = 200
    is_real_world: bool = False


# %%
args = Args(
    nerfdata_folder=pathlib.Path(
        "experiments/2024-04-15_DEBUG/nerfdata/sem-Vase-3a275e00d69810c62600e861c93ad9cc_0_0846"
    ),
    nerfcheckpoints_folder=pathlib.Path(
        "experiments/2024-04-15_DEBUG/nerfcheckpoints/"
    ),
)

# %%
# From nerfstudio/configs/method_configs.py
config = TrainerConfig(
    method_name="nerfacto",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="SO3xR3",
                optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                scheduler=ExponentialDecaySchedulerConfig(
                    lr_final=6e-6, max_steps=200000
                ),
            ),
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=0.0001, max_steps=200000
            ),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=0.0001, max_steps=200000
            ),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

# %%
# Modifications
config.data = args.nerfdata_folder
config.pipeline.datamanager.data = args.nerfdata_folder
config.max_num_iterations = args.max_num_iterations
config.output_dir = args.nerfcheckpoints_folder
config.vis = "wandb"
config.pipeline.model.disable_scene_contraction = True
config.pipeline.model.background_color = (
    "black" if not args.is_real_world else "last_sample"
)
config.pipeline.datamanager.dataparser.auto_scale_poses = False
config.pipeline.datamanager.dataparser.scale_factor = 1.0
config.pipeline.datamanager.dataparser.scene_scale = (
    0.2 if not args.is_real_world else 1.0
)
config.pipeline.datamanager.dataparser.center_method = "none"
config.pipeline.datamanager.dataparser.orientation_method = "none"

# %%
trainer = train_loop(0, 1, config)
pipeline = trainer.pipeline

# %%
print(pipeline.model.field)

# %%
from nerfstudio.configs.method_configs import all_methods
# %%
all_methods["nerfacto"]

# %%
