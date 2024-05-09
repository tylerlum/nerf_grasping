from dataclasses import dataclass
import pathlib
import time

from nerfstudio.scripts.train import _set_random_seed
from nerfstudio.engine.trainer import TrainerConfig, Trainer
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from typing import Optional, Literal, Tuple
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.models.base_model import Model
import typing
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


@dataclass
class Args:
    """The goal of this is to start creating nerfstudio things before data is ready to amortize setup time"""

    nerfdata_folder: pathlib.Path
    nerfcheckpoints_folder: pathlib.Path
    num_images: int
    max_num_iterations: int = 200


# True to enable amortizing nerfstudio start time, set to False to use normal behavior without any modification
USE_CUSTOM_APPROACH = True
LOCAL_RANK = 0
WORLD_SIZE = 1
GLOBAL_RANK = 0


class CustomPipeline(VanillaPipeline):
    # HACK: num_images is a class variable that must be defined before the model is created
    # Very important that this is set before the model is created, as it needs to know
    # Will give big confusing errors if wrong
    num_images = None

    # HACK: trainer will look for pipeline.datamanager for .get_param_groups() and .get_training_callbacks()
    # Those are empty, so we can just populate those with dummy values
    class DummyDataManager:
        def get_param_groups(self):
            return {}

        def get_training_callbacks(self, training_callback_attributes):
            return []

    def __init__(
        self,
        config: VanillaPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        """THIS IS A SMALL MODIFICATION OF __init__ in VanillaPipeline https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/pipelines/base_pipeline.py"""
        # HACK: Don't call VanillaPipeline.__init__, but its super
        super(VanillaPipeline, self).__init__()

        self.config = config
        self.test_mode = test_mode

        if USE_CUSTOM_APPROACH:
            # HACK: Don't load datamanager yet
            self.datamanager = self.DummyDataManager()

            # Manually populate this
            # ASSUME seed_pts is None
            seed_pts = None
            aabb_scale = self.config.datamanager.dataparser.scene_scale
            scene_box = SceneBox(
                aabb=torch.tensor(
                    [
                        [-aabb_scale, -aabb_scale, -aabb_scale],
                        [aabb_scale, aabb_scale, aabb_scale],
                    ],
                    dtype=torch.float32,
                )
            )
            assert hasattr(
                CustomPipeline, "num_images"
            ), "HACK: Must define CustomPipeline.num_images or model cannot be created"
            num_train_data = CustomPipeline.num_images
            metadata = (
                {
                    "depth_filenames": None,
                    "depth_unit_scale_factor": self.config.datamanager.dataparser.depth_unit_scale_factor,
                    "mask_color": self.config.datamanager.dataparser.mask_color,
                },
            )

        else:
            self.datamanager: DataManager = config.datamanager.setup(
                device=device,
                test_mode=test_mode,
                world_size=world_size,
                local_rank=local_rank,
            )
            # TODO make cleaner
            seed_pts = None
            if (
                hasattr(self.datamanager, "train_dataparser_outputs")
                and "points3D_xyz" in self.datamanager.train_dataparser_outputs.metadata
            ):
                pts = self.datamanager.train_dataparser_outputs.metadata["points3D_xyz"]
                pts_rgb = self.datamanager.train_dataparser_outputs.metadata[
                    "points3D_rgb"
                ]
                seed_pts = (pts, pts_rgb)
            self.datamanager.to(device)
            # TODO(ethan): get rid of scene_bounds from the model
            assert self.datamanager.train_dataset is not None, "Missing input dataset"

            scene_box = self.datamanager.train_dataset.scene_box
            num_train_data = len(self.datamanager.train_dataset)
            metadata = self.datamanager.train_dataset.metadata

        self._model = config.model.setup(
            scene_box=scene_box,
            num_train_data=num_train_data,
            metadata=metadata,
            device=device,
            grad_scaler=grad_scaler,
            seed_points=seed_pts,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                Model,
                DDP(self._model, device_ids=[local_rank], find_unused_parameters=True),
            )
            dist.barrier(device_ids=[local_rank])


def setup_train_loop_without_data(config: TrainerConfig) -> Trainer:
    """Main training function that sets up and runs the trainer per process
    THIS IS A MODIFICATION OF train_loop in https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/scripts/train.py
    BUT RETURNS THE TRAINER

    Args:
        config: config file specifying training regimen
    """
    _set_random_seed(config.machine.seed + GLOBAL_RANK)
    trainer = config.setup(local_rank=LOCAL_RANK, world_size=WORLD_SIZE)
    trainer.setup()
    return trainer


def finish_train_nerf(
    trainer: Trainer,
    config: TrainerConfig,
) -> Trainer:
    if USE_CUSTOM_APPROACH:
        # Setup datamanager now that data has arrived (same as start as original VanillaPipeline)
        trainer.pipeline.datamanager = trainer.pipeline.config.datamanager.setup(
            device=trainer.pipeline.device,
            test_mode=trainer.pipeline.test_mode,
            world_size=WORLD_SIZE,
            local_rank=LOCAL_RANK,
        )

        trainer.pipeline.datamanager.to(trainer.pipeline.device)
        assert (
            trainer.pipeline.datamanager.train_dataset is not None
        ), "Missing input dataset"

    trainer.train()


def get_nerfacto_default_config():
    # From nerfstudio/configs/method_configs.py nerfacto
    from nerfstudio.configs.method_configs import all_methods

    return all_methods["nerfacto"]


def get_nerfacto_custom_config(num_images: int):
    config = get_nerfacto_default_config()
    assert config.pipeline._target == VanillaPipeline  # Before it was VanillaPipeline

    CustomPipeline.num_images = num_images  # Must set before model is created
    config.pipeline._target = CustomPipeline
    return config


def setup_train_nerf_without_data(
    args: Args,
) -> Tuple[Trainer, TrainerConfig]:
    config = get_nerfacto_custom_config(num_images=args.num_images)

    # Modifications
    config.data = args.nerfdata_folder
    config.pipeline.datamanager.data = args.nerfdata_folder
    config.max_num_iterations = args.max_num_iterations
    config.output_dir = args.nerfcheckpoints_folder
    config.vis = "none"  # "wandb"

    config.pipeline.model.disable_scene_contraction = True
    config.pipeline.datamanager.dataparser.auto_scale_poses = False
    config.pipeline.datamanager.dataparser.scale_factor = 1.0
    config.pipeline.datamanager.dataparser.center_method = "none"
    config.pipeline.datamanager.dataparser.orientation_method = "none"

    # Need to set timestamp
    config.set_timestamp()

    # print and save config
    config.print_to_terminal()
    config.save_config()

    trainer = setup_train_loop_without_data(config=config)
    return trainer, config


def main() -> None:
    args = Args(
        nerfdata_folder=pathlib.Path(
            "experiments/2024-04-15_DEBUG/nerfdata/sem-Vase-3a275e00d69810c62600e861c93ad9cc_0_0846"
        ),
        nerfcheckpoints_folder=pathlib.Path(
            "experiments/2024-04-15_DEBUG/nerfcheckpoints/"
        ),
        num_images=250,
    )
    trainer, train_config = setup_train_nerf_without_data(args)

    first = True
    while (
        not (args.nerfdata_folder / "transforms.json").exists()
        or not (args.nerfdata_folder / "images").exists()
    ):
        if first:
            print(f"Waiting for {args.nerfdata_folder / 'transforms.json'} to exist")
            first = False
        time.sleep(0.01)

    num_files = len(list((args.nerfdata_folder / "images").iterdir()))
    assert (
        num_files == args.num_images
    ), f"Expected {args.num_images} files, got {num_files} in {args.nerfdata_folder / 'images'}"

    trainer = finish_train_nerf(
        trainer=trainer,
        config=train_config,
    )


if __name__ == "__main__":
    main()
