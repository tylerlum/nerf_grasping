from dataclasses import dataclass
import pathlib
import tyro
from typing import Optional
import os


@dataclass
class NerfTrainingConfig:
    """Configuration for training a NeRF model."""

    nerfdata_path: pathlib.Path = pathlib.Path("data/nerfdata")
    nerfcheckpoints_path: Optional[pathlib.Path] = None
    model: str = "nerfacto"
    num_iters: int = 200
    scene_scale: float = 0.2


def main(cfg: NerfTrainingConfig):
    print(cfg)
    object_code_and_scale_strs = [x.stem for x in cfg.nerfdata_path.iterdir()]
    if cfg.nerfcheckpoints_path is None:
        nerfcheckpoints_path = cfg.nerfdata_path.parent / "nerfcheckpoints"

    print(f"Found {len(object_code_and_scale_strs)} objects.")
    print(f"Saving checkpoints to {nerfcheckpoints_path}")

    for object_code_and_scale_str in object_code_and_scale_strs:
        nerf_training_command = (
            "ns-train nerfacto"
            + f" --data {cfg.nerfdata_path / object_code_and_scale_str}"
            + f" --max-num-iterations {cfg.num_iters}"
            + f" --output-dir {nerfcheckpoints_path}"
            + f" --vis wandb"
            + f" --pipeline.model.disable-scene-contraction True"
            + f" --pipeline.model.background-color black"
            + f" nerfstudio-data"
            + f" --auto-scale-poses False"
            + f" --scale-factor 1."
            + f" --scene-scale {cfg.scene_scale}"
            + f" --center-method none"
            + f" --orientation-method none"
        )

        print(f"Running: {nerf_training_command}")
        os.system(nerf_training_command)


if __name__ == "__main__":
    cfg = tyro.cli(NerfTrainingConfig)
    main(cfg)
