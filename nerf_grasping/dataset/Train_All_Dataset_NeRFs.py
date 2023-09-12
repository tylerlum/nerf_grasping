from dataclasses import dataclass
import pathlib
import tyro
from typing import Optional
import os
import multiprocessing


@dataclass
class NerfTrainingConfig:
    """Configuration for training a NeRF model."""

    nerfdata_path: pathlib.Path = pathlib.Path("data/nerfdata")
    nerfcheckpoints_path: Optional[pathlib.Path] = None
    model: str = "nerfacto"
    num_iters: int = 200
    scene_scale: float = 0.2
    use_multiprocess: bool = True
    num_workers: int = 8
    continue_training: bool = True


def run_command(object_code_and_scale_str, cfg, nerfcheckpoints_path):
    nerf_training_command = (
        "ns-train nerfacto"
        + f" --data {cfg.nerfdata_path / object_code_and_scale_str}"
        + f" --max-num-iterations {cfg.num_iters}"
        + f" --output-dir {nerfcheckpoints_path}"
        + f" --vis wandb"
        + f" --viewer.quit-on-train-completion True"
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


def main(cfg: NerfTrainingConfig):
    print(cfg)
    object_code_and_scale_strs = [x.stem for x in cfg.nerfdata_path.iterdir()]
    if cfg.nerfcheckpoints_path is None:
        nerfcheckpoints_path = cfg.nerfdata_path.parent / "nerfcheckpoints"

    print(f"Found {len(object_code_and_scale_strs)} objects in {cfg.nerfdata_path}.")

    existing_object_code_and_scale_strs = (
        [path.stem for path in nerfcheckpoints_path.iterdir()]
        if nerfcheckpoints_path.exists()
        else []
    )

    if cfg.continue_training:
        # Get set of objects that have already been trained on

        print(
            f"Found {len(existing_object_code_and_scale_strs)} objects in {nerfcheckpoints_path}."
        )

        object_code_and_scale_strs = list(
            set(object_code_and_scale_strs) - set(existing_object_code_and_scale_strs)
        )

        print(f"Continuing training on {len(object_code_and_scale_strs)} objects.")
    elif len(existing_object_code_and_scale_strs) > 0:
        raise ValueError(
            f"Found {len(existing_object_code_and_scale_strs)} objects in {nerfcheckpoints_path}."
            + " Set continue_training to True to continue training on these objects, or change output path."
        )

    print(f"Saving checkpoints to {nerfcheckpoints_path}")

    if cfg.use_multiprocess:
        with multiprocessing.Pool(cfg.num_workers) as pool:
            pool.starmap(
                run_command,
                [
                    (object_code_and_scale_str, cfg, nerfcheckpoints_path)
                    for object_code_and_scale_str in object_code_and_scale_strs
                ],
            )
    else:
        for object_code_and_scale_str in object_code_and_scale_strs:
            run_command(object_code_and_scale_str, cfg, nerfcheckpoints_path)


if __name__ == "__main__":
    cfg = tyro.cli(NerfTrainingConfig)
    main(cfg)
