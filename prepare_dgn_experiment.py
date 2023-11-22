import subprocess
import tyro
import pathlib
from dataclasses import dataclass
import nerf_grasping
from typing import Optional, List
import random
import math


@dataclass
class Args:
    experiment_name: str
    frac_train: float = 0.8
    frac_val: float = 0.1
    random_seed: Optional[int] = None
    train_nerfs: bool = False

    dgn_data_path: pathlib.Path = pathlib.Path(
        "~/github_repos/DexGraspNet/data"
    ).expanduser()  # TODO: change this per workstation
    nerf_grasping_data_path: pathlib.Path = (
        pathlib.Path(nerf_grasping.get_repo_root()).resolve() / "data"
    )


def print_and_run(cmd: str) -> None:
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def create_symlinks(
    src_folderpath: pathlib.Path, dest_folderpath: pathlib.Path, filenames: List[str]
) -> None:
    dest_folderpath.mkdir(exist_ok=True)
    for filename in filenames:
        src_filename = src_folderpath / filename
        dest_filename = dest_folderpath / filename
        assert src_filename.exists(), f"{src_filename} does not exist"
        if not dest_filename.exists():
            print_and_run(f"ln -s {str(src_filename)} {str(dest_filename)}")


def main() -> None:
    args = tyro.cli(Args)
    print("=" * 80)
    print(f"{pathlib.Path(__file__).name} args: {args}")
    print("=" * 80 + "\n")

    # Sanity check
    assert args.dgn_data_path.exists(), f"{args.dgn_data_path} does not exist"
    experiment_path = args.dgn_data_path / args.experiment_name
    assert experiment_path.exists(), f"{experiment_path} does not exist"
    assert (
        args.nerf_grasping_data_path.exists()
    ), f"{args.nerf_grasping_data_path} does not exist"

    # Create symlink to data
    new_experiment_path = args.nerf_grasping_data_path / args.experiment_name
    if new_experiment_path.exists():
        print(f"Skipping {new_experiment_path} because it already exists")
    else:
        print_and_run(f"ln -s {experiment_path} {new_experiment_path}")

    # Create train/val/test split
    evaled_grasp_config_dicts_path = new_experiment_path / "evaled_grasp_config_dicts"
    assert (
        evaled_grasp_config_dicts_path.exists()
    ), f"{evaled_grasp_config_dicts_path} does not exist"

    filenames = sorted(
        [f.name for f in evaled_grasp_config_dicts_path.iterdir()]
    )
    assert len(filenames) > 0, f"{evaled_grasp_config_dicts_path} is empty"
    random.Random(args.random_seed).shuffle(filenames)

    n_train, n_val = (
        math.ceil(args.frac_train * len(filenames)),
        int(args.frac_val * len(filenames)),
    )
    n_test = len(filenames) - n_train - n_val
    print(f"n_train: {n_train}, n_val: {n_val}, n_test: {n_test}")
    print()

    train_filenames = filenames[:n_train]
    val_filenames = filenames[n_train : n_train + n_val]
    test_filenames = filenames[n_train + n_val :]

    # Create symlinks to train/val/test split
    evaled_grasp_config_dicts_train_path = (
        new_experiment_path / "evaled_grasp_config_dicts_train"
    )
    evaled_grasp_config_dicts_val_path = (
        new_experiment_path / "evaled_grasp_config_dicts_val"
    )
    evaled_grasp_config_dicts_test_path = (
        new_experiment_path / "evaled_grasp_config_dicts_test"
    )

    for new_path, filenames in [
        (evaled_grasp_config_dicts_train_path, train_filenames),
        (evaled_grasp_config_dicts_val_path, val_filenames),
        (evaled_grasp_config_dicts_test_path, test_filenames),
    ]:
        create_symlinks(
            src_folderpath=evaled_grasp_config_dicts_path,
            dest_folderpath=new_path,
            filenames=filenames,
        )

    # Train nerfs
    if args.train_nerfs:
        print_and_run(
            f"python nerf_grasping/nerfstudio_train/train_nerfs.py"
            + f" --experiment_name {args.experiment_name}"
        )

    # Create dataset
    print_and_run(
        f"python nerf_grasping/dataset/Create_DexGraspNet_NeRF_Grasps_Dataset.py grid"
        + f" --evaled-grasp-config-dicts-path {new_experiment_path / 'evaled_grasp_config_dicts_train'}"
        + f" --nerf-checkpoints-path {new_experiment_path / 'nerfcheckpoints'}"
        + f" --output-filepath {new_experiment_path / 'grid_dataset.h5'}"
    )
    print_and_run(
        f"python nerf_grasping/dataset/Create_DexGraspNet_NeRF_Grasps_Dataset.py depth-image"
        + f" --evaled-grasp-config-dicts-path {new_experiment_path / 'evaled_grasp_config_dicts_train'}"
        + f" --nerf-checkpoints-path {new_experiment_path / 'nerfcheckpoints'}"
        + f" --output-filepath {new_experiment_path / 'depth_image_dataset.h5'}"
    )


if __name__ == "__main__":
    main()
