import subprocess
import tyro
import pathlib
from dataclasses import dataclass
from typing import Optional, List
import random
import nerf_grasping


@dataclass
class Args:
    experiment_name: str
    frac_train: float = 0.8
    frac_val: float = 0.1
    random_seed: Optional[int] = None
    nerf_grasping_data_path: pathlib.Path = (
        pathlib.Path(nerf_grasping.get_repo_root()).resolve().parent / "data"
    )


def print_and_run(cmd: str) -> None:
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def create_symlinks(
    src: pathlib.Path, dest: pathlib.Path, folder_names: List[str]
) -> None:
    for folder_name in folder_names:
        src_folder = src / folder_name
        dest_folder = dest / folder_name
        assert src_folder.exists(), f"{src_folder} does not exist"
        if not dest_folder.exists():
            print_and_run(f"ln -s {str(src_folder)} {str(dest_folder)}")


def main() -> None:
    args = tyro.cli(Args)
    print("=" * 80)
    print(f"{pathlib.Path(__file__).name} args: {args}")
    print("=" * 80 + "\n")

    assert (
        args.nerf_grasping_data_path.exists()
    ), f"{args.nerf_grasping_data_path} does not exist"
    experiment_path = args.nerf_grasping_data_path / args.experiment_name
    assert experiment_path.exists(), f"{experiment_path} does not exist"

    nerfdata_path = args.nerf_grasping_data_path / "nerfdata"
    assert nerfdata_path.exists(), f"{nerfdata_path} does not exist"

    all_object_and_scale_folder_names = sorted(
        [f.name for f in nerfdata_path.iterdir() if f.is_dir()]
    )
    random.Random(args.random_seed).shuffle(all_object_and_scale_folder_names)

    n_train, n_val = (
        int(args.frac_train * len(all_object_and_scale_folder_names)),
        int(args.frac_val * len(all_object_and_scale_folder_names)),
    )
    n_test = len(all_object_and_scale_folder_names) - n_train - n_val
    print(f"n_train: {n_train}, n_val: {n_val}, n_test: {n_test}")
    print()

    train_folder_names = all_object_and_scale_folder_names[:n_train]
    val_folder_names = all_object_and_scale_folder_names[n_train : n_train + n_val]
    test_folder_names = all_object_and_scale_folder_names[n_train + n_val :]

    nerfdata_train_path = args.nerf_grasping_data_path / "nerfdata_train"
    nerfdata_val_path = args.nerf_grasping_data_path / "nerfdata_val"
    nerfdata_test_path = args.nerf_grasping_data_path / "nerfdata_test"

    for path in [nerfdata_train_path, nerfdata_val_path, nerfdata_test_path]:
        path.mkdir(exist_ok=True)

    create_symlinks(
        src=nerfdata_path, dest=nerfdata_train_path, folder_names=train_folder_names
    )
    create_symlinks(
        src=nerfdata_path, dest=nerfdata_val_path, folder_names=val_folder_names
    )
    create_symlinks(
        src=nerfdata_path, dest=nerfdata_test_path, folder_names=test_folder_names
    )


if __name__ == "__main__":
    main()
