import subprocess
import tyro
import pathlib
from dataclasses import dataclass
import nerf_grasping


@dataclass
class Args:
    experiment_name: str
    dgn_data_path: pathlib.Path = pathlib.Path(
        "~/github_repos/DexGraspNet/data"
    ).expanduser()  # TODO: change this per workstation
    nerf_grasping_data_path: pathlib.Path = (
        pathlib.Path(nerf_grasping.get_repo_root()).resolve().parent / "data"
    )


def print_and_run(cmd: str) -> None:
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main() -> None:
    args = tyro.cli(Args)
    print("=" * 80)
    print(f"{pathlib.Path(__file__).name} args: {args}")
    print("=" * 80 + "\n")

    assert args.dgn_data_path.exists(), f"{args.dgn_data_path} does not exist"
    experiment_path = args.dgn_data_path / args.experiment_name
    assert experiment_path.exists(), f"{experiment_path} does not exist"
    assert (
        args.nerf_grasping_data_path.exists()
    ), f"{args.nerf_grasping_data_path} does not exist"

    new_experiment_path = args.nerf_grasping_data_path / args.experiment_name
    if new_experiment_path.exists():
        print(f"Skipping {new_experiment_path} because it already exists")
    else:
        print_and_run(f"ln -s {experiment_path} {new_experiment_path}")


if __name__ == "__main__":
    main()
