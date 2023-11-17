import subprocess
import tyro
import pathlib
from dataclasses import dataclass
import nerf_grasping


@dataclass
class Args:
    experiment_name: str
    max_num_iterations: int
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

    assert (
        args.nerf_grasping_data_path.exists()
    ), f"{args.nerf_grasping_data_path} does not exist"
    experiment_path = args.nerf_grasping_data_path / args.experiment_name
    assert experiment_path.exists(), f"{experiment_path} does not exist"

    phases = ["train", "val", "test"]
    for phase in phases:
        command = " ".join(
            [
                "python train_nerfs.py",
                f"--experiment_name={args.experiment_name}",
                f"--max_num_iterations={args.max_num_iterations}",
                f"--nerfdata_name=nerfdata_{phase}",
                f"--output_nerfcheckpoints_name=nerfcheckpoints_{phase}"
                f"--nerf_grasping_data_path={args.nerf_grasping_data_path}",
            ]
        )
        print_and_run(command)


if __name__ == "__main__":
    main()
