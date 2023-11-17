import subprocess
import tyro
import pathlib
from dataclasses import dataclass
import nerf_grasping


@dataclass
class Args:
    experiment_name: str
    max_num_iterations: int = 200
    nerfdata_name: str = "nerfdata"
    output_nerfcheckpoints_name: str = "nerfcheckpoints"
    nerf_grasping_data_path: pathlib.Path = (
        pathlib.Path(nerf_grasping.get_repo_root()).resolve() / "data"
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

    nerfdata_path = experiment_path / args.nerfdata_name
    assert nerfdata_path.exists(), f"{nerfdata_path} does not exist"

    output_nerfcheckpoints_path = experiment_path / args.output_nerfcheckpoints_name
    output_nerfcheckpoints_path.mkdir(exist_ok=True)

    for object_and_scale_nerfdata_path in nerfdata_path.iterdir():
        if not object_and_scale_nerfdata_path.is_dir():
            continue

        output_path_to_be_created = (
            output_nerfcheckpoints_path / object_and_scale_nerfdata_path.name
        )
        if output_path_to_be_created.exists():
            print(f"Skipping {output_path_to_be_created} because it already exists")
            continue

        command = " ".join(
            [
                "ns-train nerfacto",
                f"--data {str(object_and_scale_nerfdata_path)}",
                f"--max-num-iterations {args.max_num_iterations}",
                f"--output-dir {str(output_nerfcheckpoints_path)}",
                "--vis wandb",
                "--pipeline.model.disable-scene-contraction True",
                "--pipeline.model.background-color black",
                "nerfstudio-data",
                "--auto-scale-poses False",
                "--scale-factor 1.",
                "--scene-scale 0.2",
                "--center-method none",
                "--orientation-method none",
            ]
        )
        print_and_run(command)


if __name__ == "__main__":
    main()
