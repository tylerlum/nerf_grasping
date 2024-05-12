import subprocess
import tyro
import pathlib
from dataclasses import dataclass
import nerf_grasping
from tqdm import tqdm
from typing import Optional
from nerf_grasping.grasp_utils import get_nerf_configs


@dataclass
class Args:
    experiment_name: str
    nerfcheckpoints_name: str = "nerfcheckpoints"
    output_pointclouds_name: str = "pointclouds"
    nerf_grasping_data_path: pathlib.Path = (
        pathlib.Path(nerf_grasping.get_repo_root()).resolve() / "data"
    )
    randomize_order_seed: Optional[int] = None


def print_and_run(cmd: str) -> None:
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def export_pointclouds(args: Args) -> pathlib.Path:
    assert (
        args.nerf_grasping_data_path.exists()
    ), f"{args.nerf_grasping_data_path} does not exist"
    experiment_path = args.nerf_grasping_data_path / args.experiment_name
    assert experiment_path.exists(), f"{experiment_path} does not exist"

    nerfcheckpoints_path = experiment_path / args.nerfcheckpoints_name
    assert nerfcheckpoints_path.exists(), f"{nerfcheckpoints_path} does not exist"

    output_pointclouds_path = experiment_path / args.output_pointclouds_name
    output_pointclouds_path.mkdir(exist_ok=True)

    nerf_configs = get_nerf_configs(nerfcheckpoints_path)

    if args.randomize_order_seed is not None:
        import random

        print(f"Randomizing order with seed {args.randomize_order_seed}")
        random.Random(args.randomize_order_seed).shuffle(nerf_configs)

    for nerf_config in tqdm(
        nerf_configs, dynamic_ncols=True, desc="Exporting Pointclouds"
    ):
        object_code_and_scale_str = nerf_config.parents[2].name
        output_path_to_be_created = output_pointclouds_path / object_code_and_scale_str
        if output_path_to_be_created.exists():
            print(f"Skipping {output_path_to_be_created} because it already exists")
            continue

        command = " ".join(
            [
                "ns-export",
                "pointcloud",
                f"--load-config {str(nerf_config)}",
                f"--output-dir {str(output_path_to_be_created)}",
                "--normal-method open3d",
                "--bounding-box-min -0.2 -0.2 0.0",
                "--bounding-box-max 0.2 0.2 0.3",
                "--num-points 5000",
            ]
        )
        print_and_run(command)
    return output_pointclouds_path


def main() -> None:
    args = tyro.cli(Args)
    print("=" * 80)
    print(f"{pathlib.Path(__file__).name} args: {args}")
    print("=" * 80 + "\n")
    export_pointclouds(args)


if __name__ == "__main__":
    main()
