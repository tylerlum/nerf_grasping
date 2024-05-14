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
    nerf_is_z_up: bool
    nerfcheckpoints_name: str = "nerfcheckpoints"
    output_pointclouds_name: str = "pointclouds"
    nerf_grasping_data_path: pathlib.Path = (
        pathlib.Path(nerf_grasping.get_repo_root()).resolve() / "data"
    )
    randomize_order_seed: Optional[int] = None
    timeout: float = 60.0
    num_points: int = 5000

    @property
    def bounding_box_min(self) -> str:
        if self.nerf_is_z_up:
            return "-0.2 -0.2 0.0"
        else:
            return "-0.2 0.0 -0.2"

    @property
    def bounding_box_max(self) -> str:
        if self.nerf_is_z_up:
            return "0.2 0.2 0.3"
        else:
            return "0.2 0.3 0.2"


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
    print(f"Found {len(nerf_configs)} NERF configs")

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

        # Need to do this here to avoid error when saving the pointclouds
        print(f"Creating {output_path_to_be_created}")
        output_path_to_be_created.mkdir(exist_ok=False)

        command = " ".join(
            [
                "ns-export",
                "pointcloud",
                f"--load-config {str(nerf_config)}",
                f"--output-dir {str(output_path_to_be_created)}",
                "--normal-method open3d",
                f"--bounding-box-min {args.bounding_box_min}",
                f"--bounding-box-max {args.bounding_box_max}",
                f"--num-points {args.num_points}",
            ]
        )

        # NOTE: Some nerfs are terrible, so computing point clouds never finishes.
        # In this case, we should know about this and move on.
        print(f"Running: {command}")
        try:
            subprocess.run(command, shell=True, check=True, timeout=args.timeout)
            print(f"Finished generating {output_path_to_be_created}")
            print("=" * 80)
        except subprocess.TimeoutExpired:
            print(f"Command timed out after {args.timeout} seconds: {command}")
            print("~" * 80)

            timeout_path = (
                experiment_path
                / f"{args.nerfcheckpoints_name}_{args.output_pointclouds_name}_timeout.txt"
            )
            print(f"Writing to {timeout_path}")
            with open(timeout_path, "a") as f:
                f.write(f"{object_code_and_scale_str}\n")

    return output_pointclouds_path


def main() -> None:
    args = tyro.cli(Args)
    print("=" * 80)
    print(f"{pathlib.Path(__file__).name} args: {args}")
    print("=" * 80 + "\n")
    export_pointclouds(args)


if __name__ == "__main__":
    main()
