import tyro
import pathlib
import subprocess
from typing import Optional
from tqdm import tqdm

from dataclasses import dataclass
from nerf_grasping.grasp_utils import get_nerf_configs


@dataclass
class Args:
    nerfcheckpoints_path: pathlib.Path
    bounding_cube_half_length: float = 0.2
    density_of_0_level_set: float = 15.0
    n_pts_each_dim_marching_cubes: int = 31
    rescale: bool = True
    min_num_edges: Optional[int] = 100
    output_dir_path: pathlib.Path = pathlib.Path(__file__).parent / "nerf_meshdata"


def print_and_run(cmd: str) -> None:
    print(cmd)
    subprocess.run(cmd, shell=True)


def main() -> None:
    args = tyro.cli(Args)
    print("=" * 80)
    print(f"{pathlib.Path(__file__).name} args: {args}")
    print("=" * 80 + "\n")

    assert (
        args.nerfcheckpoints_path.exists()
    ), f"{args.nerfcheckpoints_path} does not exist"
    nerf_configs = get_nerf_configs(args.nerfcheckpoints_path)
    for nerf_config in tqdm(nerf_configs):
        command = " ".join(
            [
                "python nerf_grasping/baselines/nerf_to_urdf.py",
                f"--nerfcheckpoint-filepath {nerf_config}",
                f"--bounding-cube-half-length {args.bounding_cube_half_length}",
                f"--density-of-0-level-set {args.density_of_0_level_set}",
                f"--n-pts-each-dim-marching-cubes {args.n_pts_each_dim_marching_cubes}",
                "--rescale" if args.rescale else "--no-rescale",
                f"--min-num-edges {args.min_num_edges}",
                f"--output-dir-path {str(args.output_dir_path)}",
            ]
        )
        print_and_run(command)


if __name__ == "__main__":
    main()
