import tyro
import pathlib
import subprocess
import numpy as np

from dataclasses import dataclass
from nerf_grasping.baselines.nerf_to_mesh import nerf_to_mesh
from nerf_grasping.grasp_utils import load_nerf_field


@dataclass
class Args:
    nerfcheckpoint_filepath: pathlib.Path
    bounding_cube_half_length: float = 0.2
    density_of_0_level_set: float = 15.0
    n_pts_each_dim_marching_cubes: int = 31
    output_dir_path: pathlib.Path = pathlib.Path(__file__).parent / "results"


def print_and_run(cmd: str) -> None:
    print(cmd)
    subprocess.run(cmd, shell=True)


def create_urdf(
    obj_path: pathlib.Path, ixx: float = 0.1, iyy: float = 0.1, izz: float = 0.1
) -> pathlib.Path:
    assert obj_path.exists(), f"{obj_path} does not exist"
    output_folder = obj_path.parent
    obj_filename = obj_path.name
    urdf_content = f"""<robot name="root">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{izz}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{obj_filename}" scale="1 1 1"/>
      </geometry>
      <material name="">
        <color rgba="0.75 0.75 0.75 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{obj_filename}" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
</robot>"""

    urdf_path = output_folder / (output_folder.name + ".urdf")
    with open(urdf_path, "w") as urdf_file:
        urdf_file.write(urdf_content)
    return urdf_path


def main() -> None:
    args = tyro.cli(Args)
    print("=" * 80)
    print(f"{pathlib.Path(__file__).name} args: {args}")
    print("=" * 80 + "\n")

    assert (
        args.nerfcheckpoint_filepath.exists()
    ), f"{args.nerfcheckpoint_filepath} does not exist"
    assert (
        args.nerfcheckpoint_filepath.name == "config.yml"
    ), f"{args.nerfcheckpoint_filepath} is not a config.yml file"
    assert (
        args.nerfcheckpoint_filepath.parent.parent.name == "nerfacto"
    ), f"{args.nerfcheckpoint_filepath.parent.parent.name} should be nerfacto"
    # Eg. path=data/2023-10-13_13-12-28/nerfcheckpoints/sem-RubiksCube-1e3d89eb3b5427053bdd31f1cd9eec98_0_1076/nerfacto/2023-10-13_131849/config.yml
    # name=sem-RubiksCube-1e3d89eb3b5427053bdd31f1cd9eec98_0_1076
    object_and_scale = args.nerfcheckpoint_filepath.parent.parent.parent.name

    nerf_field = load_nerf_field(args.nerfcheckpoint_filepath)
    lb = -args.bounding_cube_half_length * np.ones(3)
    ub = args.bounding_cube_half_length * np.ones(3)

    output_folder = args.output_dir_path / object_and_scale
    output_folder.mkdir(exist_ok=False, parents=True)

    obj_path = output_folder / f"{object_and_scale}.obj"
    nerf_to_mesh(
        nerf_field,
        level=args.density_of_0_level_set,
        npts=args.n_pts_each_dim_marching_cubes,
        lb=lb,
        ub=ub,
        save_path=obj_path,
    )

    urdf_path = create_urdf(obj_path)

    assert urdf_path.exists(), f"{urdf_path} does not exist"
    assert obj_path.exists(), f"{obj_path} does not exist"
    print(f"Created {urdf_path} and {obj_path}")


if __name__ == "__main__":
    main()
