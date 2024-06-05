import pathlib
from tqdm import tqdm
import tyro
from nerf_grasping import dexdiffuser_utils
from dataclasses import dataclass
from typing import Optional
from nerf_grasping.frogger_utils import FroggerArgs, frogger_to_grasp_config_dict, custom_coll_callback
from nerf_grasping.baselines.nerf_to_urdf_all import Args as nerf_to_urdf_all_Args, nerf_to_urdf_all


@dataclass
class CommandlineArgs:
    output_folder: pathlib.Path
    nerfcheckpoints_path: pathlib.Path
    ckpt_path: pathlib.Path = pathlib.Path(
        "/juno/u/tylerlum/github_repos/nerf_grasping/2024-06-03_ALBERT_DexDiffuser_models/ckpt_final.pth"
    )
    num_grasps: int = 32

    def __post_init__(self) -> None:
        assert (
            self.nerfcheckpoints_path.exists()
        ), f"{self.nerfcheckpoints_path} does not exist"


def main() -> None:
    """Steps:
    1. Generate nerf meshes
    2. Run Frogger
    """
    args = tyro.cli(CommandlineArgs)

    object_code_and_scale_strs = sorted(
        x.name for x in args.nerfcheckpoints_path.iterdir()
    )
    object_codes, object_scales = [], []
    for object_code_and_scale_str in object_code_and_scale_strs:
        idx = object_code_and_scale_str.index("_0_")
        object_code = object_code_and_scale_str[:idx]
        object_scale = float(object_code_and_scale_str[idx + 1 :].replace("_", "."))
        object_codes.append(object_code)
        object_scales.append(object_scale)

    nerf_to_mesh_folder = args.output_folder.parent / "nerf_meshdata"
    nerf_to_mesh_folder.mkdir(exist_ok=True, parents=True)
    nerf_to_urdf_all(
        args=nerf_to_urdf_all_Args(
            nerfcheckpoints_path=args.nerfcheckpoints_path,
            nerf_is_z_up=False,
            density_of_0_level_set=15.0,
            rescale=True,
            output_dir_path=nerf_to_mesh_folder,
        ),
    )

    for object_code, object_scale in tqdm(zip(object_codes, object_scales), total=len(object_codes), desc="frogger"):
        obj_path = nerf_to_mesh_folder / object_code / "coacd" / "decomposed.obj"
        assert obj_path.exists(), f"{obj_path} does not exist"
        frogger_to_grasp_config_dict(
            args=FroggerArgs(
                obj_filepath=obj_path,
                obj_scale=object_scale,
                obj_name=object_code,
                obj_is_yup=True,
                num_grasps=args.num_grasps,
                output_grasp_config_dicts_folder=args.output_folder,
                max_time=20.0,
                X_W_O=None,
                mesh=None,
                custom_coll_callback=custom_coll_callback,
            )
        )


if __name__ == "__main__":
    main()
