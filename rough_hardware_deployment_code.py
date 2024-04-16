from nerf_grasping.optimizer import get_optimized_grasps as TYLER_get_optimized_grasps
from nerf_grasping.optimizer_utils import (
    get_sorted_grasps_from_dict as TYLER_get_sorted_grasps_from_dict,
)
from nerf_grasping.config.optimization_config import OptimizationConfig
from nerf_grasping.config.optimizer_config import SGDOptimizerConfig
from nerf_grasping.config.grasp_metric_config import GraspMetricConfig
from nerf_grasping.nerfstudio_train import train_nerfs
from nerf_grasping.baselines.nerf_to_mesh import nerf_to_mesh
import trimesh
import nerf_grasping
import pathlib
import tyro
import numpy as np
from dataclasses import dataclass
from nerf_grasping.grasp_utils import (
    get_nerf_configs,
    load_nerf_pipeline,
)


@dataclass
class Args:
    experiment_name: str
    init_grasp_config_dict_path: pathlib.Path
    classifier_config_path: pathlib.Path

    def __post_init__(self) -> None:
        assert (
            self.init_grasp_config_dict_path.exists()
        ), f"{self.init_grasp_config_dict_path} does not exist"
        assert (
            self.classifier_config_path.exists()
        ), f"{self.classifier_config_path} does not exist"

        assert (
            self.init_grasp_config_dict_path.is_file()
        ), f"{self.init_grasp_config_dict_path} is not a file"
        assert (
            self.classifier_config_path.is_file()
        ), f"{self.classifier_config_path} is not a file"

        assert self.init_grasp_config_dict_path.suffix == ".npy"(
            f"{self.init_grasp_config_dict_path} does not have a .npy suffix"
        )
        assert self.classifier_config_path.suffix in [
            ".yml",
            ".yaml",
        ], f"{self.classifier_config_path} does not have a .yml or .yaml suffix"


def rough_hardware_deployment_code(args: Args) -> None:
    print("=" * 80)
    print("Step 0: Figuring out frames")
    print("=" * 80 + "\n")
    print("Frames are W = world, N = nerf, O = object, Oy = object y-up, H = hand")
    print(
        "W is centered at the robot base. N is centered where origin of NeRF data collection is. O is centered at the object centroid. Oy is centered at the object centroid. H is centered at the base of the middle finger"
    )
    print(
        "W, N, O are z-up frames. Oy is y-up. H has z-up along finger and x-up along palm normal"
    )
    print("X_A_B represents 4x4 transformation matrix of frame B wrt A")
    X_W_N = trimesh.transformations.translation_matrix([0.7, 0, 0])  # TODO: Check this

    print("\n" + "=" * 80)
    print("Step 1: Collect NERF data")
    print("=" * 80 + "\n")
    nerfdata_folder = pathlib.Path(f"{args.experiment_name}/nerfdata")
    nerfdata_folder.mkdir(parents=True, exist_ok=False)
    nerf_data = ALBERT_run_hardware_nerf_data_collection(nerfdata_folder)
    assert nerfdata_folder.exists(), f"{nerfdata_folder} does not exist"

    print("\n" + "=" * 80)
    print("Step 2: Train NERF")
    print("=" * 80 + "\n")
    nerf_checkpoint_path = train_nerfs.train_nerfs(
        train_nerfs.Args(
            experiment_name=args.experiment_name,
            nerf_grasping_data_path=pathlib.Path(
                nerf_grasping.get_repo_root()
            ).resolve()
            / "data",
            is_real_world=True,
        )
    )
    assert nerf_checkpoint_path.exists(), f"{nerf_checkpoint_path} does not exist"

    print("\n" + "=" * 80)
    print("Step 3: Load NERF")
    print("=" * 80 + "\n")
    nerf_configs = get_nerf_configs(str(nerf_checkpoint_path))
    assert len(nerf_configs) == 1, f"len(nerf_configs) is {len(nerf_configs)}, not 1"
    nerf_pipeline = load_nerf_pipeline(nerf_configs[0])

    print("\n" + "=" * 80)
    print("Step 3: Convert NeRF to mesh")
    print("=" * 80 + "\n")
    mesh = nerf_to_mesh(
        field=nerf_pipeline.model.field,
        level=15,
        lb=np.array([-0.25, 0.25, 0.0]),
        ub=np.array([0.25, 0.25, 0.3]),
    )  # TODO: Maybe tune other default params, but prefer not to need to

    print("\n" + "=" * 80)
    print(
        "Step 4: Compute X_N_Oy (transformation of the object y-up frame wrt the nerf frame)"
    )
    X_O_Oy = trimesh.transformations.rotation_matrix(
        np.pi / 2, [1, 0, 0]
    )  # TODO: Check this

    USE_MESH = True
    if USE_MESH:
        centroid = mesh.centroid
    else:
        centroid = compute_centroid_from_nerf(nerf_pipeline.model.field)
    assert centroid.shape == (3,), f"centroid.shape is {centroid.shape}, not (3,)"
    X_N_O = trimesh.transformations.translation_matrix(centroid)  # TODO: Check this

    X_N_Oy = X_N_O @ X_O_Oy
    assert X_N_Oy.shape == (4, 4), f"X_N_Oy.shape is {X_N_Oy.shape}, not (4, 4)"

    print("\n" + "=" * 80)
    print("Step 5: Load grasp metric and init grasps, optimize")
    print("=" * 80 + "\n")
    optimized_grasp_config_dict = TYLER_get_optimized_grasps(
        OptimizationConfig(
            use_rich=True,
            init_grasp_config_dict_path=args.init_grasp_config_dict_path,
            grasp_metric=GraspMetricConfig(
                nerf_checkpoint_path=nerf_checkpoint_path,
                classifier_config_path=args.classifier_config_path,
                X_N_Oy=X_N_Oy,
            ),
            optimizer=SGDOptimizerConfig(),
        )
    )

    print("\n" + "=" * 80)
    print("Step 6: Compute best grasps in W frame")
    X_Oy_H_array, joint_angles_array, target_joint_angles_array = (
        TYLER_get_sorted_grasps_from_dict(
            optimized_grasp_config_dict=optimized_grasp_config_dict,
            X_N_Oy=X_N_Oy,
            error_if_no_loss=False,
            check=True,
            print_best=True,
        )
    )
    num_grasps = X_Oy_H_array.shape[0]
    assert X_Oy_H_array.shape == (num_grasps, 4, 4)
    assert joint_angles_array.shape == (num_grasps, 16)
    assert target_joint_angles_array.shape == (num_grasps, 16)

    for i in range(num_grasps):
        print(f"Trying grasp {i} / {num_grasps}")

        X_Oy_H = X_Oy_H_array[i]
        joint_angles = joint_angles_array[i]
        target_joint_angles = target_joint_angles_array[i]

        X_W_H = X_W_N @ X_N_Oy @ X_Oy_H

        if not ALBERT_is_feasible(X_Oy_H, joint_angles):
            print(f"Grasp {i} is infeasible")
            continue

        ALBERT_execute_grasp(X_W_H, joint_angles, target_joint_angles)


def main() -> None:
    args = tyro.cli(Args)
    print("=" * 80)
    print(f"{pathlib.Path(__file__).name} args: {args}")
    print("=" * 80 + "\n")
    return rough_hardware_deployment_code(args)


if __name__ == "__main__":
    main()
