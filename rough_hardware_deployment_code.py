from nerf_grasping.optimizer import get_optimized_grasps
from nerf_grasping.dataset.DexGraspNet_NeRF_Grasps_utils import (
    parse_object_code_and_scale,
)
from nerf_grasping.optimizer_utils import (
    get_sorted_grasps_from_dict,
    GraspMetric,
    DepthImageGraspMetric,
    load_classifier,
    load_depth_image_classifier,
)
from nerf_grasping.config.nerfdata_config import DepthImageNerfDataConfig
from nerf_grasping.config.optimization_config import OptimizationConfig
from nerf_grasping.config.optimizer_config import SGDOptimizerConfig
from nerf_grasping.config.grasp_metric_config import GraspMetricConfig
from nerf_grasping.nerfstudio_train import train_nerfs_return_trainer
from nerf_grasping.baselines.nerf_to_mesh import nerf_to_mesh
from nerf_grasping.nerf_utils import (
    compute_centroid_from_nerf,
)
from nerf_grasping.config.classifier_config import ClassifierConfig
import trimesh
import pathlib
import tyro
import numpy as np
from dataclasses import dataclass
import plotly.graph_objects as go


@dataclass
class Args:
    experiment_name: str
    init_grasp_config_dict_path: pathlib.Path
    classifier_config_path: pathlib.Path
    object_name: str
    experiments_folder: pathlib.Path = pathlib.Path("experiments")
    is_real_world: bool = False
    density_levelset_threshold: float = 15.0
    visualize: bool = False

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

        assert (
            self.init_grasp_config_dict_path.suffix == ".npy"
        ), f"{self.init_grasp_config_dict_path} does not have a .npy suffix"
        assert self.classifier_config_path.suffix in [
            ".yml",
            ".yaml",
        ], f"{self.classifier_config_path} does not have a .yml or .yaml suffix"


def transform_point(transform_matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    assert transform_matrix.shape == (4, 4), f"{transform_matrix.shape} is not (4, 4)"
    assert point.shape == (3,), f"{point.shape} is not (3,)"
    point = np.append(point, 1)
    return np.dot(transform_matrix, point)[:3]


def add_transform_matrix_traces(
    fig: go.Figure, transform_matrix: np.ndarray, length: float = 0.1
) -> None:
    assert transform_matrix.shape == (4, 4), f"{transform_matrix.shape} is not (4, 4)"
    origin = np.array([0, 0, 0])
    x_axis = np.array([length, 0, 0])
    y_axis = np.array([0, length, 0])
    z_axis = np.array([0, 0, length])

    origin_transformed = transform_point(transform_matrix, origin)
    x_axis_transformed = transform_point(transform_matrix, x_axis)
    y_axis_transformed = transform_point(transform_matrix, y_axis)
    z_axis_transformed = transform_point(transform_matrix, z_axis)

    for axis, color, name in zip(
        [x_axis_transformed, y_axis_transformed, z_axis_transformed],
        ["red", "green", "blue"],
        ["x", "y", "z"],
    ):
        fig.add_trace(
            go.Scatter3d(
                x=[origin_transformed[0], axis[0]],
                y=[origin_transformed[1], axis[1]],
                z=[origin_transformed[2], axis[2]],
                mode="lines",
                line=dict(color=color, width=5),
                name=name,
            )
        )


def get_hacky_table_mesh(
    table_y_Oy: float,
    W: float = 0.25,
    H: float = 0.25,
) -> trimesh.Trimesh:
    table_pos_Oy = np.array([0, table_y_Oy, 0])
    table_normal_Oy = np.array([0, 1, 0])
    table_parallel_Oy = np.array([1, 0, 0])
    assert table_pos_Oy.shape == table_normal_Oy.shape == (3,)

    table_parallel_2_Oy = np.cross(table_normal_Oy, table_parallel_Oy)
    corner1 = table_pos_Oy + W / 2 * table_parallel_Oy + H / 2 * table_parallel_2_Oy
    corner2 = table_pos_Oy + W / 2 * table_parallel_Oy - H / 2 * table_parallel_2_Oy
    corner3 = table_pos_Oy - W / 2 * table_parallel_Oy + H / 2 * table_parallel_2_Oy
    corner4 = table_pos_Oy - W / 2 * table_parallel_Oy - H / 2 * table_parallel_2_Oy

    x = np.array([corner1[0], corner2[0], corner3[0], corner4[0]])
    y = np.array([corner1[1], corner2[1], corner3[1], corner4[1]])
    z = np.array([corner1[2], corner2[2], corner3[2], corner4[2]])

    i = [0, 0, 1]
    j = [1, 2, 2]
    k = [2, 3, 3]

    table_mesh = trimesh.Trimesh(
        vertices=np.stack([x, y, z], axis=1), faces=np.stack([i, j, k], axis=1)
    )
    return table_mesh


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
    X_W_N = trimesh.transformations.translation_matrix([0.7, 0, 0])

    if args.is_real_world:
        # Z-up
        X_O_Oy = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        lb_N = np.array([-0.05, -0.05, 0.0])
        ub_N = np.array([0.05, 0.05, 0.3])
    else:
        IS_Y_UP = True
        if IS_Y_UP:
            X_O_Oy = np.eye(4)
        else:
            X_O_Oy = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        lb_N = np.array([-0.05, -0.15, -0.05])
        ub_N = np.array([0.05, 0.15, 0.05])

    experiment_folder = args.experiments_folder / args.experiment_name
    print(f"Creating a new experiment folder at {experiment_folder}")
    experiment_folder.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("Step 1: Collect NERF data")
    print("=" * 80 + "\n")
    object_nerfdata_folder = experiment_folder / "nerfdata" / args.object_name
    if not object_nerfdata_folder.exists():
        assert args.is_real_world, "NeRF data must be collected in the real world"
        object_nerfdata_folder.mkdir(parents=True)
        ALBERT_run_hardware_nerf_data_collection(object_nerfdata_folder)
        assert (
            object_nerfdata_folder.exists()
        ), f"{object_nerfdata_folder} does not exist"
    else:
        print(f"{object_nerfdata_folder} already exists, skipping data collection")
    assert (
        object_nerfdata_folder / "transforms.json"
    ).exists(), f"{object_nerfdata_folder / 'transforms.json'} does not exist"
    assert (
        object_nerfdata_folder / "images"
    ).exists(), f"{object_nerfdata_folder / 'images'} does not exist"

    print("\n" + "=" * 80)
    print("Step 2: Train NERF")
    print("=" * 80 + "\n")
    nerf_trainer = train_nerfs_return_trainer.train_nerf(
        args=train_nerfs_return_trainer.Args(
            nerfdata_folder=object_nerfdata_folder,
            nerfcheckpoints_folder=experiment_folder / "nerfcheckpoints",
        )
    )
    nerf_model = nerf_trainer.pipeline.model
    nerf_field = nerf_model.field
    nerf_config = nerf_trainer.config.get_base_dir() / "config.yml"
    assert nerf_config.exists(), f"{nerf_config} does not exist"
    print(f"NERF config saved at {nerf_config}")

    print("\n" + "=" * 80)
    print("Step 3: Convert NeRF to mesh")
    print("=" * 80 + "\n")
    nerf_to_mesh_file = (
        experiment_folder
        / "nerf_to_mesh"
        / args.object_name
        / "coacd"
        / "decomposed.obj"
    )
    nerf_to_mesh_file.parent.mkdir(parents=True, exist_ok=True)
    mesh_N = nerf_to_mesh(
        field=nerf_field,
        level=args.density_levelset_threshold,
        lb=lb_N,
        ub=ub_N,
        save_path=nerf_to_mesh_file,
    )

    print("\n" + "=" * 80)
    print(
        "Step 4: Compute X_N_Oy (transformation of the object y-up frame wrt the nerf frame)"
    )
    print("=" * 80 + "\n")
    USE_MESH = False
    mesh_centroid_N = mesh_N.centroid
    nerf_centroid_N = compute_centroid_from_nerf(
        nerf_field,
        lb=lb_N,
        ub=ub_N,
        level=args.density_levelset_threshold,
        num_pts_x=100,
        num_pts_y=100,
        num_pts_z=100,
    )
    print(f"mesh_centroid_N: {mesh_centroid_N}")
    print(f"nerf_centroid_N: {nerf_centroid_N}")
    centroid_N = mesh_centroid_N if USE_MESH else nerf_centroid_N
    print(f"USE_MESH: {USE_MESH}, centroid_N: {centroid_N}")
    assert centroid_N.shape == (3,), f"centroid_N.shape is {centroid_N.shape}, not (3,)"
    X_N_O = trimesh.transformations.translation_matrix(centroid_N)

    X_N_Oy = X_N_O @ X_O_Oy
    X_Oy_N = np.linalg.inv(X_N_Oy)
    assert X_N_Oy.shape == (4, 4), f"X_N_Oy.shape is {X_N_Oy.shape}, not (4, 4)"

    # For debugging
    mesh_Oy = trimesh.Trimesh(vertices=mesh_N.vertices, faces=mesh_N.faces)
    mesh_Oy.apply_transform(X_Oy_N)
    nerf_to_mesh_Oy_file = (
        experiment_folder
        / "nerf_to_mesh_Oy"
        / args.object_name
        / "coacd"
        / "decomposed.obj"
    )
    nerf_to_mesh_Oy_file.parent.mkdir(parents=True, exist_ok=True)
    mesh_Oy.export(nerf_to_mesh_Oy_file)
    mesh_centroid_Oy = transform_point(X_Oy_N, centroid_N)
    nerf_centroid_Oy = transform_point(X_Oy_N, centroid_N)

    if args.is_real_world:
        print("Assuming table is at z = 0 in nerf frame")
        _, _, centroid_z_N = centroid_N
        table_y_Oy = -centroid_z_N
    else:
        print("Using ground truth table mesh bounding box min y to get table height")
        # If in sim, assume it is a parsable object_code_and_scale_str that we can get ground truth mesh of to get the table height

        # This isn't cheating because in real world we would know the table height in advance
        try:
            object_code, object_scale = parse_object_code_and_scale(args.object_name)
            meshdata_folder = pathlib.Path(
                "/juno/u/tylerlum/github_repos/DexGraspNet/data/rotated_meshdata"
            )
            assert meshdata_folder.exists(), f"{meshdata_folder} does not exist"
            true_mesh_Oy = trimesh.load(
                meshdata_folder / object_code / "coacd" / "decomposed.obj"
            )
            true_mesh_Oy.apply_scale(object_scale)
        except Exception as e:
            print(f"ERROR: {e}")
            print("Using nerf mesh as ground truth table mesh")
            true_mesh_Oy = mesh_Oy
        true_mesh_Oy_min_bounds, _ = true_mesh_Oy.bounds
        _, true_mesh_Oy_min_y, _ = true_mesh_Oy_min_bounds
        table_y_Oy = true_mesh_Oy_min_y
    print(f"table_y_Oy: {table_y_Oy}")

    if args.visualize:
        # Visualize N
        fig_N = go.Figure()
        fig_N.add_trace(
            go.Mesh3d(
                x=mesh_N.vertices[:, 0],
                y=mesh_N.vertices[:, 1],
                z=mesh_N.vertices[:, 2],
                i=mesh_N.faces[:, 0],
                j=mesh_N.faces[:, 1],
                k=mesh_N.faces[:, 2],
                color="lightblue",
                name="Mesh",
                opacity=0.5,
            )
        )
        fig_N.add_trace(
            go.Scatter3d(
                x=[mesh_centroid_N[0]],
                y=[mesh_centroid_N[1]],
                z=[mesh_centroid_N[2]],
                mode="markers",
                marker=dict(size=10, color="red"),
                name="Mesh centroid",
            )
        )
        fig_N.add_trace(
            go.Scatter3d(
                x=[nerf_centroid_N[0]],
                y=[nerf_centroid_N[1]],
                z=[nerf_centroid_N[2]],
                mode="markers",
                marker=dict(size=10, color="green"),
                name="NeRF centroid",
            )
        )
        fig_N.update_layout(title="Mesh in nerf frame")
        add_transform_matrix_traces(fig=fig_N, transform_matrix=np.eye(4), length=0.1)
        fig_N.show()

        # Visualize Oy
        fig_Oy = go.Figure()
        fig_Oy.add_trace(
            go.Mesh3d(
                x=mesh_Oy.vertices[:, 0],
                y=mesh_Oy.vertices[:, 1],
                z=mesh_Oy.vertices[:, 2],
                i=mesh_Oy.faces[:, 0],
                j=mesh_Oy.faces[:, 1],
                k=mesh_Oy.faces[:, 2],
                color="lightblue",
                name="Mesh",
                opacity=0.5,
            )
        )
        fig_Oy.add_trace(
            go.Scatter3d(
                x=[mesh_centroid_Oy[0]],
                y=[mesh_centroid_Oy[1]],
                z=[mesh_centroid_Oy[2]],
                mode="markers",
                marker=dict(size=10, color="red"),
                name="Mesh centroid",
            )
        )
        fig_Oy.add_trace(
            go.Scatter3d(
                x=[nerf_centroid_Oy[0]],
                y=[nerf_centroid_Oy[1]],
                z=[nerf_centroid_Oy[2]],
                mode="markers",
                marker=dict(size=10, color="green"),
                name="NeRF centroid",
            )
        )
        fig_Oy.update_layout(title="Mesh in object y-up frame")
        add_transform_matrix_traces(fig=fig_Oy, transform_matrix=np.eye(4), length=0.1)

        table_mesh_Oy = get_hacky_table_mesh(table_y_Oy)
        table_vertices = table_mesh_Oy.vertices
        fig_Oy.add_trace(
            go.Mesh3d(
                x=table_vertices[:, 0],
                y=table_vertices[:, 1],
                z=table_vertices[:, 2],
                i=table_mesh_Oy.faces[:, 0],
                j=table_mesh_Oy.faces[:, 1],
                k=table_mesh_Oy.faces[:, 2],
                color="green",
                opacity=0.1,
                name="table",
            )
        )

        fig_Oy.show()

    print("\n" + "=" * 80)
    print("Step 5: Load grasp metric")
    print("=" * 80 + "\n")
    print(f"Loading classifier config from {args.classifier_config_path}")
    classifier_config = tyro.extras.from_yaml(
        ClassifierConfig, args.classifier_config_path.open()
    )

    USE_DEPTH_IMAGES = isinstance(
        classifier_config.nerfdata_config, DepthImageNerfDataConfig
    )
    if USE_DEPTH_IMAGES:
        classifier_model = load_depth_image_classifier(classifier=classifier_config)
        grasp_metric = DepthImageGraspMetric(
            nerf_model=nerf_model,
            classifier_model=classifier_model,
            fingertip_config=classifier_config.nerfdata_config.fingertip_config,
            camera_config=classifier_config.nerfdata_config.fingertip_camera_config,
            X_N_Oy=X_N_Oy,
        )
    else:
        classifier_model = load_classifier(classifier_config=classifier_config)
        grasp_metric = GraspMetric(
            nerf_field=nerf_field,
            classifier_model=classifier_model,
            fingertip_config=classifier_config.nerfdata_config.fingertip_config,
            X_N_Oy=X_N_Oy,
        )

    print("\n" + "=" * 80)
    print("Step 6: Optimize grasps")
    print("=" * 80 + "\n")
    optimized_grasp_config_dict = get_optimized_grasps(
        cfg=OptimizationConfig(
            use_rich=True,
            init_grasp_config_dict_path=args.init_grasp_config_dict_path,
            grasp_metric=GraspMetricConfig(
                nerf_checkpoint_path=nerf_config,
                classifier_config_path=args.classifier_config_path,
                X_N_Oy=X_N_Oy,
            ),  # This is not used because we are passing in a grasp_metric
            optimizer=SGDOptimizerConfig(
                num_grasps=32,
                num_steps=0,
                finger_lr=1e-4,
                grasp_dir_lr=1e-4,
                wrist_lr=1e-4,
            ),
            output_path=pathlib.Path(
                experiment_folder
                / "optimized_grasp_config_dicts"
                / f"{args.object_name}.npy"
            ),
        ),
        grasp_metric=grasp_metric,
    )

    print("\n" + "=" * 80)
    print("Step 7: Convert optimized grasps to joint angles")
    print("=" * 80 + "\n")
    X_Oy_H_array, joint_angles_array, target_joint_angles_array = (
        get_sorted_grasps_from_dict(
            optimized_grasp_config_dict=optimized_grasp_config_dict,
            error_if_no_loss=True,
            check=False,
            print_best=False,
        )
    )
    num_grasps = X_Oy_H_array.shape[0]
    assert X_Oy_H_array.shape == (num_grasps, 4, 4)
    assert joint_angles_array.shape == (num_grasps, 16)
    assert target_joint_angles_array.shape == (num_grasps, 16)

    print("\n" + "=" * 80)
    print("Step 8: Execute best grasps")
    print("=" * 80 + "\n")
    if args.is_real_world:
        mesh_W = trimesh.Trimesh(vertices=mesh_N.vertices, faces=mesh_N.faces)
        mesh_W.apply_transform(X_W_N)
        for i in range(num_grasps):
            print(f"Trying grasp {i} / {num_grasps}")

            X_Oy_H = X_Oy_H_array[i]
            joint_angles = joint_angles_array[i]
            target_joint_angles = target_joint_angles_array[i]

            X_W_H = X_W_N @ X_N_Oy @ X_Oy_H

            if not ALBERT_is_feasible(
                X_W_H=X_W_H, joint_angles=joint_angles, mesh_W=mesh_W
            ):
                print(f"Grasp {i} is infeasible")
                continue

            ALBERT_execute_grasp(
                X_W_H=X_W_H,
                joint_angles=joint_angles,
                target_joint_angles=target_joint_angles,
                mesh_W=mesh_W,
            )
    else:
        print(
            "Skipping execution because is_real_world is False, evaluate with DexGraspNet sim using"
        )
        print(
            f"CUDA_VISIBLE_DEVICES=0 python scripts/eval_grasp_config_dict.py --hand_model_type ALLEGRO_HAND --validation_type GRAVITY_AND_TABLE --gpu 0 --meshdata_root_path ../data/rotated_meshdata --input_grasp_config_dicts_path {str(experiment_folder.absolute() / 'optimized_grasp_config_dicts')} --output_evaled_grasp_config_dicts_path {str(experiment_folder.absolute() / 'evaled_optimized_grasp_config_dicts')} --object_code_and_scale_str {args.object_name} --max_grasps_per_batch 5000 --num_random_pose_noise_samples_per_grasp 5"
        )


def main() -> None:
    args = tyro.cli(Args)
    print("=" * 80)
    print(f"{pathlib.Path(__file__).name} args: {args}")
    print("=" * 80 + "\n")
    rough_hardware_deployment_code(args)


if __name__ == "__main__":
    main()
