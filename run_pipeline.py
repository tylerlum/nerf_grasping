# %%
from nerf_grasping.fr3_algr_ik.ik import solve_ik
from nerf_grasping.optimizer import get_optimized_grasps
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
from datetime import datetime


@dataclass
class Args:
    nerfdata_path: pathlib.Path
    init_grasp_config_dict_path: pathlib.Path
    classifier_config_path: pathlib.Path
    object_name: str = "unnamed_object"
    output_folder: pathlib.Path = pathlib.Path("experiments") / datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    density_levelset_threshold: float = 15.0
    obj_is_z_up: bool = True
    lb_x: float = -0.2
    lb_y: float = -0.2
    lb_z: float = 0.0
    ub_x: float = 0.2
    ub_y: float = 0.2
    ub_z: float = 0.3
    nerf_frame_offset_x: float = 0.65
    visualize: bool = False

    def __post_init__(self) -> None:
        assert self.nerfdata_path.exists(), f"{self.nerfdata_path} does not exist"
        assert self.nerfdata_path.is_dir(), f"{self.nerfdata_path} is not a directory"
        assert (
            self.nerfdata_path / "transforms.json"
        ).exists(), f"{self.nerfdata_path / 'transforms.json'} does not exist"
        assert (
            self.nerfdata_path / "images"
        ).exists(), f"{self.nerfdata_path / 'images'} does not exist"

        assert (
            self.init_grasp_config_dict_path.exists()
        ), f"{self.init_grasp_config_dict_path} does not exist"
        assert (
            self.init_grasp_config_dict_path.suffix == ".npy"
        ), f"{self.init_grasp_config_dict_path} does not have a .npy suffix"

        assert (
            self.classifier_config_path.exists()
        ), f"{self.classifier_config_path} does not exist"
        assert self.classifier_config_path.suffix in [
            ".yml",
            ".yaml",
        ], f"{self.classifier_config_path} does not have a .yml or .yaml suffix"

    @property
    def lb_N(self) -> np.ndarray:
        return np.array([self.lb_x, self.lb_y, self.lb_z])

    @property
    def ub_N(self) -> np.ndarray:
        return np.array([self.ub_x, self.ub_y, self.ub_z])

    @property
    def X_W_N(self) -> np.ndarray:
        return trimesh.transformations.translation_matrix(
            [self.nerf_frame_offset_x, 0, 0]
        )

    @property
    def X_O_Oy(self) -> np.ndarray:
        return (
            trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
            if self.obj_is_z_up
            else np.eye(4)
        )


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


def run_pipeline(args: Args) -> None:
    print("=" * 80)
    print("Step 1: Figuring out frames")
    print("=" * 80 + "\n")
    print("Frames are W = world, N = nerf, O = object, Oy = object y-up, H = hand")
    print(
        "W is centered at the robot base. N is centered where origin of NeRF data collection is. O is centered at the object centroid. Oy is centered at the object centroid. H is centered at the base of the middle finger"
    )
    print(
        "W, N, O are z-up frames. Oy is y-up. H has z-up along finger and x-up along palm normal"
    )
    print("X_A_B represents 4x4 transformation matrix of frame B wrt A")
    X_W_N, X_O_Oy = args.X_W_N, args.X_O_Oy
    lb_N, ub_N = args.lb_N, args.ub_N

    print(f"Creating a new experiment folder at {args.output_folder}")
    args.output_folder.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("Step 2: Train NERF")
    print("=" * 80 + "\n")
    nerf_trainer = train_nerfs_return_trainer.train_nerf(
        args=train_nerfs_return_trainer.Args(
            nerfdata_folder=args.nerfdata_path,
            nerfcheckpoints_folder=args.output_folder / "nerfcheckpoints",
            max_num_iterations=2000,
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
    nerf_to_mesh_folder = args.output_folder / "nerf_to_mesh"
    nerf_to_mesh_folder.mkdir(parents=True, exist_ok=True)
    mesh_N = nerf_to_mesh(
        field=nerf_field,
        level=args.density_levelset_threshold,
        lb=lb_N,
        ub=ub_N,
        save_path=nerf_to_mesh_folder / f"{args.object_name}.obj",
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
    nerf_to_mesh_Oy_folder = args.output_folder / "nerf_to_mesh_Oy"
    nerf_to_mesh_Oy_folder.mkdir(parents=True, exist_ok=True)
    mesh_Oy.export(
        nerf_to_mesh_Oy_folder / f"{args.object_name}.obj",
    )
    mesh_centroid_Oy = transform_point(X_Oy_N, centroid_N)
    nerf_centroid_Oy = transform_point(X_Oy_N, centroid_N)

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
                args.output_folder
                / "optimized_grasp_config_dicts"
                / f"{args.object_name}.npy"
            ),
        ),
        grasp_metric=grasp_metric,
    )

    print("\n" + "=" * 80)
    print("Step 7: Convert optimized grasps to joint angles")
    print("=" * 80 + "\n")
    X_Oy_Hs, q_algr_pres, q_algr_posts = get_sorted_grasps_from_dict(
        optimized_grasp_config_dict=optimized_grasp_config_dict,
        error_if_no_loss=True,
        check=False,
        print_best=False,
    )
    num_grasps = X_Oy_Hs.shape[0]
    assert X_Oy_Hs.shape == (num_grasps, 4, 4)
    assert q_algr_pres.shape == (num_grasps, 16)
    assert q_algr_posts.shape == (num_grasps, 16)

    print("\n" + "=" * 80)
    print("Step 8: Solve IK for each grasp")
    print("=" * 80 + "\n")
    mesh_W = trimesh.Trimesh(vertices=mesh_N.vertices, faces=mesh_N.faces)
    mesh_W.apply_transform(X_W_N)

    X_W_Hs = np.stack([X_W_N @ X_N_Oy @ X_Oy_Hs[i] for i in range(num_grasps)], axis=0)
    assert X_W_Hs.shape == (num_grasps, 4, 4)

    q_stars = []
    for i in range(num_grasps):
        X_W_H = X_W_Hs[i]
        q_algr_pre = q_algr_pres[i]
        q_algr_post = q_algr_posts[i]

        try:
            q_star = solve_ik(X_W_H=X_W_H, q_algr_pre=q_algr_pre, visualize=False)
            print(f"Success for grasp {i} / {num_grasps}")
            q_stars.append(q_star)
        except RuntimeError as e:
            print(f"Failed to solve IK for grasp {i} / {num_grasps}: {e}")
            q_stars.append(None)
    assert len(q_stars) == num_grasps
    num_passed = sum([q_star is not None for q_star in q_stars])
    print(
        f"Number of grasps passed IK: {num_passed} / {num_grasps} ({num_passed / num_grasps * 100:.2f}%)"
    )

    return q_stars, X_W_Hs, q_algr_pres, q_algr_posts, mesh_W


def main() -> None:
    args = tyro.cli(Args)
    print("=" * 80)
    print(f"args: {args}")
    print("=" * 80 + "\n")
    run_pipeline(args)


if __name__ == "__main__":
    main()
