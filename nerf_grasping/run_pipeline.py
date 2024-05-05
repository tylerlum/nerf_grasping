from tqdm import tqdm
from typing import Optional, Tuple, List
from nerfstudio.models.base_model import Model
from nerf_grasping.grasp_utils import load_nerf_pipeline
from nerf_grasping.fr3_algr_ik.ik import solve_ik
from nerf_grasping.optimizer import get_optimized_grasps
from nerf_grasping.optimizer_utils import (
    get_sorted_grasps_from_dict,
    GraspMetric,
    DepthImageGraspMetric,
    load_classifier,
    load_depth_image_classifier,
    is_in_limits,
    clamp_in_limits,
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

import nerf_grasping


@dataclass
class PipelineConfig:
    init_grasp_config_dict_path: pathlib.Path
    classifier_config_path: pathlib.Path
    object_code: str = "unnamed_object"
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
    num_grasps: int = 32
    num_steps: int = 0
    random_seed: Optional[int] = None
    n_random_rotations_per_grasp: int = 5
    object_scale: float = 0.9999
    nerf_config: Optional[pathlib.Path] = None

    def __post_init__(self) -> None:
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

        if self.nerf_config is not None:
            assert self.nerf_config.exists(), f"{self.nerf_config} does not exist"
            assert (
                self.nerf_config.suffix == ".yml"
            ), f"{self.nerf_config} does not have a .yml suffix"

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

    @property
    def object_code_and_scale_str(self) -> str:
        return f"{self.object_code}_{self.object_scale:.4f}".replace(".", "_")


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


def run_pipeline(
    nerf_model: Model,
    cfg: PipelineConfig,
) -> Tuple[
    List[Optional[np.ndarray]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    trimesh.Trimesh,
    np.ndarray,
    np.ndarray,
]:
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
    X_W_N, X_O_Oy = cfg.X_W_N, cfg.X_O_Oy
    lb_N, ub_N = cfg.lb_N, cfg.ub_N

    print(f"Creating a new experiment folder at {cfg.output_folder}")
    cfg.output_folder.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("Step 2: Get NERF")
    print("=" * 80 + "\n")
    nerf_field = nerf_model.field
    nerf_config = (
        cfg.nerf_config
        if cfg.nerf_config is not None
        else pathlib.Path("DUMMY_NERF_CONFIG/config.yml")
    )  # Dummy value to put in, not used because nerf_model is passed in

    print("\n" + "=" * 80)
    print("Step 3: Convert NeRF to mesh")
    print("=" * 80 + "\n")
    nerf_to_mesh_folder = cfg.output_folder / "nerf_to_mesh" / cfg.object_code / "coacd"
    nerf_to_mesh_folder.mkdir(parents=True, exist_ok=True)
    mesh_N = nerf_to_mesh(
        field=nerf_field,
        level=cfg.density_levelset_threshold,
        lb=lb_N,
        ub=ub_N,
        save_path=nerf_to_mesh_folder / "decomposed.obj",
    )

    # HACK: Save to /tmp/mesh_viz_object.obj as well
    mesh_N.export("/tmp/mesh_viz_object.obj")

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
        level=cfg.density_levelset_threshold,
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
    nerf_to_mesh_Oy_folder = (
        cfg.output_folder / "nerf_to_mesh_Oy" / cfg.object_code / "coacd"
    )
    nerf_to_mesh_Oy_folder.mkdir(parents=True, exist_ok=True)
    mesh_Oy.export(nerf_to_mesh_Oy_folder / "decomposed.obj")
    mesh_centroid_Oy = transform_point(X_Oy_N, centroid_N)
    nerf_centroid_Oy = transform_point(X_Oy_N, centroid_N)

    if cfg.visualize:
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
    print(f"Loading classifier config from {cfg.classifier_config_path}")
    classifier_config = tyro.extras.from_yaml(
        ClassifierConfig, cfg.classifier_config_path.open()
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
            init_grasp_config_dict_path=cfg.init_grasp_config_dict_path,
            grasp_metric=GraspMetricConfig(
                nerf_checkpoint_path=nerf_config,
                classifier_config_path=cfg.classifier_config_path,
                X_N_Oy=X_N_Oy,
            ),  # This is not used because we are passing in a grasp_metric
            optimizer=SGDOptimizerConfig(
                num_grasps=cfg.num_grasps,
                num_steps=cfg.num_steps,
                finger_lr=1e-4,
                grasp_dir_lr=1e-4,
                wrist_lr=1e-4,
            ),
            output_path=pathlib.Path(
                cfg.output_folder
                / "optimized_grasp_config_dicts"
                / f"{cfg.object_code_and_scale_str}.npy"
            ),
            random_seed=cfg.random_seed,
            n_random_rotations_per_grasp=cfg.n_random_rotations_per_grasp,
            wandb=None,
        ),
        grasp_metric=grasp_metric,
    )

    print("\n" + "=" * 80)
    print("Step 7: Convert optimized grasps to joint angles")
    print("=" * 80 + "\n")
    X_Oy_Hs, q_algrs, q_algr_posts, q_algr_pres = get_sorted_grasps_from_dict(
        optimized_grasp_config_dict=optimized_grasp_config_dict,
        error_if_no_loss=True,
        check=False,
        print_best=False,
    )
    q_algr_pres = clamp_in_limits(q_algr_pres)
    # q_algr_pres = q_algrs
    # DELTA = 0.3
    # q_algr_pres[:, 1] -= DELTA
    # q_algr_pres[:, 2] -= DELTA
    # q_algr_pres[:, 3] -= DELTA

    # q_algr_pres[:, 5] -= DELTA
    # q_algr_pres[:, 6] -= DELTA
    # q_algr_pres[:, 7] -= DELTA

    # q_algr_pres[:, 9] -= DELTA
    # q_algr_pres[:, 10] -= DELTA
    # q_algr_pres[:, 11] -= DELTA
                # q_algr_pre[1] -= 0.1
                # q_algr_pre[2] -= 0.1
                # q_algr_pre[3] -= 0.1

                # q_algr_pre[5] -= 0.1
                # q_algr_pre[6] -= 0.1
                # q_algr_pre[7] -= 0.1

                # q_algr_pre[9] -= 0.1
                # q_algr_pre[10] -= 0.1
                # q_algr_pre[11] -= 0.1

                # q_algr_pre[14] -= 0.1

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

    q_algr_pres_is_in_limits = is_in_limits(q_algr_pres)
    assert q_algr_pres_is_in_limits.shape == (num_grasps,)
    pass_idxs = set(np.where(q_algr_pres_is_in_limits)[0])
    print(
        f"Number of grasps in limits: {len(pass_idxs)} / {num_grasps} ({len(pass_idxs) / num_grasps * 100:.2f}%)"
    )
    print(f"pass_idxs: {pass_idxs}")

    q_stars = []
    for i in range(num_grasps):
        X_W_H = X_W_Hs[i]
        q_algr_pre = q_algr_pres[i]
        q_algr_post = q_algr_posts[i]

        try:
            q_star = solve_ik(X_W_H=X_W_H, q_algr_pre=q_algr_pre, visualize=False)
            print(f"Success for grasp {i}")
            q_stars.append(q_star)
        except RuntimeError as e:
            print(f"Failed to solve IK for grasp {i}")
            q_stars.append(None)
    assert len(q_stars) == num_grasps
    num_passed = sum([q_star is not None for q_star in q_stars])
    print(
        f"Number of grasps passed IK: {num_passed} / {num_grasps} ({num_passed / num_grasps * 100:.2f}%)"
    )

    print("\n" + "=" * 80)
    print("Step 9: Solve trajopt for each grasp")
    print("=" * 80 + "\n")
    from nerf_grasping.fr3_algr_trajopt.trajopt import (
        solve_trajopt,
        TrajOptParams,
        DEFAULT_Q_FR3,
        DEFAULT_Q_ALGR,
    )

    trajopt_cfg = TrajOptParams(
        num_control_points=21,
        min_self_coll_dist=0.005,
        influence_dist=0.01,
        nerf_frame_offset=cfg.nerf_frame_offset_x,
        s_start_self_col=0.5,
        lqr_pos_weight=1e-1,
        lqr_vel_weight=20.0,
        presolve_no_collision=True,
    )
    USE_DEFAULT_Q_0 = True
    if USE_DEFAULT_Q_0:
        print("Using default q_0")
        q_fr3_0 = DEFAULT_Q_FR3
        q_algr_0 = DEFAULT_Q_ALGR
    else:
        raise NotImplementedError

    passing_trajopt_idxs = []
    failed_trajopt_idxs = []
    for i, q_star in tqdm(enumerate(q_stars), total=num_grasps):
        if q_star is None:
            print("$" * 80)
            print(f"Trajectory optimization skipped for grasp {i}")
            print("$" * 80 + "\n")
            continue

        try:
            spline, dspline, T_traj, trajopt = solve_trajopt(
                q_fr3_0=q_fr3_0,
                q_algr_0=q_algr_0,
                q_fr3_f=q_star[:7],
                q_algr_f=q_star[7:],
                cfg=trajopt_cfg,
                # mesh_path=pathlib.Path("/tmp/mesh_viz_object.obj"),
                mesh_path=None,
                visualize=False,
                verbose=False,
            )
            print("^" * 80)
            print(f"Trajectory optimization succeeded for grasp {i}")
            print("^" * 80 + "\n")
            passing_trajopt_idxs.append(i)
        except RuntimeError as e:
            print("~" * 80)
            print(f"Trajectory optimization failed for grasp {i}")
            print("~" * 80 + "\n")
            failed_trajopt_idxs.append(i)
# spline, dspline, T_traj, trajopt = solve_trajopt( q_fr3_0=q_fr3_0, q_algr_0=q_algr_0, q_fr3_f=q_star[:7], q_algr_f=q_star[7:], cfg=trajopt_cfg, mesh_path=pathlib.Path("/tmp/mesh_viz_object.obj"))

    print(f"passing_trajopt_idxs: {passing_trajopt_idxs}")
    print(f"failed_trajopt_idxs: {failed_trajopt_idxs}")
    breakpoint()
    return (
        q_stars,
        X_W_Hs,
        q_algr_pres,
        q_algr_posts,
        mesh_W,
        X_N_Oy,
        q_algr_pres_is_in_limits,
    )


@dataclass
class CommandlineArgs(PipelineConfig):
    nerfdata_path: Optional[pathlib.Path] = None
    nerfcheckpoint_path: Optional[pathlib.Path] = None
    max_num_iterations: int = 2000

    def __post_init__(self) -> None:
        if self.nerfdata_path is not None and self.nerfcheckpoint_path is None:
            assert self.nerfdata_path.exists(), f"{self.nerfdata_path} does not exist"
            assert (
                self.nerfdata_path / "transforms.json"
            ).exists(), f"{self.nerfdata_path / 'transforms.json'} does not exist"
            assert (
                self.nerfdata_path / "images"
            ).exists(), f"{self.nerfdata_path / 'images'} does not exist"
        elif self.nerfdata_path is None and self.nerfcheckpoint_path is not None:
            assert (
                self.nerfcheckpoint_path.exists()
            ), f"{self.nerfcheckpoint_path} does not exist"
            assert (
                self.nerfcheckpoint_path.suffix == ".yml"
            ), f"{self.nerfcheckpoint_path} does not have a .yml suffix"
        else:
            raise ValueError(
                "Exactly one of nerfdata_path or nerfcheckpoint_path must be specified"
            )


def main() -> None:
    args = tyro.cli(CommandlineArgs)
    print("=" * 80)
    print(f"args: {args}")
    print("=" * 80 + "\n")

    if args.nerfdata_path is not None:
        nerf_checkpoints_folder = args.output_folder / "nerfcheckpoints"
        nerf_trainer = train_nerfs_return_trainer.train_nerf(
            args=train_nerfs_return_trainer.Args(
                nerfdata_folder=args.nerfdata_path,
                nerfcheckpoints_folder=nerf_checkpoints_folder,
                max_num_iterations=args.max_num_iterations,
            )
        )
        nerf_model = nerf_trainer.pipeline.model
        nerf_config = nerf_trainer.config.get_base_dir() / "config.yml"
    elif args.nerfcheckpoint_path is not None:
        nerf_pipeline = load_nerf_pipeline(args.nerfcheckpoint_path)
        nerf_model = nerf_pipeline.model
        nerf_config = args.nerfcheckpoint_path
    else:
        raise ValueError(
            "Exactly one of nerfdata_path or nerfcheckpoint_path must be specified"
        )

    args.nerf_config = nerf_config
    run_pipeline(nerf_model=nerf_model, cfg=args)


if __name__ == "__main__":
    main()
