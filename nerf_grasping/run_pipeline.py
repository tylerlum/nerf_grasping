from tqdm import tqdm
import math
import time
from typing import Optional, Tuple, List
from nerfstudio.models.base_model import Model
from nerf_grasping.grasp_utils import load_nerf_pipeline
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


def compute_grasps(
    nerf_model: Model,
    cfg: PipelineConfig,
) -> Tuple[
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

    # Save to /tmp/mesh_viz_object.obj as well
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

    mesh_W = trimesh.Trimesh(vertices=mesh_N.vertices, faces=mesh_N.faces)
    mesh_W.apply_transform(X_W_N)

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
    losses = optimized_grasp_config_dict["loss"]
    X_Oy_Hs, q_algr_pres, q_algr_posts, q_algr_extra_open = get_sorted_grasps_from_dict(
        optimized_grasp_config_dict=optimized_grasp_config_dict,
        error_if_no_loss=True,
        check=False,
        print_best=False,
    )

    MODE = "EXTRA_OPEN"  # TODO: Compare these
    print("!" * 80)
    print(f"MODE: {MODE}")
    print("!" * 80 + "\n")
    if MODE == "DEFAULT":
        q_algr_pres = q_algr_pres
    elif MODE == "EXTRA_OPEN":
        q_algr_pres = q_algr_extra_open
    elif MODE == "JOINTS_OPEN":
        DELTA = 0.1
        q_algr_pres[:, 1] -= DELTA
        q_algr_pres[:, 2] -= DELTA
        q_algr_pres[:, 3] -= DELTA

        q_algr_pres[:, 5] -= DELTA
        q_algr_pres[:, 6] -= DELTA
        q_algr_pres[:, 7] -= DELTA

        q_algr_pres[:, 9] -= DELTA
        q_algr_pres[:, 10] -= DELTA
        q_algr_pres[:, 11] -= DELTA
    else:
        raise ValueError(f"Invalid MODE: {MODE}")
    q_algr_pres = clamp_in_limits(q_algr_pres)

    num_grasps = X_Oy_Hs.shape[0]
    assert X_Oy_Hs.shape == (num_grasps, 4, 4)
    assert q_algr_pres.shape == (num_grasps, 16)
    assert q_algr_posts.shape == (num_grasps, 16)

    q_algr_pres_is_in_limits = is_in_limits(q_algr_pres)
    assert q_algr_pres_is_in_limits.shape == (num_grasps,)
    pass_idxs = set(np.where(q_algr_pres_is_in_limits)[0])
    print(
        f"Number of grasps in limits: {len(pass_idxs)} / {num_grasps} ({len(pass_idxs) / num_grasps * 100:.2f}%)"
    )
    print(f"pass_idxs: {pass_idxs}")

    X_W_Hs = np.stack([X_W_N @ X_N_Oy @ X_Oy_Hs[i] for i in range(num_grasps)], axis=0)
    assert X_W_Hs.shape == (num_grasps, 4, 4)

    return (
        X_W_Hs,
        q_algr_pres,
        q_algr_posts,
        mesh_W,
        X_N_Oy,
        losses,
    )


def run_drake(
    cfg: PipelineConfig,
    X_W_Hs: np.ndarray,
    q_algr_pres: np.ndarray,
    q_algr_posts: np.ndarray,
) -> None:
    from nerf_grasping.fr3_algr_ik.ik import solve_ik

    num_grasps = X_W_Hs.shape[0]
    q_stars = []
    for i in tqdm(range(num_grasps)):
        X_W_H = X_W_Hs[i]
        q_algr_pre = q_algr_pres[i]

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
    not_attempted_trajopt_idxs = []
    for i, q_star in tqdm(enumerate(q_stars), total=num_grasps):
        if q_star is None:
            not_attempted_trajopt_idxs.append(i)
            continue

        try:
            spline, dspline, T_traj, trajopt = solve_trajopt(
                q_fr3_0=q_fr3_0,
                q_algr_0=q_algr_0,
                q_fr3_f=q_star[:7],
                q_algr_f=q_star[7:],
                cfg=trajopt_cfg,
                mesh_path=pathlib.Path("/tmp/mesh_viz_object.obj"),
                visualize=False,
                verbose=False,
                ignore_obj_collision=False,
            )
            passing_trajopt_idxs.append(i)

        except RuntimeError as e:
            failed_trajopt_idxs.append(i)

    print(f"passing_trajopt_idxs: {passing_trajopt_idxs}")
    print(f"failed_trajopt_idxs: {failed_trajopt_idxs}")
    print(f"not_attempted_trajopt_idxs: {not_attempted_trajopt_idxs}")

    TRAJ_IDX = passing_trajopt_idxs[0] if len(passing_trajopt_idxs) > 0 else 0
    print(f"Visualizing trajectory {TRAJ_IDX}")
    try:
        spline, dspline, T_traj, trajopt = solve_trajopt(
            q_fr3_0=q_fr3_0,
            q_algr_0=q_algr_0,
            q_fr3_f=q_stars[TRAJ_IDX][:7],
            q_algr_f=q_stars[TRAJ_IDX][7:],
            cfg=trajopt_cfg,
            mesh_path=pathlib.Path("/tmp/mesh_viz_object.obj"),
            visualize=True,
            verbose=False,
            ignore_obj_collision=False,
        )
    except RuntimeError as e:
        print(f"Failed to visualize trajectory {TRAJ_IDX}")

    while True:
        input_options = "\n".join(
            [
                "=====================",
                "OPTIONS",
                "b for breakpoint",
                "r to run trajopt with object collision",
                "o to run trajopt without object collision",
                "n to go to next traj",
                "p to go to prev traj",
                "q to quit",
                "=====================",
            ]
        )

        x = input("\n" + input_options + "\n\n")
        if x == "b":
            print("Breakpoint")
            breakpoint()
        elif x == "r":
            print(f"Visualizing trajectory {TRAJ_IDX} with object collision")
            if q_stars[TRAJ_IDX] is not None:
                try:
                    spline, dspline, T_traj, trajopt = solve_trajopt(
                        q_fr3_0=q_fr3_0,
                        q_algr_0=q_algr_0,
                        q_fr3_f=q_stars[TRAJ_IDX][:7],
                        q_algr_f=q_stars[TRAJ_IDX][7:],
                        cfg=trajopt_cfg,
                        mesh_path=pathlib.Path("/tmp/mesh_viz_object.obj"),
                        visualize=True,
                        verbose=False,
                        ignore_obj_collision=False,
                    )
                except RuntimeError as e:
                    print(f"Failed to visualize trajectory {TRAJ_IDX}")
            else:
                print(f"Trajectory {TRAJ_IDX} is None, skipping")
        elif x == "o":
            print(f"Visualizing trajectory {TRAJ_IDX} without object collision")
            if q_stars[TRAJ_IDX] is not None:
                try:
                    spline, dspline, T_traj, trajopt = solve_trajopt(
                        q_fr3_0=q_fr3_0,
                        q_algr_0=q_algr_0,
                        q_fr3_f=q_stars[TRAJ_IDX][:7],
                        q_algr_f=q_stars[TRAJ_IDX][7:],
                        cfg=trajopt_cfg,
                        mesh_path=pathlib.Path("/tmp/mesh_viz_object.obj"),
                        visualize=True,
                        verbose=False,
                        ignore_obj_collision=True,
                    )
                except RuntimeError as e:
                    print(f"Failed to visualize trajectory {TRAJ_IDX}")
            else:
                print(f"Trajectory {TRAJ_IDX} is None, skipping")
        elif x == "n":
            TRAJ_IDX += 1
            if TRAJ_IDX >= num_grasps:
                TRAJ_IDX = 0
            print(f"Using trajectory {TRAJ_IDX}")
        elif x == "p":
            TRAJ_IDX -= 1
            if TRAJ_IDX < 0:
                TRAJ_IDX = num_grasps - 1
            print(f"Using trajectory {TRAJ_IDX}")
        elif x == "q":
            print("Quitting")
            break
        else:
            print(f"Invalid input: {x}")

    print("Breakpoint to visualize")
    breakpoint()


def run_curobo(
    cfg: PipelineConfig,
    X_W_Hs: np.ndarray,
    q_algr_pres: np.ndarray,
    q_algr_posts: np.ndarray,
    losses: Optional[np.ndarray] = None,
    q_fr3: Optional[np.ndarray] = None,
    q_algr: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[int], tuple]:
    # Timing
    APPROACH_TIME = 5
    STAY_OPEN_TIME = 0.5
    CLOSE_TIME = 0.5
    STAY_CLOSED_TIME = 0.5
    LIFT_TIME = 3.0

    from nerf_grasping.curobo_fr3_algr_zed2i.ik_fr3_algr_zed2i import (
        solve_iks,
    )
    from nerf_grasping.curobo_fr3_algr_zed2i.trajopt_batch import (
        solve_trajopt_batch,
        get_trajectories_from_result,
        rescale_if_out_of_velocity_limits,
        compute_rescale_factors_to_stay_within_limits,
    )

    n_grasps = X_W_Hs.shape[0]
    assert X_W_Hs.shape == (n_grasps, 4, 4)
    assert q_algr_pres.shape == (n_grasps, 16)
    if q_fr3 is None:
        print("Using default q_fr3")
        from nerf_grasping.curobo_fr3_algr_zed2i.trajopt_fr3_algr_zed2i import (
            DEFAULT_Q_FR3,
        )

        q_fr3 = DEFAULT_Q_FR3
    if q_algr is None:
        print("Using default q_algr")
        from nerf_grasping.curobo_fr3_algr_zed2i.trajopt_fr3_algr_zed2i import (
            DEFAULT_Q_ALGR,
        )

        q_algr = DEFAULT_Q_ALGR
    assert q_fr3.shape == (7,)
    assert q_algr.shape == (16,)

    print("\n" + "=" * 80)
    print("Step 9: Solve motion gen for each grasp")
    print("=" * 80 + "\n")

    # Enable trajopt often makes it fail, haven't been able to figure out why
    motion_gen_result, ik_result, ik_result2 = solve_trajopt_batch(
        X_W_Hs=X_W_Hs,
        q_algrs=q_algr_pres,
        q_fr3_starts=q_fr3[None, ...].repeat(n_grasps, axis=0),
        q_algr_starts=q_algr[None, ...].repeat(n_grasps, axis=0),
        collision_check_object=True,
        obj_filepath=pathlib.Path("/tmp/mesh_viz_object.obj"),
        obj_xyz=(cfg.nerf_frame_offset_x, 0.0, 0.0),
        obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        collision_check_table=True,
        use_cuda_graph=True,
        enable_graph=True,
        enable_opt=False,
        timeout=5.0,
        collision_sphere_buffer=0.01,
    )

    motion_gen_success_idxs = (
        motion_gen_result.success.flatten().nonzero().flatten().tolist()
    )
    ik_success_idxs = ik_result.success.flatten().nonzero().flatten().tolist()
    ik_success_idxs2 = ik_result2.success.flatten().nonzero().flatten().tolist()

    qs, qds, dts = get_trajectories_from_result(
        result=motion_gen_result,
        desired_trajectory_time=APPROACH_TIME,
    )
    qds, dts = rescale_if_out_of_velocity_limits(qds=qds, dts=dts, verbose=True)
    nonzero_q_idxs = [i for i, q in enumerate(qs) if np.absolute(q).sum() > 1e-2]
    overall_success_idxs = sorted(
        list(
            set(motion_gen_success_idxs)
            .intersection(set(ik_success_idxs).intersection(set(ik_success_idxs2)))
            .intersection(set(nonzero_q_idxs))
        )
    )  # All must be successful or else it may be successful for the wrong trajectory

    print("\n" + "=" * 80)
    print(
        "Motion generation without trajectory optimization complete, printing results"
    )
    print("=" * 80 + "\n")
    print(
        f"motion_gen_success_idxs: {motion_gen_success_idxs} ({len(motion_gen_success_idxs)} / {n_grasps} = {len(motion_gen_success_idxs) / n_grasps * 100:.2f}%)"
    )
    print(
        f"ik_success_idxs: {ik_success_idxs} ({len(ik_success_idxs)} / {n_grasps} = {len(ik_success_idxs) / n_grasps * 100:.2f}%)"
    )
    print(
        f"ik_success_idxs2: {ik_success_idxs2} ({len(ik_success_idxs2)} / {n_grasps} = {len(ik_success_idxs2) / n_grasps * 100:.2f}%)"
    )
    print(
        f"nonzero_q_idxs: {nonzero_q_idxs} ({len(nonzero_q_idxs)} / {n_grasps} = {len(nonzero_q_idxs) / n_grasps * 100:.2f}%)"
    )
    print(
        f"overall_success_idxs: {overall_success_idxs} ({len(overall_success_idxs)} / {n_grasps} = {len(overall_success_idxs) / n_grasps * 100:.2f}%)"
    )

    print("\n" + "=" * 80)
    print("Step 10: Add closing motion")
    print("=" * 80 + "\n")
    closing_qs, closing_qds = [], []
    for i, (q, qd, dt) in enumerate(zip(qs, qds, dts)):
        # Keep arm joints same, change hand joints
        open_q = q[-1]
        close_q = np.concatenate([open_q[:7], q_algr_posts[i]])

        # Stay open
        N_STAY_OPEN_STEPS = int(STAY_OPEN_TIME / dt)
        interpolated_qs0 = interpolate(start=open_q, end=open_q, N=N_STAY_OPEN_STEPS)
        assert interpolated_qs0.shape == (N_STAY_OPEN_STEPS, 23)

        # Close
        N_CLOSE_STEPS = int(CLOSE_TIME / dt)
        interpolated_qs1 = interpolate(start=open_q, end=close_q, N=N_CLOSE_STEPS)
        assert interpolated_qs1.shape == (N_CLOSE_STEPS, 23)

        # Stay closed
        N_STAY_CLOSED_STEPS = int(STAY_CLOSED_TIME / dt)
        interpolated_qs2 = interpolate(
            start=close_q, end=close_q, N=N_STAY_CLOSED_STEPS
        )
        assert interpolated_qs2.shape == (N_STAY_CLOSED_STEPS, 23)

        closing_q = np.concatenate(
            [interpolated_qs0, interpolated_qs1, interpolated_qs2], axis=0
        )
        assert closing_q.shape == (
            N_STAY_OPEN_STEPS + N_CLOSE_STEPS + N_STAY_CLOSED_STEPS,
            23,
        )

        closing_qd = np.diff(closing_q, axis=0) / dt
        closing_qd = np.concatenate([closing_qd, closing_qd[-1:]], axis=0)

        closing_qs.append(closing_q)
        closing_qds.append(closing_qd)

    print("\n" + "=" * 80)
    print("Step 11: Add lifing motion")
    print("=" * 80 + "\n")
    # Using same qs found from motion gen to ensure they are not starting in collision
    # Not using closing_qs because they potentially could have issues?
    q_start_lifts = np.array([q[-1] for q in qs])
    assert q_start_lifts.shape == (n_grasps, 23)

    X_W_H_lifts = X_W_Hs.copy()
    LIFT_AMOUNT = 0.2  # Lift up 20 cm
    X_W_H_lifts[:, 2, 3] += LIFT_AMOUNT

    # HACK: If motion_gen above fails, then it leaves q as all 0s, which causes next step to fail
    #       So we populate those with another valid one
    assert len(overall_success_idxs) > 0, "overall_success_idxs is empty"
    valid_idx = overall_success_idxs[0]
    for i in range(n_grasps):
        if i in overall_success_idxs:
            continue
        q_start_lifts[i] = q_start_lifts[valid_idx]
        X_W_H_lifts[i] = X_W_H_lifts[valid_idx]

    # Solve IK in N_WAYPOINTS between X_W_Hs and X_W_H_lifts
    # Then interpolate between those waypoints
    N_WAYPOINTS = 5

    X_W_H_interps_list = []
    for i in range(n_grasps):
        X_W_H = X_W_Hs[i]
        X_W_H_lift = X_W_H_lifts[i]

        # We don't want to solve IK for the current X_W_H because that is what we already have from the planned trajectory
        # Doing it again would break things by potentially causing a sudden change in q
        X_W_H_interps = X_W_H[None, ...].repeat(N_WAYPOINTS, axis=0)
        X_W_H_interps[:, 2, 3] = np.linspace(
            X_W_H[2, 3], X_W_H_lift[2, 3], N_WAYPOINTS + 1
        )[
            1:
        ]  # Skip the first one
        assert X_W_H_interps.shape == (N_WAYPOINTS, 4, 4)

        X_W_H_interps_list.append(X_W_H_interps)
    X_W_H_interps = np.stack(X_W_H_interps_list, axis=0)
    assert X_W_H_interps.shape == (n_grasps, N_WAYPOINTS, 4, 4)
    X_W_H_interps = X_W_H_interps.reshape(-1, 4, 4)

    lift_ik_start_time = time.time()
    (
        lift_waypoint_qs,
        lift_waypoint_success,
        DEBUG_lift_ik_result,
        DEBUG_lift_ik_solver,
    ) = solve_iks(
        X_W_Hs=X_W_H_interps,
        collision_check_object=False,
        obj_filepath=pathlib.Path("/tmp/mesh_viz_object.obj"),
        obj_xyz=(cfg.nerf_frame_offset_x, 0.0, 0.0),
        obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        collision_check_table=True,  # Not sure if needed, but safer to avoid table collisions
        use_cuda_graph=True,
    )
    lift_ik_end_time = time.time()
    print(f"Time to solve_iks: {lift_ik_end_time - lift_ik_start_time:.2f}s")
    assert lift_waypoint_qs.shape == (
        n_grasps * N_WAYPOINTS,
        23,
    ), f"lift_waypoint_qs.shape: {lift_waypoint_qs.shape}"
    assert lift_waypoint_success.shape == (
        n_grasps * N_WAYPOINTS,
    ), f"lift_waypoint_success.shape: {lift_waypoint_success.shape}"
    lift_waypoint_qs = lift_waypoint_qs.reshape(n_grasps, N_WAYPOINTS, 23)
    lift_waypoint_success = lift_waypoint_success.reshape(n_grasps, N_WAYPOINTS)
    lift_waypoint_qs = lift_waypoint_qs.reshape(n_grasps, N_WAYPOINTS, 23)

    if not lift_waypoint_success.all():
        print("WARNING: Not all lift IK waypoints were successful")
        print(f"lift_waypoint_success: {lift_waypoint_success}")

    lift_success = lift_waypoint_success.all(axis=1)
    assert lift_success.shape == (n_grasps,)
    lift_success_idxs = [i for i in range(n_grasps) if lift_success[i]]
    print(
        f"lift_success_idxs: {lift_success_idxs} ({len(lift_success_idxs)} / {n_grasps} = {len(lift_success_idxs) / n_grasps * 100:.2f}%)"
    )

    # Interpolate between waypoints
    raw_lift_qs = []
    raw_lift_qds = []
    raw_lift_dts = []
    TIME_BETWEEN_WAYPOINTS = LIFT_TIME / N_WAYPOINTS
    for i in range(n_grasps):
        waypoints = lift_waypoint_qs[i]
        assert waypoints.shape == (N_WAYPOINTS, 23)
        dt = dts[i]
        closing_q = closing_qs[i]

        N_INTERPOLATED = int(TIME_BETWEEN_WAYPOINTS / dt)

        all_interpolated_qs = []
        all_interpolated_qds = []
        # There are N_WAYPOINTS new waypoints, but including the first one N_WAYPOINTS + 1
        # Thus there are N_WAYPOINTS segments
        for curr_waypoint_idx in range(N_WAYPOINTS):
            curr_waypoint = waypoints[curr_waypoint_idx]
            if curr_waypoint_idx == 0:
                prev_waypoint = closing_q[-1]
            else:
                prev_waypoint = waypoints[curr_waypoint_idx - 1]
            assert (
                curr_waypoint.shape == prev_waypoint.shape == (23,)
            ), f"curr_waypoint.shape: {curr_waypoint.shape}, prev_waypoint.shape: {prev_waypoint.shape}"

            interpolated_qs = interpolate(
                start=prev_waypoint, end=curr_waypoint, N=N_INTERPOLATED
            )
            assert interpolated_qs.shape == (N_INTERPOLATED, 23)

            interpolated_qds = np.diff(interpolated_qs, axis=0) / dt
            interpolated_qds = np.concatenate(
                [interpolated_qds, interpolated_qds[-1:]], axis=0
            )

            # Handle joint velocity limits
            rescale_factor = compute_rescale_factors_to_stay_within_limits(
                qds=[interpolated_qds], dts=[dt]
            )[0]
            assert rescale_factor >= 1.0
            if rescale_factor > 1.0:
                print(
                    f"Rescaling interpolated_qs by {rescale_factor} for grasp {i} segment {curr_waypoint_idx}"
                )
                NEW_N_INTERPOLATED = math.ceil(N_INTERPOLATED * rescale_factor)
                interpolated_qs = interpolate(
                    start=prev_waypoint, end=curr_waypoint, N=NEW_N_INTERPOLATED
                )
                assert interpolated_qs.shape == (NEW_N_INTERPOLATED, 23)

                interpolated_qds = np.diff(interpolated_qs, axis=0) / dt
                interpolated_qds = np.concatenate(
                    [interpolated_qds, interpolated_qds[-1:]], axis=0
                )

            all_interpolated_qs.append(interpolated_qs)
            all_interpolated_qds.append(interpolated_qds)

        all_interpolated_qs = np.concatenate(all_interpolated_qs, axis=0)
        all_interpolated_qds = np.concatenate(all_interpolated_qds, axis=0)

        raw_lift_qs.append(all_interpolated_qs)
        raw_lift_qds.append(all_interpolated_qds)
        raw_lift_dts.append(dt)

    final_success_idxs = sorted(
        list(set(overall_success_idxs).intersection(set(lift_success_idxs)))
    )
    print("\n" + "~" * 80)
    print(
        f"final_success_idxs: {final_success_idxs} ({len(final_success_idxs)} / {n_grasps} = {len(final_success_idxs) / n_grasps * 100:.2f}%)"
    )
    if losses is not None:
        assert losses.shape == (n_grasps,)
        print(f"losses = {losses}")
        print(f"losses of successful grasps: {[losses[i] for i in final_success_idxs]}")
    print("~" * 80 + "\n")

    # Adjust the lift qs to have the same hand position as the closing qs
    # We only want the arm position of the lift qs
    adjusted_lift_qs, adjusted_lift_qds = [], []
    for i, (
        closing_q,
        closing_qd,
        dt,
        raw_lift_q,
        raw_lift_qd,
        raw_lift_dt,
    ) in enumerate(
        zip(closing_qs, closing_qds, dts, raw_lift_qs, raw_lift_qds, raw_lift_dts)
    ):
        # TODO: Figure out how to handle if lift_qs has different dt, only a problem if set enable_opt=True
        assert dt == raw_lift_dt, f"dt: {dt}, lift_dt: {raw_lift_dt}"

        # Only want the arm position of the lift closing_q (keep same hand position as before)
        adjusted_lift_q = raw_lift_q.copy()
        last_closing_q = closing_q[-1]
        adjusted_lift_q[:, 7:] = last_closing_q[None, 7:]

        adjusted_lift_qd = raw_lift_qd.copy()
        adjusted_lift_qd[:, 7:] = 0.0

        adjusted_lift_qs.append(adjusted_lift_q)
        adjusted_lift_qds.append(adjusted_lift_qd)

    print("\n" + "=" * 80)
    print("Step 12: Aggregate qs and qds")
    print("=" * 80 + "\n")
    q_trajs, qd_trajs = [], []
    for q, qd, closing_q, closing_qd, lift_q, lift_qd in zip(
        qs, qds, closing_qs, closing_qds, adjusted_lift_qs, adjusted_lift_qds
    ):
        q_traj = np.concatenate([q, closing_q, lift_q], axis=0)
        qd_traj = np.concatenate([qd, closing_qd, lift_qd], axis=0)
        q_trajs.append(q_traj)
        qd_trajs.append(qd_traj)

    print("\n" + "=" * 80)
    print("Step 13: Compute T_trajs")
    print("=" * 80 + "\n")
    T_trajs = []
    for q, dt in zip(q_trajs, dts):
        n_timesteps = q.shape[0]
        T_trajs.append(n_timesteps * dt)

    DEBUG_TUPLE = (
        motion_gen_result,
        ik_result,
        ik_result2,
        DEBUG_lift_ik_result,
    )
    return q_trajs, qd_trajs, T_trajs, final_success_idxs, DEBUG_TUPLE


def run_pipeline(
    nerf_model: Model,
    cfg: PipelineConfig,
    q_fr3: Optional[np.ndarray] = None,
    q_algr: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[int], tuple]:

    start_time = time.time()
    (
        X_W_Hs,
        q_algr_pres,
        q_algr_posts,
        mesh_W,
        X_N_Oy,
        losses,
    ) = compute_grasps(nerf_model=nerf_model, cfg=cfg)
    compute_grasps_time = time.time()
    print("@" * 80)
    print(f"Time to compute_grasps: {compute_grasps_time - start_time:.2f}s")
    print("@" * 80 + "\n")

    qs, qds, T_trajs, success_idxs, DEBUG_TUPLE = run_curobo(
        cfg=cfg,
        X_W_Hs=X_W_Hs,
        q_algr_pres=q_algr_pres,
        q_algr_posts=q_algr_posts,
        losses=losses,
        q_fr3=q_fr3,
        q_algr=q_algr,
    )
    curobo_time = time.time()
    print("@" * 80)
    print(f"Time to run_curobo: {curobo_time - compute_grasps_time:.2f}s")
    print("@" * 80 + "\n")

    print("\n" + "=" * 80)
    print(f"Total time: {curobo_time - start_time:.2f}s")
    print("=" * 80 + "\n")
    return qs, qds, T_trajs, success_idxs, DEBUG_TUPLE


def visualize(
    cfg: PipelineConfig,
    qs: List[np.ndarray],
    qds: List[np.ndarray],
    T_trajs: List[float],
    success_idxs: List[int],
    DEBUG_TUPLE: tuple,
) -> None:
    # Visualize
    print("\n" + "=" * 80)
    print("Visualizing")
    print("=" * 80 + "\n")
    from nerf_grasping.curobo_fr3_algr_zed2i.visualizer import (
        start_visualizer,
        draw_collision_spheres_default_config,
        remove_collision_spheres_default_config,
        set_robot_state,
        animate_robot,
        create_urdf,
    )

    from nerf_grasping.curobo_fr3_algr_zed2i.ik_fr3_algr_zed2i import (
        max_penetration_from_qs,
        max_penetration_from_q,
    )

    OBJECT_URDF_PATH = create_urdf(obj_path=pathlib.Path("/tmp/mesh_viz_object.obj"))
    pb_robot = start_visualizer(object_urdf_path=OBJECT_URDF_PATH)
    draw_collision_spheres_default_config(pb_robot)
    time.sleep(1.0)

    if len(success_idxs) == 0:
        print("WARNING: No successful trajectories")

    TRAJ_IDX = success_idxs[0] if len(success_idxs) > 0 else 0

    dts = []
    for q, T_traj in zip(qs, T_trajs):
        n_timesteps = q.shape[0]
        dt = T_traj / n_timesteps
        dts.append(dt)

    remove_collision_spheres_default_config()
    q, qd, dt = qs[TRAJ_IDX], qds[TRAJ_IDX], dts[TRAJ_IDX]
    print(f"Visualizing trajectory {TRAJ_IDX}")
    animate_robot(robot=pb_robot, qs=q, dt=dt)

    while True:
        input_options = "\n".join(
            [
                "=====================",
                "OPTIONS",
                "b for breakpoint",
                "v to visualize traj",
                "d to print collision distance",
                "i to move hand to exact X_W_H and q_algr_pre IK solution",
                "w to visualize waypoints of lift",
                "n to go to next traj",
                "p to go to prev traj",
                "c to draw collision spheres",
                "r to remove collision spheres",
                "q to quit",
                "=====================",
            ]
        )
        x = input("\n" + input_options + "\n\n")
        if x == "b":
            print("Breakpoint")
            breakpoint()
        elif x == "v":
            q, qd, dt = qs[TRAJ_IDX], qds[TRAJ_IDX], dts[TRAJ_IDX]
            print(f"Visualizing trajectory {TRAJ_IDX}")
            animate_robot(robot=pb_robot, qs=q, dt=dt)
        elif x == "d":
            print(
                "WARNING: This doesn't make sense when we include the full trajectory of grasping"
            )

            q, qd, dt = qs[TRAJ_IDX], qds[TRAJ_IDX], dts[TRAJ_IDX]
            print(f"For trajectory {TRAJ_IDX}")
            d_world, d_self = max_penetration_from_qs(
                qs=q,
                collision_activation_distance=0.0,
                include_object=True,
                obj_filepath=pathlib.Path("/tmp/mesh_viz_object.obj"),
                obj_xyz=(cfg.nerf_frame_offset_x, 0.0, 0.0),
                obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                include_table=True,
            )
            print(f"np.max(d_world): {np.max(d_world)}")
            print(f"np.max(d_self): {np.max(d_self)}")
        elif x == "i":
            print(
                f"Moving hand to exact X_W_H and q_algr_pre of trajectory {TRAJ_IDX} with IK collision check"
            )
            ik_result2 = DEBUG_TUPLE[2]  # BRITTLE
            ik_q = ik_result2.solution[TRAJ_IDX].flatten().detach().cpu().numpy()
            assert ik_q.shape == (23,)
            set_robot_state(robot=pb_robot, q=ik_q)
            d_world, d_self = max_penetration_from_q(
                q=ik_q,
                include_object=True,
                obj_filepath=pathlib.Path("/tmp/mesh_viz_object.obj"),
                obj_xyz=(cfg.nerf_frame_offset_x, 0.0, 0.0),
                obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                include_table=True,
            )
            print(f"np.max(d_world): {np.max(d_world)}")
            print(f"np.max(d_self): {np.max(d_self)}")
        elif x == "w":
            print(f"Visualizing waypoints of trajectory {TRAJ_IDX}")
            lift_ik_result = DEBUG_TUPLE[3]  # BRITTLE

            ik_qs = lift_ik_result.solution.detach().cpu().numpy()
            N = ik_qs.shape[0]
            assert ik_qs.shape == (N, 1, 23)
            ik_qs = ik_qs.reshape(N, 23)

            n_grasps = len(qs)
            assert N % n_grasps == 0
            n_waypoints = N // n_grasps

            ik_qs = ik_qs.reshape(n_grasps, n_waypoints, 23)
            ik_q = ik_qs[TRAJ_IDX]
            ik_q[:, 7:] = qs[TRAJ_IDX][-1, 7:]  # Keep hand position the same as closing

            animate_robot(robot=pb_robot, qs=ik_q, dt=1)
        elif x == "n":
            TRAJ_IDX += 1
            if TRAJ_IDX >= len(qs):
                TRAJ_IDX = 0
            print(f"Updated to trajectory {TRAJ_IDX}")
        elif x == "p":
            TRAJ_IDX -= 1
            if TRAJ_IDX < 0:
                TRAJ_IDX = len(qs) - 1
            print(f"Updated to trajectory {TRAJ_IDX}")
        elif x == "c":
            print("Drawing collision spheres")
            draw_collision_spheres_default_config(robot=pb_robot)
        elif x == "r":
            print("Removing collision spheres")
            remove_collision_spheres_default_config()
        elif x == "q":
            print("Quitting")
            break
        else:
            print(f"Invalid input: {x}")

    breakpoint()


def interpolate(start, end, N):
    d = start.shape[0]
    assert start.shape == end.shape == (d,)
    interpolated = np.zeros((N, d))
    for i in range(d):
        interpolated[:, i] = np.linspace(start[i], end[i], N)
    return interpolated


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
        start_time = time.time()
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
        end_time = time.time()
        print("@" * 80)
        print(f"Time to train_nerf: {end_time - start_time:.2f}s")
        print("@" * 80 + "\n")
    elif args.nerfcheckpoint_path is not None:
        start_time = time.time()
        nerf_pipeline = load_nerf_pipeline(args.nerfcheckpoint_path)
        nerf_model = nerf_pipeline.model
        nerf_config = args.nerfcheckpoint_path
        end_time = time.time()
        print("@" * 80)
        print(f"Time to load_nerf_pipeline: {end_time - start_time:.2f}s")
        print("@" * 80 + "\n")
    else:
        raise ValueError(
            "Exactly one of nerfdata_path or nerfcheckpoint_path must be specified"
        )

    args.nerf_config = nerf_config
    qs, qds, T_trajs, success_idxs, DEBUG_TUPLE = run_pipeline(
        nerf_model=nerf_model, cfg=args
    )

    visualize(
        cfg=args,
        qs=qs,
        qds=qds,
        T_trajs=T_trajs,
        success_idxs=success_idxs,
        DEBUG_TUPLE=DEBUG_TUPLE,
    )


if __name__ == "__main__":
    main()
