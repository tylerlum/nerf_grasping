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
    X_Oy_Hs, q_algr_pres, q_algr_posts, q_algr_extra_open = get_sorted_grasps_from_dict(
        optimized_grasp_config_dict=optimized_grasp_config_dict,
        error_if_no_loss=True,
        check=False,
        print_best=False,
    )

    MODE = "JOINTS_OPEN"
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

    q_stars = None

    X_W_Hs = np.stack([X_W_N @ X_N_Oy @ X_Oy_Hs[i] for i in range(num_grasps)], axis=0)
    assert X_W_Hs.shape == (num_grasps, 4, 4)

    METHOD = "CUROBO"
    if METHOD == "DRAKE":
        run_drake(cfg=cfg, X_W_Hs=X_W_Hs, q_algr_pres=q_algr_pres)
    elif METHOD == "CUROBO":
        run_curobo(cfg=cfg, X_W_Hs=X_W_Hs, q_algr_pres=q_algr_pres)
    else:
        raise ValueError(f"Invalid METHOD: {METHOD}")

    return (
        q_stars,
        X_W_Hs,
        q_algr_pres,
        q_algr_posts,
        mesh_W,
        X_N_Oy,
        q_algr_pres_is_in_limits,
    )


def run_drake(cfg, X_W_Hs, q_algr_pres) -> None:
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
            print("$" * 80)
            print(f"Trajectory optimization skipped for grasp {i}")
            print("$" * 80 + "\n")
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
            print("^" * 80)
            print(f"Trajectory optimization succeeded for grasp {i}")
            print("^" * 80 + "\n")
            passing_trajopt_idxs.append(i)

        except RuntimeError as e:
            print("~" * 80)
            print(f"Trajectory optimization failed for grasp {i}")
            print("~" * 80 + "\n")
            failed_trajopt_idxs.append(i)

    print(f"passing_trajopt_idxs: {passing_trajopt_idxs}")
    print(f"failed_trajopt_idxs: {failed_trajopt_idxs}")
    spline, dspline, T_traj, trajopt = solve_trajopt(
        q_fr3_0=q_fr3_0,
        q_algr_0=q_algr_0,
        q_fr3_f=q_stars[passing_trajopt_idxs[0]][:7],
        q_algr_f=q_stars[passing_trajopt_idxs[0]][7:],
        cfg=trajopt_cfg,
        mesh_path=pathlib.Path("/tmp/mesh_viz_object.obj"),
        visualize=True,
        verbose=False,
        ignore_obj_collision=False,
    )
    breakpoint()


def run_curobo(cfg, X_W_Hs, q_algr_pres):
    num_grasps = X_W_Hs.shape[0]

    from nerf_grasping.curobo_fr3_algr_zed2i.ik_fr3_algr_zed2i import solve_ik as solve_ik_curobo
    from nerf_grasping.curobo_fr3_algr_zed2i.ik_fr3_algr_zed2i import max_penetration_from_X_W_H

    pass_ik_idxs= []
    fail_ik_idxs = []
    for i in tqdm(range(num_grasps), desc="Curobo IK"):
        X_W_H = X_W_Hs[i]
        q_algr_pre = q_algr_pres[i]
        print(f"Trying grasp {i}")
        try:
            q_solution, _, _ = solve_ik_curobo(
                X_W_H=X_W_H,
                q_algr_constraint=q_algr_pre,
                collision_check_object=True,
                obj_filepath=pathlib.Path("/tmp/mesh_viz_object.obj"),
                obj_xyz=(cfg.nerf_frame_offset_x, 0.0, 0.0),
                obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                collision_check_table=True,
                raise_if_no_solution=True,
            )
            pass_ik_idxs.append(i)
        except RuntimeError as e:
            q_solution = None
            fail_ik_idxs.append(i)
        
        d_world, d_self = max_penetration_from_X_W_H(
            X_W_H=X_W_H,
            q_algr_constraint=q_algr_pre,
            include_object=True,
            obj_filepath=pathlib.Path("/tmp/mesh_viz_object.obj"),
            obj_xyz=(cfg.nerf_frame_offset_x, 0.0, 0.0),
            obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            include_table=True,
        )
        print(f"d_world = {d_world}, d_self = {d_self}")
    print(f"pass_ik_idxs: {pass_ik_idxs}")
    print(f"fail_ik_idxs: {fail_ik_idxs}")

    print("\n" + "=" * 80)
    print("Step 9: Solve trajopt for each grasp")
    print("=" * 80 + "\n")

    from nerf_grasping.curobo_fr3_algr_zed2i.trajopt_fr3_algr_zed2i import solve_trajopt as solve_trajopt_curobo

    print("=" * 80)
    print("Trying with full object collision check")
    print("=" * 80 + "\n")
    pass_trajopt_idxs = []
    pass_trajopt_2_idxs = []
    fail_trajopt_idxs = []
    skip_trajopt_idxs = []
    for i in tqdm(range(num_grasps), desc="Curobo TrajOpt"):
        print(f"Trying grasp {i}")
        if i not in pass_ik_idxs:
            print(f"Skipping grasp {i}")
            skip_trajopt_idxs.append(i)
            continue

        X_W_H = X_W_Hs[i]
        q_algr_pre = q_algr_pres[i]
        try:
            raise RuntimeError("Forcing failure")
            print("=" * 80)
            print("Trying with full object collision check with trajopt")
            print("=" * 80 + "\n")
            q, qd, qdd, dt, result, motion_gen = solve_trajopt_curobo(
                X_W_H=X_W_H,
                q_algr_constraint=q_algr_pre,
                collision_check_object=True,
                obj_filepath=pathlib.Path("/tmp/mesh_viz_object.obj"),
                obj_xyz=(cfg.nerf_frame_offset_x, 0.0, 0.0),
                obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                collision_check_table=True,
                enable_opt=True,
                enable_graph=True,
                raise_if_fail=True,
                use_cuda_graph=False
            )
            print("SUCCESS TRAJOPT with full object collision check")
            failed = False
            pass_trajopt_idxs.append(i)
        except RuntimeError as e:
            print("FAILED TRAJOPT with full object collision check")
            failed = True

        if failed:
            print("=" * 80)
            print("Trying with full object collision check without trajopt")
            print("=" * 80 + "\n")
            try:
                q, qd, qdd, dt, result, motion_gen = solve_trajopt_curobo(
                    X_W_H=X_W_H,
                    q_algr_constraint=q_algr_pre,
                    collision_check_object=True,
                    obj_filepath=pathlib.Path("/tmp/mesh_viz_object.obj"),
                    obj_xyz=(cfg.nerf_frame_offset_x, 0.0, 0.0),
                    obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                    collision_check_table=True,
                    enable_opt=False,
                    enable_graph=True,
                    raise_if_fail=False,
                    use_cuda_graph=False,
                )
                print("SUCCESS TRAJOPT with full object collision check without trajopt")
                failed = False
                pass_trajopt_2_idxs.append(i)
            except RuntimeError as e:
                print("FAILED TRAJOPT with full object collision check without trajopt")
                failed = True
                fail_trajopt_idxs.append(i)
    print(f"pass_trajopt_idxs: {pass_trajopt_idxs}")
    print(f"pass_trajopt_2_idxs: {pass_trajopt_2_idxs}")
    print(f"fail_trajopt_idxs: {fail_trajopt_idxs}")
    pass_trajopt_2_idxs = [1, 2, 5, 12, 21, 28]  # HACK

    # Visualize
    OBJECT_URDF_PATH = create_urdf(obj_path=pathlib.Path("/tmp/mesh_viz_object.obj"))
    q, qd, qdd, dt, result, motion_gen = solve_trajopt_curobo(
        X_W_H=X_W_Hs[pass_trajopt_2_idxs[0]],
        q_algr_constraint=q_algr_pres[pass_trajopt_2_idxs[0]],
        collision_check_object=True,
        obj_filepath=pathlib.Path("/tmp/mesh_viz_object.obj"),
        obj_xyz=(cfg.nerf_frame_offset_x, 0.0, 0.0),
        obj_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        collision_check_table=True,
        enable_opt=False,
        enable_graph=True,
        raise_if_fail=True,
        use_cuda_graph=False
    )
    import pybullet as pb
    from curobo.util_file import (
        get_robot_configs_path,
        join_path,
        load_yaml,
    )

    from nerf_grasping.curobo_fr3_algr_zed2i.pybullet_utils import (
        draw_collision_spheres,
        remove_collision_spheres,
    )
    import yaml
    import time

    from nerf_grasping.curobo_fr3_algr_zed2i.trajopt_fr3_algr_zed2i import (
        DEFAULT_Q_ALGR,
        DEFAULT_Q_FR3,
    )
    FR3_ALGR_ZED2I_URDF_PATH = pathlib.Path(
        "/juno/u/tylerlum/github_repos/nerf_grasping/nerf_grasping/fr3_algr_ik/allegro_ros2/models/fr3_algr_zed2i.urdf"
    )
    assert FR3_ALGR_ZED2I_URDF_PATH.exists()

    OBJECT_OBJ_PATH = pathlib.Path("/tmp/mesh_viz_object.obj")
    assert OBJECT_OBJ_PATH.exists()

    # %%
    COLLISION_SPHERES_YAML_PATH = load_yaml(
        join_path(get_robot_configs_path(), "fr3_algr_zed2i.yml")
    )["robot_cfg"]["kinematics"]["collision_spheres"]
    COLLISION_SPHERES_YAML_PATH = pathlib.Path(
        join_path(get_robot_configs_path(), COLLISION_SPHERES_YAML_PATH)
    )
    assert COLLISION_SPHERES_YAML_PATH.exists()


    pb.connect(pb.GUI)
    r = pb.loadURDF(
        str(FR3_ALGR_ZED2I_URDF_PATH),
        useFixedBase=True,
        basePosition=[0, 0, 0],
        baseOrientation=[0, 0, 0, 1],
    )
    num_total_joints = pb.getNumJoints(r)
    assert num_total_joints == 39

    obj = pb.loadURDF(
        str(OBJECT_URDF_PATH),
        useFixedBase=True,
        basePosition=[
            0.65,
            0,
            0,
        ],
        baseOrientation=[0, 0, 0, 1],
    )

    # %%
    joint_names = [
        pb.getJointInfo(r, i)[1].decode("utf-8")
        for i in range(num_total_joints)
        if pb.getJointInfo(r, i)[2] != pb.JOINT_FIXED
    ]
    link_names = [
        pb.getJointInfo(r, i)[12].decode("utf-8")
        for i in range(num_total_joints)
        if pb.getJointInfo(r, i)[2] != pb.JOINT_FIXED
    ]

    actuatable_joint_idxs = [
        i for i in range(num_total_joints) if pb.getJointInfo(r, i)[2] != pb.JOINT_FIXED
    ]
    num_actuatable_joints = len(actuatable_joint_idxs)
    assert num_actuatable_joints == 23
    arm_actuatable_joint_idxs = actuatable_joint_idxs[:7]
    hand_actuatable_joint_idxs = actuatable_joint_idxs[7:]

    for i, joint_idx in enumerate(arm_actuatable_joint_idxs):
        pb.resetJointState(r, joint_idx, DEFAULT_Q_FR3[i])

    for i, joint_idx in enumerate(hand_actuatable_joint_idxs):
        pb.resetJointState(r, joint_idx, DEFAULT_Q_ALGR[i])

    # %%
    collision_config = yaml.safe_load(
        open(
            COLLISION_SPHERES_YAML_PATH,
            "r",
        )
    )
    draw_collision_spheres(
        robot=r,
        config=collision_config,
    )

    # %%
    N_pts = q.shape[0]
    remove_collision_spheres()

    last_update_time = time.time()
    for i in tqdm(range(N_pts)):
        position = q[i]
        assert position.shape == (23,)
        # print(f"{i} / {N_pts} {position}")

        for i, joint_idx in enumerate(arm_actuatable_joint_idxs):
            pb.resetJointState(r, joint_idx, position[i])
        for i, joint_idx in enumerate(hand_actuatable_joint_idxs):
            pb.resetJointState(r, joint_idx, position[i + 7])

        time_since_last_update = time.time() - last_update_time
        if time_since_last_update <= dt:
            time.sleep(dt - time_since_last_update)
        last_update_time = time.time()

    while True:
        x = input("Press v to visualize traj, c to draw collision spheres, r to remove collision spheres, q to quit")
        if x == "v":
            last_update_time = time.time()
            for i in tqdm(range(N_pts)):
                position = q[i]
                assert position.shape == (23,)
                # print(f"{i} / {N_pts} {position}")

                for i, joint_idx in enumerate(arm_actuatable_joint_idxs):
                    pb.resetJointState(r, joint_idx, position[i])
                for i, joint_idx in enumerate(hand_actuatable_joint_idxs):
                    pb.resetJointState(r, joint_idx, position[i + 7])

                time_since_last_update = time.time() - last_update_time
                if time_since_last_update <= dt:
                    time.sleep(dt - time_since_last_update)
                last_update_time = time.time()
        elif x == "c":
            draw_collision_spheres(
                robot=r,
                config=collision_config,
            )
        elif x == "r":
            remove_collision_spheres()
        elif x == "q":
            break

    breakpoint()

def create_urdf(obj_path: pathlib.Path) -> pathlib.Path:
    assert obj_path.suffix == ".obj"
    filename = obj_path.name
    parent_folder = obj_path.parent
    urdf_path = parent_folder / f"{obj_path.stem}.urdf"
    urdf_text = f"""<?xml version="0.0" ?>
<robot name="model.urdf">
  <link name="baseLink">
    <contact>
      <lateral_friction value="0.8"/>
      <rolling_friction value="0.001"/>g
      <contact_cfm value="0.0"/>
      <contact_erp value="1.0"/>
    </contact>
    <inertial>
    <origin rpy="0 0 0" xyz="0.01 0.0 0.01"/>
       <mass value=".066"/>
       <inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/>
    </inertial>
    <visual>
      <geometry>
        <mesh filename="{filename}" scale="1 1 1"/>
      </geometry>
      <material name="white">
        <color rgba="1. 1. 1. 1."/>
      </material>
    </visual>
    <collision>
      <geometry>
    	 	<mesh filename="{filename}" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
</robot>"""
    with urdf_path.open("w") as f:
        f.write(urdf_text)
    return urdf_path

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
