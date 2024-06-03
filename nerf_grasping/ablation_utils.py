from __future__ import annotations
import math
import pypose as pp
from collections import defaultdict
from nerf_grasping.optimizer_utils import (
    AllegroGraspConfig,
    GraspMetric,
    DepthImageGraspMetric,
    predict_in_collision_with_object,
    predict_in_collision_with_table,
    get_hand_surface_points_Oy,
    get_joint_limits,
)
from dataclasses import asdict
from nerf_grasping.config.optimization_config import OptimizationConfig
import pathlib
import torch
from nerf_grasping.classifier import Classifier, Simple_CNN_LSTM_Classifier
from nerf_grasping.config.classifier_config import ClassifierConfig
from nerf_grasping.config.nerfdata_config import DepthImageNerfDataConfig
from nerf_grasping.config.optimizer_config import (
    SGDOptimizerConfig,
    CEMOptimizerConfig,
    RandomSamplingConfig,
)
from typing import Tuple, Union, Dict
import nerf_grasping
from functools import partial
import numpy as np
import tyro
import wandb

from rich.console import Console
from rich.table import Table

from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from contextlib import nullcontext
import plotly.graph_objects as go
import numpy as np
import pathlib
import pytorch_kinematics as pk
from pytorch_kinematics.chain import Chain
import pypose as pp
import torch

import nerf_grasping
from nerf_grasping import grasp_utils

from typing import List, Tuple, Dict, Any, Iterable, Union, Optional
from nerfstudio.fields.base_field import Field
from nerfstudio.models.base_model import Model
from nerf_grasping.classifier import (
    Classifier,
    DepthImageClassifier,
    Simple_CNN_LSTM_Classifier,
)
from nerf_grasping.learned_metric.DexGraspNet_batch_data import (
    BatchDataInput,
    DepthImageBatchDataInput,
)
from nerf_grasping.dataset.DexGraspNet_NeRF_Grasps_utils import (
    transform_point,
    transform_points,
)
from nerf_grasping.nerf_utils import (
    get_cameras,
    render,
    get_densities_in_grid,
    get_density,
)
from nerf_grasping.config.grasp_metric_config import GraspMetricConfig
from nerf_grasping.config.fingertip_config import UnionFingertipConfig
from nerf_grasping.config.camera_config import CameraConfig
from nerf_grasping.config.classifier_config import ClassifierConfig
from nerf_grasping.config.nerfdata_config import DepthImageNerfDataConfig
from nerf_grasping.dataset.nerf_densities_global_config import (
    NERF_DENSITIES_GLOBAL_NUM_X,
    NERF_DENSITIES_GLOBAL_NUM_Y,
    NERF_DENSITIES_GLOBAL_NUM_Z,
    lb_Oy,
    ub_Oy,
)
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from contextlib import nullcontext


def nerf_to_point_cloud(
    nerf_config: pathlib.Path,
    output_dir: pathlib.Path,
    lb_N: np.ndarray,
    ub_N: np.ndarray,
    X_N_Oy: np.ndarray,
    num_points: int = 5000,
):
    assert lb_N.shape == (3,)
    assert ub_N.shape == (3,)

    # TODO: This is slow because it loads NeRF from file and then outputs point cloud to file
    from nerfstudio.scripts.export import ExportPointCloud
    cfg = ExportPointCloud(
        load_config=nerf_config,
        output_dir=output_dir,
        normal_method="open3d",
        bounding_box_min=(lb_N[0], lb_N[1], lb_N[2]),
        bounding_box_max=(ub_N[0], ub_N[1], ub_N[2]),
        num_points=num_points,
    )
    cfg.main()

    assert output_dir.exists(), f"{output_dir} does not exist"
    point_cloud_files = sorted(list(output_dir.glob("*.ply")))
    assert len(point_cloud_files) == 1, f"Expected 1 ply file, but got {point_cloud_files}"
    point_cloud_file = point_cloud_files[0]

    #### BELOW IS TIGHTLY CONNECTED TO create_grasp_bps_dataset ####
    # Load point cloud
    from nerf_grasping.dexdiffuser.create_grasp_bps_dataset import process_point_cloud
    point_cloud = o3d.io.read_point_cloud(str(point_cloud_file))
    point_cloud, _ = point_cloud.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )
    point_cloud, _ = point_cloud.remove_radius_outlier(nb_points=16, radius=0.05)
    points = np.asarray(point_cloud.points)

    inlier_points = process_point_cloud(points)
    N_PTS = inlier_points.shape[0]
    assert inlier_points.shape == (N_PTS, 3), f"inlier_points.shape = {inlier_points.shape}"

    MIN_N_POINTS = 3000
    assert N_PTS >= MIN_N_POINTS, f"Expected at least {MIN_N_POINTS} points, but got {N_PTS}"
    final_points = inlier_points[:MIN_N_POINTS]

    # Frames
    X_Oy_N = np.linalg.inv(X_N_Oy)
    final_points_N = final_points
    final_points_Oy = transform_points(T=X_Oy_N, points=final_points_N)

    # BPS
    from bps import bps
    N_BASIS_PTS = 4096
    BASIS_RADIUS = 0.3
    basis_points = bps.generate_random_basis(
        n_points=N_BASIS_PTS, radius=BASIS_RADIUS, random_seed=13
    ) + np.array(
        [0.0, BASIS_RADIUS / 2, 0.0]
    )  # Shift up to get less under the table
    basis_points_Oy = basis_points
    assert basis_points.shape == (
        cfg.N_BASIS_PTS,
        3,
    ), f"Expected shape ({cfg.N_BASIS_PTS}, 3), got {basis_points.shape}"
    bps_values = bps.encode(
        final_points_Oy.unsqueeze(dim=0),
        bps_arrangement="custom",
        bps_cell_type="dists",
        custom_basis=basis_points_Oy,
        verbose=0,
    ).squeeze(dim=0)
    assert bps_values.shape == (
        cfg.N_BASIS_PTS,
    ), f"Expected shape ({cfg.N_BASIS_PTS},), got {bps_values.shape}"
    return bps_values


    


class AblationGraspMetric(torch.nn.Module):
    """
    Wrapper for NeRF + grasp classifier to evaluate
    a particular AllegroGraspConfig.
    """

    def __init__(
        self,
        nerf_field: Field,
        ablation_model: Classifier,
        fingertip_config: UnionFingertipConfig,
        X_N_Oy: np.ndarray,
        return_type: str = "failure_probability",
    ) -> None:
        super().__init__()
        self.nerf_field = nerf_field
        self.ablation_model = ablation_model
        self.fingertip_config = fingertip_config
        self.X_N_Oy = X_N_Oy
        self.ray_origins_finger_frame = grasp_utils.get_ray_origins_finger_frame(
            fingertip_config
        )
        self.return_type = return_type

    def forward(
        self,
        grasp_config: AllegroGraspConfig,
    ) -> torch.Tensor:
        B = grasp_config.batch_size

        # Grasp
        wrist_trans_array = (
            grasp_config.wrist_pose.translation()
        )
        wrist_rot_array = (
            grasp_config.wrist_pose.rotation().matrix()
        )
        joint_angles_array = grasp_config.joint_angles
        grasp_dirs_array = grasp_config.grasp_dirs
        assert wrist_trans_array.shape == (B, 3)
        assert wrist_rot_array.shape == (B, 3, 3)
        assert joint_angles_array.shape == (B, 16)
        assert grasp_dirs_array.shape == (B, 4, 3)
        g = torch.cat(
            [
                wrist_trans_array,
                wrist_rot_array[:, :, :2].reshape(B, 6),
                joint_angles_array,
                grasp_dirs_array.reshape(B, 4*3),
            ],
            dim=1,
        )
        assert g.shape == (
            B,
            3 + 6 + 16 + 4*3,
        ), f"g.shape = {g.shape}"

        # BPS
        f_O =
        assert f_O.shape == (B, 4096), f"f_O.shape = {f_O.shape}"
        self.ablation_model(f_O=f_O, g=g)

    def compute_ray_samples(
        self,
        grasp_config: AllegroGraspConfig,
    ) -> torch.Tensor:
        # Let Oy be object yup frame (centroid of object)
        # Let N be nerf frame (where the nerf is defined)
        # For NeRFs trained from sim data, Oy and N are the same.
        # But for real-world data, Oy and N are different (N is a point on the table, used as NeRF origin)
        # When sampling from the NeRF, we must give ray samples in N frame
        # But classifier is trained on Oy frame
        # Thus, we must transform grasp_frame_transforms from Oy frame to N frame
        # Let Fi be the finger frame (origin at each fingertip i)
        # Let p_Fi be points in Fi frame
        # self.ray_origins_finger_frame = p_Fi
        # grasp_frame_transforms = T_{Oy <- Fi}
        # X_N_Oy = T_{N <- Oy}
        # TODO: Batch this to avoid OOM (refer to Create_DexGraspNet_NeRF_Grasps_Dataset.py)

        # Prepare transforms
        T_Oy_Fi = grasp_config.grasp_frame_transforms
        assert T_Oy_Fi.lshape == (grasp_config.batch_size, grasp_config.num_fingers)

        assert self.X_N_Oy.shape == (
            4,
            4,
        )
        X_N_Oy_repeated = (
            torch.from_numpy(self.X_N_Oy)
            .float()
            .unsqueeze(dim=0)
            .repeat_interleave(
                grasp_config.batch_size * grasp_config.num_fingers, dim=0
            )
            .reshape(grasp_config.batch_size, grasp_config.num_fingers, 4, 4)
        )

        T_N_Oy = pp.from_matrix(
            X_N_Oy_repeated,
            pp.SE3_type,
        ).to(T_Oy_Fi.device)

        # Transform grasp_frame_transforms to nerf frame
        T_N_Fi = T_N_Oy @ T_Oy_Fi

        # Generate RaySamples.
        ray_samples = grasp_utils.get_ray_samples(
            self.ray_origins_finger_frame,
            T_N_Fi,
            self.fingertip_config,
        )
        return ray_samples

    def compute_nerf_densities(
        self,
        ray_samples,
    ) -> torch.Tensor:
        # Query NeRF at RaySamples.
        densities = self.nerf_field.get_density(ray_samples.to("cuda"))[0][
            ..., 0
        ]  # Shape [B, 4, n_x, n_y, n_z]
        return densities

    def get_failure_probability(
        self,
        grasp_config: AllegroGraspConfig,
    ) -> torch.Tensor:
        return self(grasp_config)

    @classmethod
    def from_config(
        cls,
        grasp_metric_config: GraspMetricConfig,
        console: Optional[Console] = None,
    ) -> AblationGraspMetric:
        assert grasp_metric_config.X_N_Oy is not None
        return cls.from_configs(
            nerf_config=grasp_metric_config.nerf_checkpoint_path,
            classifier_config=grasp_metric_config.classifier_config,
            X_N_Oy=grasp_metric_config.X_N_Oy,
            classifier_checkpoint=grasp_metric_config.classifier_checkpoint,
            console=console,
        )

    @classmethod
    def from_configs(
        cls,
        nerf_config: pathlib.Path,
        classifier_config: ClassifierConfig,
        X_N_Oy: np.ndarray,
        classifier_checkpoint: int = -1,
        console: Optional[Console] = None,
    ) -> AblationGraspMetric:
        assert not isinstance(
            classifier_config.nerfdata_config, DepthImageNerfDataConfig
        ), f"classifier_config.nerfdata_config must not be a DepthImageNerfDataConfig, but is {classifier_config.nerfdata_config}"

        # Load nerf
        with (
            Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description} "),
                TimeElapsedColumn(),
                console=console,
            )
            if console is not None
            else nullcontext()
        ) as progress:
            task = (
                progress.add_task("Loading NeRF", total=1)
                if progress is not None
                else None
            )

            nerf_field = grasp_utils.load_nerf_field(nerf_config)

            if progress is not None and task is not None:
                progress.update(task, advance=1)

        # Load classifier
        classifier = load_classifier(
            classifier_config=classifier_config,
            classifier_checkpoint=classifier_checkpoint,
            console=console,
        )

        return cls(
            nerf_field,
            classifier,
            classifier_config.nerfdata_config.fingertip_config,
            X_N_Oy,
        )


def get_optimized_grasps() -> dict:
    # Create rich.Console object.
    if cfg.random_seed is not None:
        torch.random.manual_seed(cfg.random_seed)
        np.random.seed(cfg.random_seed)

    # TODO: Find a way to load a particular split of the grasp_data.
    init_grasp_config_dict = np.load(
        cfg.init_grasp_config_dict_path, allow_pickle=True
    ).item()

    # HACK: For now, just take every 400th grasp.
    # for key in init_grasp_config_dict.keys():
    #     init_grasp_config_dict[key] = init_grasp_config_dict[key][::400]

    init_grasp_configs = AllegroGraspConfig.from_grasp_config_dict(
        init_grasp_config_dict
    )
    print(f"Loaded {init_grasp_configs.batch_size} initial grasp configs.")

    # Create grasp metric
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Put this here to ensure that the random seed is set before sampling random rotations.
    if cfg.random_seed is not None:
        torch.manual_seed(cfg.random_seed)

    BATCH_SIZE = cfg.eval_batch_size
    all_success_preds = []
    all_predicted_in_collision_obj = []
    all_predicted_in_collision_table = []
    with torch.no_grad():
        # Sample random rotations
        N_SAMPLES = 1 + cfg.n_random_rotations_per_grasp
        new_grasp_configs_list = []
        for i in range(N_SAMPLES):
            new_grasp_configs = AllegroGraspConfig.from_grasp_config_dict(
                init_grasp_config_dict
            )
            if i != 0:
                random_rotate_transforms = (
                    sample_random_rotate_transforms_only_around_y(
                        new_grasp_configs.batch_size
                    )
                )
                new_grasp_configs.hand_config.set_wrist_pose(
                    random_rotate_transforms @ new_grasp_configs.hand_config.wrist_pose
                )
            new_grasp_configs_list.append(new_grasp_configs)

        new_grasp_config_dicts = defaultdict(list)
        for i in range(N_SAMPLES):
            config_dict = new_grasp_configs_list[i].as_dict()
            for k, v in config_dict.items():
                new_grasp_config_dicts[k].append(v)
        for k, v in new_grasp_config_dicts.items():
            new_grasp_config_dicts[k] = np.concatenate(v, axis=0)
        new_grasp_configs = AllegroGraspConfig.from_grasp_config_dict(
            new_grasp_config_dicts
        )
        assert new_grasp_configs.batch_size == init_grasp_configs.batch_size * N_SAMPLES

        # Filter grasps that are less IK feasible
        if cfg.filter_less_feasible_grasps:
            wrist_pose_matrix = new_grasp_configs.wrist_pose.matrix()
            x_dirs = wrist_pose_matrix[:, :, 0]
            z_dirs = wrist_pose_matrix[:, :, 2]

            cos_theta = math.cos(math.radians(60))
            fingers_forward = z_dirs[:, 0] >= cos_theta
            palm_upwards = x_dirs[:, 1] >= cos_theta
            new_grasp_configs = new_grasp_configs[fingers_forward & ~palm_upwards]
            print(
                f"Filtered less feasible grasps. New batch size: {new_grasp_configs.batch_size}"
            )

        # HACK TO DEBUG 1
        # new_grasp_configs = new_grasp_configs[[0,1]]

        # Evaluate grasp metric and collisions
        n_batches = math.ceil(new_grasp_configs.batch_size / BATCH_SIZE)
        for batch_i in tqdm(
            range(n_batches), desc=f"Evaling grasp metric with batch_size={BATCH_SIZE}"
        ):
            start_idx = batch_i * BATCH_SIZE
            end_idx = np.clip(
                (batch_i + 1) * BATCH_SIZE,
                a_min=None,
                a_max=new_grasp_configs.batch_size,
            )

            temp_grasp_configs = new_grasp_configs[start_idx:end_idx].to(device=device)

            # Metric
            success_preds = (
                (1 - grasp_metric.get_failure_probability(temp_grasp_configs))
                .detach()
                .cpu()
                .numpy()
            )
            all_success_preds.append(success_preds)

            # Collision with object and table
            USE_OBJECT = False
            USE_TABLE = False
            hand_surface_points_Oy = None
            if USE_OBJECT or USE_TABLE:
                hand_surface_points_Oy = get_hand_surface_points_Oy(
                    grasp_config=temp_grasp_configs
                )
            if USE_OBJECT:
                predicted_in_collision_obj = predict_in_collision_with_object(
                    nerf_field=grasp_metric.nerf_field,
                    hand_surface_points_Oy=hand_surface_points_Oy,
                )
                all_predicted_in_collision_obj.append(predicted_in_collision_obj)
            else:
                all_predicted_in_collision_obj.append(np.zeros_like(success_preds))
            if USE_TABLE:
                table_y_Oy = -cfg.grasp_metric.X_N_Oy[2, 3]
                predicted_in_collision_table = predict_in_collision_with_table(
                    table_y_Oy=table_y_Oy,
                    hand_surface_points_Oy=hand_surface_points_Oy,
                )
                all_predicted_in_collision_table.append(predicted_in_collision_table)
            else:
                all_predicted_in_collision_table.append(np.zeros_like(success_preds))

        # Aggregate
        all_success_preds = np.concatenate(all_success_preds)
        all_predicted_in_collision_obj = np.concatenate(all_predicted_in_collision_obj)
        all_predicted_in_collision_table = np.concatenate(
            all_predicted_in_collision_table
        )
        assert all_success_preds.shape == (new_grasp_configs.batch_size,)
        assert all_predicted_in_collision_obj.shape == (new_grasp_configs.batch_size,)
        assert all_predicted_in_collision_table.shape == (new_grasp_configs.batch_size,)

        # Filter out grasps that are in collision
        new_all_success_preds = np.where(
            np.logical_or(
                all_predicted_in_collision_obj, all_predicted_in_collision_table
            ),
            np.zeros_like(all_success_preds),
            all_success_preds,
        )
        ordered_idxs_best_first = np.argsort(new_all_success_preds)[::-1].copy()
        print("=" * 80)
        print(f"ordered_idxs_best_first = {ordered_idxs_best_first[:10]}")
        print("=" * 80)
        breakpoint()
        # breakpoint()  # TODO: Debug here
        # ordered_idxs_best_first = [550, 759, 524, 151, 150, 533, 1179, 662, 591, 638]
        ordered_idxs_best_first = [981, 937, 985, 874, 135, 65, 987, 1262, 1065, 472]

        print(f'Forced ordered_idxs_best_first = {ordered_idxs_best_first[:10]}')
        new_grasp_configs = new_grasp_configs[ordered_idxs_best_first]

    init_grasp_configs = new_grasp_configs[: cfg.optimizer.num_grasps]

    # Create Optimizer.
    if isinstance(cfg.optimizer, SGDOptimizerConfig):
        optimizer = SGDOptimizer(
            init_grasp_configs,
            grasp_metric,
            cfg.optimizer,
        )
    elif isinstance(cfg.optimizer, CEMOptimizerConfig):
        optimizer = CEMOptimizer(
            init_grasp_configs,
            grasp_metric,
            cfg.optimizer,
        )
    elif isinstance(cfg.optimizer, RandomSamplingConfig):
        optimizer = RandomSamplingOptimizer(
            init_grasp_configs,
            grasp_metric,
            cfg.optimizer,
        )
    else:
        raise ValueError(f"Invalid optimizer config: {cfg.optimizer}")

