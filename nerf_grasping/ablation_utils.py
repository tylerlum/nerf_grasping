from __future__ import annotations
from tqdm import tqdm
from nerf_grasping.dexdiffuser.dex_evaluator import DexEvaluator
import nerf_grasping
import math
import pypose as pp
from collections import defaultdict
from nerf_grasping.optimizer import (
    sample_random_rotate_transforms_only_around_y,
)
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
from nerfstudio.pipelines.base_pipeline import Pipeline

import open3d as o3d

def nerf_to_bps(
    nerf_pipeline: Pipeline,
    output_dir: pathlib.Path,
    lb_N: np.ndarray,
    ub_N: np.ndarray,
    X_N_By: np.ndarray,
    num_points: int = 5000,
) -> np.ndarray:
    assert lb_N.shape == (3,)
    assert ub_N.shape == (3,)

    # TODO: This is slow because it loads NeRF from file and then outputs point cloud to file
    from nerf_grasping.nerfstudio_point_cloud_copy import ExportPointCloud

    cfg = ExportPointCloud(
        normal_method="open3d",
        bounding_box_min=(lb_N[0], lb_N[1], lb_N[2]),
        bounding_box_max=(ub_N[0], ub_N[1], ub_N[2]),
        num_points=num_points,
    )
    point_cloud = cfg.main(nerf_pipeline)

    #### BELOW IS TIGHTLY CONNECTED TO create_grasp_bps_dataset ####
    # Load point cloud
    from nerf_grasping.dexdiffuser.create_grasp_bps_dataset import process_point_cloud

    point_cloud, _ = point_cloud.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )
    point_cloud, _ = point_cloud.remove_radius_outlier(nb_points=16, radius=0.05)
    points = np.asarray(point_cloud.points)

    inlier_points = process_point_cloud(points)
    N_PTS = inlier_points.shape[0]
    assert inlier_points.shape == (
        N_PTS,
        3,
    ), f"inlier_points.shape = {inlier_points.shape}"

    MIN_N_POINTS = 3000
    assert (
        N_PTS >= MIN_N_POINTS
    ), f"Expected at least {MIN_N_POINTS} points, but got {N_PTS}"
    final_points = inlier_points[:MIN_N_POINTS]

    # Frames
    final_points_N = final_points

    # BPS
    from bps import bps

    N_BASIS_PTS = 4096
    basis_point_path = (
        pathlib.Path(nerf_grasping.get_package_root())
        / "dexdiffuser"
        / "basis_points.npy"
    )
    assert basis_point_path.exists(), f"{basis_point_path} does not exist"
    with open(basis_point_path, "rb") as f:
        basis_points_By = np.load(f)
    assert basis_points_By.shape == (
        N_BASIS_PTS,
        3,
    ), f"Expected shape ({N_BASIS_PTS}, 3), got {basis_points_By.shape}"
    basis_points_N = transform_points(T=X_N_By, points=basis_points_By)
    bps_values = bps.encode(
        final_points_N.unsqueeze(dim=0),
        bps_arrangement="custom",
        bps_cell_type="dists",
        custom_basis=basis_points_N,
        verbose=0,
    ).squeeze(dim=0)
    assert bps_values.shape == (
        N_BASIS_PTS,
    ), f"Expected shape ({N_BASIS_PTS},), got {bps_values.shape}"
    return bps_values


def get_optimized_grasps(
    cfg: OptimizationConfig,
    nerf_pipeline: Pipeline,
    lb_N: np.ndarray,
    ub_N: np.ndarray,
    X_N_By: np.ndarray,
    ckpt_path: str,
    output_dir: pathlib.Path,
) -> dict:
    BATCH_SIZE = cfg.eval_batch_size

    N_BASIS_PTS = 4096
    device = torch.device("cuda")
    dex_evaluator = DexEvaluator(3 + 6 + 16 + 12, N_BASIS_PTS).to(device)
    # HACK
    LOAD_CKPT = False
    if LOAD_CKPT:
        dex_evaluator.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f"Not loading")

    # Get BPS
    bps_values = nerf_to_bps(
        nerf_pipeline=nerf_pipeline,
        output_dir=output_dir,
        lb_N=lb_N,
        ub_N=ub_N,
        X_N_By=X_N_By,
    )
    assert bps_values.shape == (
        N_BASIS_PTS,
    ), f"Expected shape ({N_BASIS_PTS},), got {bps_values.shape}"

    f_O = torch.from_numpy(bps_values).to(device)
    f_O = f_O.unsqueeze(dim=0).repeat(BATCH_SIZE, 1)
    assert f_O.shape == (
        BATCH_SIZE,
        N_BASIS_PTS,
    ), f"f_O.shape = {f_O.shape}"

    # Load grasp configs
    # TODO: Find a way to load a particular split of the grasp_data.
    init_grasp_config_dict = np.load(
        cfg.init_grasp_config_dict_path, allow_pickle=True
    ).item()

    init_grasp_configs = AllegroGraspConfig.from_grasp_config_dict(
        init_grasp_config_dict
    )
    print(f"Loaded {init_grasp_configs.batch_size} initial grasp configs.")

    # Put this here to ensure that the random seed is set before sampling random rotations.
    if cfg.random_seed is not None:
        torch.manual_seed(cfg.random_seed)

    all_success_preds = []
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
            wrist_trans_array = temp_grasp_configs.wrist_pose.translation()
            wrist_rot_array = temp_grasp_configs.wrist_pose.rotation().matrix()
            joint_angles_array = temp_grasp_configs.joint_angles
            grasp_dirs_array = temp_grasp_configs.grasp_dirs
            N_FINGERS = 4
            assert wrist_trans_array.shape == (BATCH_SIZE, 3)
            assert wrist_rot_array.shape == (BATCH_SIZE, 3, 3)
            assert joint_angles_array.shape == (BATCH_SIZE, 16)
            assert grasp_dirs_array.shape == (BATCH_SIZE, N_FINGERS, 3)
            g_O = torch.cat(
                [
                    wrist_trans_array,
                    wrist_rot_array[:, :, :2].view(BATCH_SIZE, 6),
                    joint_angles_array,
                    grasp_dirs_array.view(BATCH_SIZE, 12),
                ],
                dim=1,
            ).to(device=device)
            assert g_O.shape == (BATCH_SIZE, 3 + 6 + 16 + 12)
            success_preds = dex_evaluator(f_O=f_O, g_O=g_O)[:, -1]
            assert success_preds.shape == (
                BATCH_SIZE,
            ), f"success_preds.shape = {success_preds.shape}, expected ({BATCH_SIZE},)"
            all_success_preds.append(success_preds)

        # Aggregate
        all_success_preds = np.concatenate(all_success_preds)
        assert all_success_preds.shape == (new_grasp_configs.batch_size,)

        # Sort by success_preds
        new_all_success_preds = all_success_preds
        ordered_idxs_best_first = np.argsort(new_all_success_preds)[::-1].copy()

        new_grasp_configs = new_grasp_configs[ordered_idxs_best_first]
        sorted_success_preds = new_all_success_preds[ordered_idxs_best_first]

    init_grasp_configs = new_grasp_configs[: cfg.optimizer.num_grasps]

    # TODO: Optimize if needed
    grasp_config_dict = init_grasp_configs.as_dict()
    grasp_config_dict["loss"] = (1 - sorted_success_preds).detach().cpu().numpy()

    print(f"Saving final grasp config dict to {cfg.output_path}")
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cfg.output_path), grasp_config_dict, allow_pickle=True)

    if wandb.run is not None:
        wandb.finish()
    return grasp_config_dict
