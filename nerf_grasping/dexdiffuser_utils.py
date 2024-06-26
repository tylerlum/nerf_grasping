from __future__ import annotations
import trimesh
from typing import List, Tuple
from nerf_grasping.ablation_utils import (
    nerf_to_bps,
    visualize_point_cloud_and_bps_and_grasp,
)
from nerf_grasping.dexdiffuser.diffusion import Diffusion
from nerf_grasping.dexdiffuser.diffusion_config import Config, TrainingConfig
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
    AllegroHandConfig,
    GraspMetric,
    DepthImageGraspMetric,
    predict_in_collision_with_object,
    predict_in_collision_with_table,
    get_hand_surface_points_Oy,
    get_joint_limits,
    hand_config_to_hand_model,
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
from nerf_grasping.dexgraspnet_utils.joint_angle_targets import (
    compute_fingertip_dirs,
)

import open3d as o3d


def normalize_with_warning(v: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    B = v.shape[0]
    assert v.shape == (B, 3), f"Expected shape ({B}, 3), got {v.shape}"
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    if np.any(norm < atol):
        print("^" * 80)
        print(
            f"Warning: Found {np.sum(norm < atol)} vectors with norm less than {atol}"
        )
        print("^" * 80)
    return v / (norm + atol)


def rot6d_to_matrix(rot6d: np.ndarray, check: bool = True) -> np.ndarray:
    B = rot6d.shape[0]
    assert rot6d.shape == (B, 6), f"Expected shape ({B}, 6), got {rot6d.shape}"

    # Step 1: Reshape to (B, 3, 2)
    rot3x2 = rot6d.reshape(B, 3, 2)

    # Step 2: Normalize the first column
    col1 = rot3x2[:, :, 0]
    col1_normalized = normalize_with_warning(col1)

    # Step 3: Orthogonalize the second column with respect to the first column
    col2 = rot3x2[:, :, 1]
    dot_product = np.sum(col1_normalized * col2, axis=1, keepdims=True)
    col2_orthogonal = col2 - dot_product * col1_normalized

    # Step 4: Normalize the second column
    col2_normalized = normalize_with_warning(col2_orthogonal)

    # Step 5: Compute the cross product to obtain the third column
    col3 = np.cross(col1_normalized, col2_normalized)

    # Combine the columns to form the rotation matrix
    rotation_matrices = np.stack((col1_normalized, col2_normalized, col3), axis=-1)

    # Step 6: Check orthogonality and determinant
    if check:
        for i in range(B):
            mat = rotation_matrices[i]
            assert np.allclose(
                np.dot(mat.T, mat), np.eye(3), atol=1e-3
            ), f"Matrix {i} is not orthogonal, got {np.dot(mat.T, mat)}"
            assert np.allclose(
                np.linalg.det(mat), 1.0, atol=1e-3
            ), f"Matrix {i} does not have determinant 1, got {np.linalg.det(mat)}"

    assert rotation_matrices.shape == (
        B,
        3,
        3,
    ), f"Expected shape ({B}, 3, 3), got {rotation_matrices.shape}"
    return rotation_matrices


def compute_grasp_orientations(
    grasp_dirs: torch.Tensor,
    wrist_pose: pp.LieTensor,
    joint_angles: torch.Tensor,
) -> pp.LieTensor:
    B = grasp_dirs.shape[0]
    N_FINGERS = 4
    assert grasp_dirs.shape == (
        B,
        N_FINGERS,
        3,
    ), f"Expected shape ({B}, {N_FINGERS}, 3), got {grasp_dirs.shape}"
    assert wrist_pose.lshape == (B,), f"Expected shape ({B},), got {wrist_pose.lshape}"

    # Normalize
    z_dirs = grasp_dirs
    z_dirs = z_dirs / z_dirs.norm(dim=-1, keepdim=True)

    # Get hand model
    hand_config = AllegroHandConfig.from_values(
        wrist_pose=wrist_pose,
        joint_angles=joint_angles,
    )
    hand_model = hand_config_to_hand_model(
        hand_config=hand_config,
    )

    # Math to get x_dirs, y_dirs
    (center_to_right_dirs, center_to_tip_dirs) = compute_fingertip_dirs(
        joint_angles=joint_angles,
        hand_model=hand_model,
    )
    option_1_ok = (
        torch.cross(center_to_tip_dirs, z_dirs).norm(dim=-1, keepdim=True) > 1e-4
    )

    y_dirs = torch.where(
        option_1_ok,
        center_to_tip_dirs
        - (center_to_tip_dirs * z_dirs).sum(dim=-1, keepdim=True) * z_dirs,
        center_to_right_dirs
        - (center_to_right_dirs * z_dirs).sum(dim=-1, keepdim=True) * z_dirs,
    )

    assert (y_dirs.norm(dim=-1).min() > 0).all()
    y_dirs = y_dirs / y_dirs.norm(dim=-1, keepdim=True)

    x_dirs = torch.cross(y_dirs, z_dirs)
    assert (x_dirs.norm(dim=-1).min() > 0).all()
    x_dirs = x_dirs / x_dirs.norm(dim=-1, keepdim=True)
    grasp_orientations = torch.stack([x_dirs, y_dirs, z_dirs], dim=-1)
    # Make sure y and z are orthogonal
    assert (torch.einsum("...l,...l->...", y_dirs, z_dirs).abs().max() < 1e-3).all(), (
        f"y_dirs = {y_dirs}",
        f"z_dirs = {z_dirs}",
        f"torch.einsum('...l,...l->...', y_dirs, z_dirs).abs().max() = {torch.einsum('...l,...l->...', y_dirs, z_dirs).abs().max()}",
    )
    assert grasp_orientations.shape == (
        B,
        N_FINGERS,
        3,
        3,
    ), f"Expected shape ({B}, {N_FINGERS}, 3, 3), got {grasp_orientations.shape}"
    grasp_orientations = pp.from_matrix(
        grasp_orientations,
        pp.SO3_type,
        atol=1e-3,  # Looser tolerances, esp if larger batch dim
        rtol=1e-3,  # Looser tolerances, esp if larger batch dim
    )
    assert grasp_orientations.lshape == (
        B,
        N_FINGERS,
    ), f"Expected shape ({B}, {N_FINGERS}), got {grasp_orientations.lshape}"

    return grasp_orientations


def get_optimized_grasps(
    cfg: OptimizationConfig,
    nerf_pipeline: Pipeline,
    lb_N: np.ndarray,
    ub_N: np.ndarray,
    X_N_By: np.ndarray,
    X_Oy_By: np.ndarray,
    ckpt_path: str | pathlib.Path,
    return_exactly_requested_num_grasps: bool = True,
    sample_grasps_multiplier: int = 10,
) -> dict:
    ckpt_path = pathlib.Path(ckpt_path)

    NUM_GRASPS = cfg.optimizer.num_grasps

    config = Config(
        training=TrainingConfig(
            log_path=ckpt_path.parent,
        )
    )
    runner = Diffusion(config, load_multigpu_ckpt=True)
    runner.load_checkpoint(config, name=ckpt_path.stem)
    device = runner.device

    # Get BPS
    N_BASIS_PTS = 4096
    bps_values, basis_points_By, point_cloud_points_By = nerf_to_bps(
        nerf_pipeline=nerf_pipeline,
        lb_N=lb_N,
        ub_N=ub_N,
        X_N_By=X_N_By,
    )
    assert bps_values.shape == (
        N_BASIS_PTS,
    ), f"Expected shape ({N_BASIS_PTS},), got {bps_values.shape}"

    # We sample more grasps than needed to account for filtering
    NUM_GRASP_SAMPLES = sample_grasps_multiplier * NUM_GRASPS
    bps_values_repeated = torch.from_numpy(bps_values).float().to(device)
    bps_values_repeated = bps_values_repeated.unsqueeze(dim=0).repeat(
        NUM_GRASP_SAMPLES, 1
    )
    assert bps_values_repeated.shape == (
        NUM_GRASP_SAMPLES,
        N_BASIS_PTS,
    ), f"bps_values_repeated.shape = {bps_values_repeated.shape}"

    # Sample grasps
    xT = torch.randn(NUM_GRASP_SAMPLES, config.data.grasp_dim, device=runner.device)
    x = runner.sample(xT=xT, cond=bps_values_repeated)

    PLOT = False
    if PLOT:
        X_By_Oy = np.linalg.inv(X_Oy_By)
        X_By_N = np.linalg.inv(X_N_By)

        mesh_N = trimesh.load("/tmp/mesh_viz_object.obj")
        mesh_By = trimesh.load("/tmp/mesh_viz_object.obj")
        mesh_By.apply_transform(X_By_N)

        IDX = 0
        while True:
            visualize_point_cloud_and_bps_and_grasp(
                grasp=x[IDX],
                X_W_Oy=X_By_Oy,
                basis_points=basis_points_By,
                bps=bps_values_repeated[IDX].detach().cpu().numpy(),
                mesh=mesh_By,
                point_cloud_points=point_cloud_points_By,
                GRASP_IDX="?",
                object_code="?",
                passed_eval="?",
            )
            user_input = input("Next action?")
            if user_input == "q":
                break
            elif user_input == "n":
                IDX += 1
                IDX = IDX % NUM_GRASP_SAMPLES
            elif user_input == "p":
                IDX -= 1
                IDX = IDX % NUM_GRASP_SAMPLES
            else:
                print("Invalid input")
        breakpoint()

    # grasp to AllegroGraspConfig
    # TODO: make the numpy torch conversions less bad
    N_FINGERS = 4
    assert x.shape == (
        NUM_GRASP_SAMPLES,
        config.data.grasp_dim,
    ), f"Expected shape ({NUM_GRASP_SAMPLES}, {config.data.grasp_dim}), got {x.shape}"
    trans = x[:, :3].detach().cpu().numpy()
    rot6d = x[:, 3:9].detach().cpu().numpy()
    joint_angles = x[:, 9:25].detach().cpu().numpy()
    grasp_dirs = (
        x[:, 25:37].reshape(NUM_GRASP_SAMPLES, N_FINGERS, 3).detach().cpu().numpy()
    )

    rot = rot6d_to_matrix(rot6d)

    wrist_pose_matrix = (
        torch.eye(4, device=device).unsqueeze(0).repeat(NUM_GRASP_SAMPLES, 1, 1).float()
    )
    wrist_pose_matrix[:, :3, :3] = torch.from_numpy(rot).float().to(device)
    wrist_pose_matrix[:, :3, 3] = torch.from_numpy(trans).float().to(device)

    try:
        wrist_pose = pp.from_matrix(
            wrist_pose_matrix,
            pp.SE3_type,
        ).to(device)
    except ValueError as e:
        print("Error in pp.from_matrix")
        print(e)
        print("rot = ", rot)
        print("Orthogonalization did not work, running with looser tolerances")
        wrist_pose = pp.from_matrix(
            wrist_pose_matrix,
            pp.SE3_type,
            atol=1e-3,
            rtol=1e-3,
        ).to(device)

    assert wrist_pose.lshape == (NUM_GRASP_SAMPLES,)

    grasp_orientations = compute_grasp_orientations(
        grasp_dirs=torch.from_numpy(grasp_dirs).float().to(device),
        wrist_pose=wrist_pose,
        joint_angles=torch.from_numpy(joint_angles).float().to(device),
    )

    # Convert to AllegroGraspConfig to dict
    grasp_configs = AllegroGraspConfig.from_values(
        wrist_pose=wrist_pose,
        joint_angles=torch.from_numpy(joint_angles).float().to(device),
        grasp_orientations=grasp_orientations,
    )

    if cfg.filter_less_feasible_grasps:
        wrist_pose_matrix = grasp_configs.wrist_pose.matrix()
        x_dirs = wrist_pose_matrix[:, :, 0]
        z_dirs = wrist_pose_matrix[:, :, 2]

        fingers_forward_cos_theta = math.cos(math.radians(cfg.fingers_forward_theta_deg))
        palm_upwards_cos_theta = math.cos(math.radians(cfg.palm_upwards_theta_deg))
        fingers_forward = z_dirs[:, 0] >= fingers_forward_cos_theta
        palm_upwards = x_dirs[:, 1] >= palm_upwards_cos_theta
        grasp_configs = grasp_configs[fingers_forward & ~palm_upwards]
        print(f"Filtered less feasible grasps. New batch size: {grasp_configs.batch_size}")

    if len(grasp_configs) < NUM_GRASPS:
        print(
            f"WARNING: After filtering, only {len(grasp_configs)} grasps remain, less than the requested {NUM_GRASPS} grasps"
        )

    if return_exactly_requested_num_grasps:
        grasp_configs = grasp_configs[:NUM_GRASPS]
    print(f"Returning {grasp_configs.batch_size} grasps")

    grasp_config_dicts = grasp_configs.as_dict()
    grasp_config_dicts["loss"] = np.linspace(
        0, 0.001, len(grasp_configs)
    )  # HACK: Currently don't have a loss, but need something here to sort

    return grasp_config_dicts
