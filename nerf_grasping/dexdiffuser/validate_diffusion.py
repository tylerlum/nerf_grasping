import os
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
import time
import torch
import numpy as np
import torch.optim as optim
import torch.utils.data as data
from nerf_grasping.dexdiffuser.dex_sampler import DexSampler
from nerf_grasping.dexdiffuser.diffusion_config import Config, TrainingConfig
from nerf_grasping.dexdiffuser.grasp_bps_dataset import GraspBPSSampleDataset
from torch.utils.data import random_split
from nerf_grasping.dexdiffuser.diffusion import Diffusion, get_dataset

def main() -> None:
    config = Config(
        training=TrainingConfig(
            log_path=Path("/home/albert/research/nerf_grasping/nerf_grasping/dexdiffuser/logs_20240531201249"),
        ),
    )
    runner = Diffusion(config)
    runner.load_checkpoint(config)

    train_dataset, test_dataset, full_dataset = get_dataset(config)
    GRASP_IDX = 0
    _, bps, _ = test_dataset[GRASP_IDX]
    xT = torch.randn(1, config.data.grasp_dim, device=runner.device)
    x = runner.sample(xT=xT, cond=bps[None].to(runner.device)).squeeze().cpu()
    print(f"Sampled grasp shape: {x.shape}")

    import numpy as np
    import trimesh
    import pathlib
    import transforms3d
    import open3d as o3d
    import plotly.graph_objects as go
    from nerf_grasping.dexgraspnet_utils.hand_model import HandModel
    from nerf_grasping.dexgraspnet_utils.hand_model_type import (
        HandModelType,
    )
    from nerf_grasping.dexgraspnet_utils.pose_conversion import (
        hand_config_to_pose,
    )
    from nerf_grasping.dexgraspnet_utils.joint_angle_targets import (
        compute_optimized_joint_angle_targets_given_grasp_orientations,
    )

    MESHDATA_ROOT = (
        "/home/albert/research/nerf_grasping/rsync_meshes/rotated_meshdata_v2"
    )
    print("=" * 79)
    print(f"len(full_dataset): {len(full_dataset)}")

    print("\n" + "=" * 79)
    print(f"Getting grasp and bps for grasp_idx {GRASP_IDX}")
    print("=" * 79)
    grasp = x
    passed_eval = np.array(1)
    # grasp, bps, passed_eval = full_dataset[GRASP_IDX]
    print(f"grasp.shape: {grasp.shape}")
    print(f"bps.shape: {bps.shape}")
    print(f"passed_eval.shape: {passed_eval.shape}")

    print("\n" + "=" * 79)
    print("Getting debugging extras")
    print("=" * 79)
    basis_points = full_dataset.get_basis_points()
    object_code = full_dataset.get_object_code(GRASP_IDX)
    object_scale = full_dataset.get_object_scale(GRASP_IDX)
    object_state = full_dataset.get_object_state(GRASP_IDX)
    print(f"basis_points.shape: {basis_points.shape}")

    # Mesh
    mesh_path = pathlib.Path(f"{MESHDATA_ROOT}/{object_code}/coacd/decomposed.obj")
    assert mesh_path.exists(), f"{mesh_path} does not exist"
    print(f"Reading mesh from {mesh_path}")
    mesh = trimesh.load(mesh_path)

    xyz, quat_xyzw = object_state[:3], object_state[3:7]
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    transform = np.eye(4)  # X_W_Oy
    transform[:3, :3] = transforms3d.quaternions.quat2mat(quat_wxyz)
    transform[:3, 3] = xyz
    mesh.apply_scale(object_scale)
    mesh.apply_transform(transform)

    # Point cloud
    point_cloud_filepath = full_dataset.get_point_cloud_filepath(GRASP_IDX)
    print(f"Reading point cloud from {point_cloud_filepath}")
    point_cloud = o3d.io.read_point_cloud(point_cloud_filepath)
    point_cloud, _ = point_cloud.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )
    point_cloud, _ = point_cloud.remove_radius_outlier(nb_points=16, radius=0.05)
    point_cloud_points = np.asarray(point_cloud.points)
    print(f"point_cloud_points.shape: {point_cloud_points.shape}")

    # Grasp
    assert grasp.shape == (
        3 + 6 + 16 + 4 * 3,
    ), f"Expected shape (3 + 6 + 16 + 4 * 3), got {grasp.shape}"
    grasp = grasp.detach().cpu().numpy()
    grasp_trans, grasp_rot6d, grasp_joints, grasp_dirs = (
        grasp[:3],
        grasp[3:9],
        grasp[9:25],
        grasp[25:].reshape(4, 3),
    )
    grasp_rot = np.zeros((3, 3))
    grasp_rot[:3, :2] = grasp_rot6d.reshape(3, 2)
    grasp_rot[:3, 0] = grasp_rot[:3, 0] / np.linalg.norm(grasp_rot[:3, 0])
    # make grasp_rot[:3, 1] orthogonal to grasp_rot[:3, 0]
    grasp_rot[:3, 1] = (
        grasp_rot[:3, 1]
        - np.dot(grasp_rot[:3, 0], grasp_rot[:3, 1]) * grasp_rot[:3, 0]
    )
    grasp_rot[:3, 1] = grasp_rot[:3, 1] / np.linalg.norm(grasp_rot[:3, 1])
    assert (
        np.dot(grasp_rot[:3, 0], grasp_rot[:3, 1]) < 1e-3
    ), f"Expected dot product < 1e-3, got {np.dot(grasp_rot[:3, 0], grasp_rot[:3, 1])}"
    grasp_rot[:3, 2] = np.cross(grasp_rot[:3, 0], grasp_rot[:3, 1])
    grasp_transform = np.eye(4)  # X_Oy_H
    grasp_transform[:3, :3] = grasp_rot
    grasp_transform[:3, 3] = grasp_trans
    print(f"grasp_transform:\n{grasp_transform}")
    grasp_transform = transform @ grasp_transform  # X_W_H = X_W_Oy @ X_Oy_H
    grasp_trans = grasp_transform[:3, 3]
    grasp_rot = grasp_transform[:3, :3]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hand_pose = hand_config_to_pose(
        grasp_trans[None], grasp_rot[None], grasp_joints[None]
    ).to(device)
    hand_model_type = HandModelType.ALLEGRO_HAND
    grasp_orientations = np.zeros(
        (4, 3, 3)
    )  # NOTE: should have applied transform with this, but didn't because we only have z-dir, hopefully transforms[:3, :3] ~= np.eye(3)
    grasp_orientations[:, :, 2] = (
        grasp_dirs  # Leave the x-axis and y-axis as zeros, hacky but works
    )
    hand_model = HandModel(hand_model_type=hand_model_type, device=device)
    hand_model.set_parameters(hand_pose)
    hand_plotly = hand_model.get_plotly_data(i=0, opacity=0.8)

    (
        optimized_joint_angle_targets,
        _,
    ) = compute_optimized_joint_angle_targets_given_grasp_orientations(
        joint_angles_start=hand_model.hand_pose[:, 9:],
        hand_model=hand_model,
        grasp_orientations=torch.from_numpy(grasp_orientations[None]).to(device),
    )
    new_hand_pose = hand_config_to_pose(
        grasp_trans[None],
        grasp_rot[None],
        optimized_joint_angle_targets.detach().cpu().numpy(),
    ).to(device)
    hand_model.set_parameters(new_hand_pose)
    hand_plotly_optimized = hand_model.get_plotly_data(
        i=0, opacity=0.3, color="lightgreen"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=basis_points[:, 0],
            y=basis_points[:, 1],
            z=basis_points[:, 2],
            mode="markers",
            marker=dict(
                size=1,
                color=bps,
                colorscale="rainbow",
                colorbar=dict(title="Basis points", orientation="h"),
            ),
            name="Basis points",
        )
    )
    fig.add_trace(
        go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1],
            z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0],
            j=mesh.faces[:, 1],
            k=mesh.faces[:, 2],
            name="Object",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=point_cloud_points[:, 0],
            y=point_cloud_points[:, 1],
            z=point_cloud_points[:, 2],
            mode="markers",
            marker=dict(size=1.5, color="black"),
            name="Point cloud",
        )
    )
    fig.update_layout(
        title=dict(
            text=f"Grasp idx: {GRASP_IDX}, Object: {object_code}, Passed Eval: {passed_eval}"
        ),
    )
    VISUALIZE_HAND = True
    if VISUALIZE_HAND:
        for trace in hand_plotly:
            fig.add_trace(trace)
        for trace in hand_plotly_optimized:
            fig.add_trace(trace)
    fig.write_html("/home/albert/research/nerf_grasping/dex_diffuser_debug.html")


if __name__ == "__main__":
    main()
