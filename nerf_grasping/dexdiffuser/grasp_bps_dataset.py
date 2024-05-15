from typing import Tuple
import torch
from torch.utils import data
import h5py


class NaiveGraspBPSDataset(data.Dataset):
    def __init__(self, grasps: torch.Tensor, bpss: torch.Tensor) -> None:
        self.grasps = grasps
        self.bpss = bpss

    def __len__(self) -> int:
        return len(self.grasps)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.grasps[idx], self.bpss[idx]


class GraspBPSDataset(data.Dataset):
    def __init__(
        self,
        input_hdf5_filepath: str,
    ) -> None:
        self.input_hdf5_filepath = input_hdf5_filepath
        with h5py.File(self.input_hdf5_filepath, "r") as hdf5_file:
            # Essentials
            self.grasps = torch.from_numpy(hdf5_file["/grasps"][()]).float()
            self.bpss = torch.from_numpy(hdf5_file["/bpss"][()]).float()
            self.grasp_bps_idxs = torch.from_numpy(hdf5_file["/grasp_bps_idx"][()])
            self.length = hdf5_file.attrs["num_grasps"]

            assert (
                self.grasps.shape[0] == self.length
            ), f"Expected {self.length} grasps, got {self.grasps.shape[0]}"
            assert (
                self.grasp_bps_idxs.shape[0] == self.length
            ), f"Expected {self.length} grasp_bps_idxs, got {self.grasp_bps_idxs.shape[0]}"

            # Extras
            self.basis_points = torch.from_numpy(hdf5_file["/basis_points"][()]).float()
            self.point_cloud_filepaths = hdf5_file["/point_cloud_filepath"][()]
            self.object_codes = hdf5_file["/object_code"][()]
            self.object_scales = hdf5_file["/object_scale"][()]
            self.object_states = torch.from_numpy(
                hdf5_file["/object_state"][()]
            ).float()
            n_basis_points = self.basis_points.shape[0]
            assert self.basis_points.shape == (
                n_basis_points,
                3,
            ), f"Expected shape ({n_basis_points}, 3), got {self.basis_points.shape}"
            assert (
                self.point_cloud_filepaths.shape[0] == self.bpss.shape[0]
            ), f"Expected {self.bpss.shape[0]} point_cloud_filepaths, got {self.point_cloud_filepaths.shape[0]}"
            assert (
                self.object_codes.shape[0] == self.bpss.shape[0]
            ), f"Expected {self.bpss.shape[0]} object_codes, got {self.object_codes.shape[0]}"
            assert (
                self.object_scales.shape[0] == self.bpss.shape[0]
            ), f"Expected {self.bpss.shape[0]} object_scales, got {self.object_scales.shape[0]}"
            assert (
                self.object_states.shape[0] == self.length
            ), f"Expected {self.length} object_states, got {self.object_states.shape[0]}"

    def __len__(self) -> int:
        return self.length

    ###### Extras ######
    def __getitem__(self, grasp_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        bps_idx = self.grasp_bps_idxs[grasp_idx]
        return self.grasps[grasp_idx], self.bpss[bps_idx]

    def get_basis_points(self) -> torch.Tensor:
        return self.basis_points.clone()

    def get_point_cloud_filepath(self, grasp_idx: int) -> str:
        bpss_idx = self.grasp_bps_idxs[grasp_idx]
        return self.point_cloud_filepaths[bpss_idx].decode("utf-8")

    def get_object_code(self, grasp_idx: int) -> str:
        bpss_idx = self.grasp_bps_idxs[grasp_idx]
        return self.object_codes[bpss_idx].decode("utf-8")

    def get_object_scale(self, grasp_idx: int) -> float:
        bpss_idx = self.grasp_bps_idxs[grasp_idx]
        return self.object_scales[bpss_idx]

    def get_object_state(self, grasp_idx: int) -> torch.Tensor:
        return self.object_states[grasp_idx]


def main() -> None:
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

    INPUT_HDF5_FILEPATH = "/juno/u/tylerlum/github_repos/nerf_grasping/data/2024-05-14_rotated_stable_grasps_bps/data.h5"
    GRASP_IDX = 77930
    MESHDATA_ROOT = (
        "/juno/u/tylerlum/github_repos/DexGraspNet/data/rotated_meshdata_stable"
    )

    print("\n" + "=" * 79)
    print(f"Reading dataset from {INPUT_HDF5_FILEPATH}")
    dataset = GraspBPSDataset(input_hdf5_filepath=INPUT_HDF5_FILEPATH)
    print("=" * 79)
    print(f"len(dataset): {len(dataset)}")

    print("\n" + "=" * 79)
    print(f"Getting grasp and bps for grasp_idx {GRASP_IDX}")
    print("=" * 79)
    grasp, bps = dataset[GRASP_IDX]
    print(f"grasp.shape: {grasp.shape}")
    print(f"bps.shape: {bps.shape}")

    print("\n" + "=" * 79)
    print("Getting debugging extras")
    print("=" * 79)
    basis_points = dataset.get_basis_points()
    object_code = dataset.get_object_code(GRASP_IDX)
    object_scale = dataset.get_object_scale(GRASP_IDX)
    object_state = dataset.get_object_state(GRASP_IDX)
    print(f"basis_points.shape: {basis_points.shape}")

    # Mesh
    mesh_path = pathlib.Path(f"{MESHDATA_ROOT}/{object_code}/coacd/decomposed.obj")
    assert mesh_path.exists(), f"{mesh_path} does not exist"
    print(f"Reading mesh from {mesh_path}")
    mesh = trimesh.load(mesh_path)

    xyz, quat_xyzw = object_state[:3], object_state[3:7]
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    transform = np.eye(4)
    transform[:3, :3] = transforms3d.quaternions.quat2mat(quat_wxyz)
    transform[:3, 3] = xyz
    mesh.apply_scale(object_scale)
    mesh.apply_transform(transform)

    # Point cloud
    point_cloud_filepath = dataset.get_point_cloud_filepath(GRASP_IDX)
    print(f"Reading point cloud from {point_cloud_filepath}")
    point_cloud = o3d.io.read_point_cloud(point_cloud_filepath)
    point_cloud, _ = point_cloud.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )
    point_cloud, _ = point_cloud.remove_radius_outlier(nb_points=16, radius=0.05)
    point_cloud_points = np.asarray(point_cloud.points)
    print(f"point_cloud_points.shape: {point_cloud_points.shape}")

    # Grasp
    assert grasp.shape == (3 + 6 + 16 + 4 * 3,), f"Expected shape (3 + 6 + 16 + 4 * 3), got {grasp.shape}"
    grasp = grasp.detach().cpu().numpy()
    grasp_trans, grasp_rot6d, grasp_joints, grasp_dirs = (
        grasp[:3],
        grasp[3:9],
        grasp[9:25],
        grasp[25:].reshape(4, 3),
    )
    grasp_rot = np.zeros((3, 3))
    grasp_rot[:3, :2] = grasp_rot6d.reshape(3, 2)
    assert np.dot(grasp_rot[:3, 0], grasp_rot[:3, 1]) < 1e-3, f"Expected dot product < 1e-3, got {np.dot(grasp_rot[:3, 0], grasp_rot[:3, 1])}"
    grasp_rot[:3, 2] = np.cross(grasp_rot[:3, 0], grasp_rot[:3, 1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hand_pose = hand_config_to_pose(grasp_trans[None], grasp_rot[None], grasp_joints[None]).to(device)
    hand_model_type = HandModelType.ALLEGRO_HAND
    grasp_orientations = np.zeros((4, 3, 3))
    grasp_orientations[:, :, 2] = grasp_dirs  # Leave the x-axis and y-axis as zeros, hacky but works
    hand_model = HandModel(hand_model_type=hand_model_type, device=device)
    hand_model.set_parameters(hand_pose)
    hand_plotly = hand_model.get_plotly_data(i=0)

    (
        optimized_joint_angle_targets,
        _,
    ) = compute_optimized_joint_angle_targets_given_grasp_orientations(
        joint_angles_start=hand_model.hand_pose[:, 9:],
        hand_model=hand_model,
        grasp_orientations=torch.from_numpy(grasp_orientations[None]).to(device),
    )
    new_hand_pose = hand_config_to_pose(grasp_trans[None], grasp_rot[None], optimized_joint_angle_targets.detach().cpu().numpy()).to(device)
    hand_model.set_parameters(new_hand_pose)
    hand_plotly_optimized = hand_model.get_plotly_data(i=0, opacity=0.5)



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
        title=dict(text=f"Grasp idx: {GRASP_IDX}, Object: {object_code}"),
    )
    for trace in hand_plotly:
        fig.add_trace(trace)
    for trace in hand_plotly_optimized:
        fig.add_trace(trace)
    fig.show()


if __name__ == "__main__":
    main()
