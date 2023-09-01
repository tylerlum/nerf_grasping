# %%
import numpy as np
import pathlib
import pytorch_kinematics as pk
import pypose as pp
import torch

import nerf_grasping
from nerf_grasping import grasp_utils

from typing import List, Tuple, Dict, Any, Iterable, Union
import nerfstudio
from nerf_grasping.classifier import Classifier
from nerf_grasping.learned_metric.DexGraspNet_batch_data import BatchDataInput
from nerf_grasping.config.fingertip_config import UnionFingertipConfig

ALLEGRO_URDF_PATH = list(
    pathlib.Path(nerf_grasping.get_package_root()).rglob(
        "*allegro_hand_description_right.urdf"
    )
)[0]

Z_AXIS = torch.tensor([0, 0, 1], dtype=torch.float32)

FINGERTIP_LINK_NAMES = [
    "link_3.0_tip",
    "link_7.0_tip",
    "link_11.0_tip",
    "link_15.0_tip",
]


def load_allegro(allegro_path: pathlib.Path = ALLEGRO_URDF_PATH) -> pk.chain.Chain:
    return pk.build_chain_from_urdf(open(allegro_path).read())


class AllegroHandConfig(torch.nn.Module):
    """
    A container specifying a batch of configurations for an Allegro hand, i.e., the
    wrist pose and the joint configurations.
    """

    def __init__(
        self,
        batch_size: int = 1,  # TODO(pculbert): refactor for arbitrary batch sizes.
        chain: pk.chain.Chain = load_allegro(),
        requires_grad: bool = True,
    ):
        # TODO(pculbert): add device/dtype kwargs.
        super().__init__()
        self.chain = chain
        self.wrist_pose = pp.Parameter(
            pp.randn_SE3(batch_size), requires_grad=requires_grad
        )
        self.joint_angles = torch.nn.Parameter(
            torch.zeros(batch_size, 16), requires_grad=requires_grad
        )
        self.batch_size = batch_size

    @classmethod
    def from_values(
        cls,
        wrist_pose: pp.LieTensor,
        joint_angles: torch.Tensor,
        chain: pk.chain.Chain = load_allegro(),
        requires_grad: bool = True,
    ):
        """
        Factory method to create an AllegroHandConfig from a wrist pose and joint angles.
        """
        batch_size = wrist_pose.shape[0]
        assert wrist_pose.shape == (batch_size, 7)
        assert joint_angles.shape == (batch_size, 16)

        hand_config = cls(batch_size, chain, requires_grad).to(
            device=wrist_pose.device, dtype=wrist_pose.dtype
        )
        hand_config.set_wrist_pose(wrist_pose)
        hand_config.set_joint_angles(joint_angles)
        return hand_config

    @classmethod
    def from_hand_config_dicts(cls, hand_config_dicts: List[Dict[str, Any]]):
        # Assemble these samples into the data we need for the grasp config.
        hand_data_tuples = [
            grasp_utils.get_hand_config_from_hand_config_dict(x)
            for x in hand_config_dicts
        ]

        # List of tuples -> tuple of lists.
        wrist_pose, joint_angles = list(zip(*hand_data_tuples))
        wrist_pose = torch.stack(wrist_pose, dim=0).float()
        joint_angles = torch.stack(joint_angles, dim=0).float()
        return cls.from_values(wrist_pose=wrist_pose, joint_angles=joint_angles)

    def set_wrist_pose(self, wrist_pose: pp.LieTensor):
        assert (
            wrist_pose.shape == self.wrist_pose.shape
        ), f"New wrist pose, shape {wrist_pose.shape} does not match current wrist pose shape {self.wrist_pose.shape}"
        self.wrist_pose.data = wrist_pose.data.clone()

    def set_joint_angles(self, joint_angles: torch.Tensor):
        assert (
            joint_angles.shape == self.joint_angles.shape
        ), f"New hand config, shape {joint_angles.shape}, does not match shape of current hand config, {self.joint_angles.shape}."
        self.joint_angles.data = joint_angles

    def get_fingertip_transforms(self) -> pp.LieTensor:
        # Pretty hacky -- need to cast chain to the same device as the wrist pose.
        self.chain = self.chain.to(device=self.wrist_pose.device)

        # Run batched FK from current hand config.
        link_poses_hand_frame = self.chain.forward_kinematics(self.joint_angles)

        # Pull out fingertip poses + cast to PyPose.
        fingertip_poses = [link_poses_hand_frame[ln] for ln in FINGERTIP_LINK_NAMES]
        fingertip_pyposes = [
            pp.from_matrix(fp.get_matrix(), pp.SE3_type) for fp in fingertip_poses
        ]

        # Apply wrist transformation to get world-frame fingertip poses.
        return torch.stack(
            [self.wrist_pose @ fp for fp in fingertip_pyposes], dim=1
        )  # shape [B, batch_size, 7]

    def as_dicts(self):
        """
        Returns a list of dicts, one for each sample in the batch.
        """
        return [
            grasp_utils.get_hand_config_dict_from_hand_config(ww, jj)
            for ww, jj in zip(self.wrist_pose, self.joint_angles)
        ]


class AllegroGraspConfig(torch.nn.Module):
    """Container defining a batch of grasps -- both pre-grasps
    and grasping directions -- for use in grasp optimization."""

    def __init__(
        self,
        batch_size: int = 1,
        chain: pk.chain.Chain = load_allegro(),
        requires_grad: bool = True,
        num_fingers: int = 4,
    ):
        # TODO(pculbert): refactor for arbitrary batch sizes.
        # TODO(pculbert): add device/dtype kwargs.

        self.batch_size = batch_size
        super().__init__()
        self.hand_config = AllegroHandConfig(batch_size, chain, requires_grad)

        # NOTE: grasp orientations has a batch dim for fingers,
        # since we choose one grasp dir / finger.
        # grasp_orientations refers to the orientation of each finger in world frame
        # (i.e. the third column of grasp_orientations rotation matrix is the finger approach direction in world frame)
        self.grasp_orientations = pp.Parameter(
            pp.identity_SO3(batch_size, num_fingers),
            requires_grad=requires_grad,
        )
        self.num_fingers = num_fingers

    @classmethod
    def from_path(cls, path: pathlib.Path):
        """
        Factory method to create an AllegroGraspConfig from a path to a saved state dict.
        """
        state_dict = torch.load(str(path))
        batch_size = state_dict["hand_config.wrist_pose"].shape[0]
        grasp_config = cls(batch_size)
        grasp_config.load_state_dict(state_dict)
        return grasp_config

    @classmethod
    def from_values(
        cls,
        wrist_pose: pp.LieTensor,
        joint_angles: torch.Tensor,
        grasp_orientations: pp.LieTensor,
        num_fingers: int = 4,
    ):
        """
        Factory method to create an AllegroGraspConfig from values
        for the wrist pose, joint angles, and grasp orientations.
        """
        batch_size = wrist_pose.shape[0]
        # TODO (pculbert): refactor for arbitrary batch sizes via lshape.

        # Check shapes.
        assert joint_angles.shape == (batch_size, 16)
        assert wrist_pose.shape == (batch_size, 7)
        assert grasp_orientations.shape == (batch_size, num_fingers, 4)

        grasp_config = cls(batch_size, num_fingers=num_fingers).to(
            device=wrist_pose.device, dtype=wrist_pose.dtype
        )
        grasp_config.hand_config.set_wrist_pose(wrist_pose)
        grasp_config.hand_config.set_joint_angles(joint_angles)
        grasp_config.set_grasp_orientations(grasp_orientations)
        return grasp_config

    @classmethod
    def randn(
        cls,
        batch_size: int = 1,
        std_orientation: float = 0.1,
        std_wrist_pose: float = 0.1,
        std_joint_angles: float = 0.1,
        num_fingers: int = 4,
    ):
        """
        Factory method to create a random AllegroGraspConfig.
        """
        grasp_config = cls(batch_size)

        # TODO(pculbert): think about setting a mean pose that's
        # reasonable, tune the default stds.

        grasp_orientations = pp.so3(
            std_orientation
            * torch.randn(
                batch_size,
                num_fingers,
                3,
                device=grasp_config.grasp_orientations.device,
                dtype=grasp_config.grasp_orientations.dtype,
            )
        ).Exp()

        wrist_pose = pp.se3(
            std_wrist_pose
            * torch.randn(
                batch_size,
                6,
                dtype=grasp_config.grasp_orientations.dtype,
                device=grasp_config.grasp_orientations.device,
            )
        ).Exp()

        joint_angles = std_joint_angles * torch.randn(
            batch_size,
            16,
            dtype=grasp_config.grasp_orientations.dtype,
            device=grasp_config.grasp_orientations.device,
        )

        return grasp_config.from_values(wrist_pose, joint_angles, grasp_orientations)

    @classmethod
    def from_grasp_config_dicts(
        cls,
        grasp_config_dicts: List[Dict[str, Any]],
        num_fingers: int = 4,
    ):
        """
        Factory method get grasp configs from a grasp config_dicts
        """
        # Load grasp data + instantiate correctly-sized config object.
        batch_size = len(grasp_config_dicts)
        grasp_config = cls(batch_size, num_fingers=num_fingers)
        device = grasp_config.grasp_orientations.device
        dtype = grasp_config.grasp_orientations.dtype

        # Load hand config
        grasp_config.hand_config = AllegroHandConfig.from_hand_config_dicts(
            grasp_config_dicts
        )

        grasp_orientations_list = []
        for i in range(batch_size):
            grasp_config_dict = grasp_config_dicts[i]
            grasp_orientations = torch.tensor(
                grasp_config_dict["grasp_orientations"], device=device, dtype=dtype
            )
            assert grasp_orientations.shape == (grasp_config.num_fingers, 3, 3)
            grasp_orientations = pp.from_matrix(grasp_orientations, pp.SO3_type)
            grasp_orientations_list.append(grasp_orientations)
        grasp_orientations_list = torch.stack(grasp_orientations_list, dim=0)

        # Set the grasp config's data.
        grasp_config.set_grasp_orientations(grasp_orientations_list)

        return grasp_config

    def as_dicts(self):
        dict_list = self.hand_config.as_dicts()
        assert (
            len(dict_list) == self.batch_size
        ), f"Batch size {self.batch_size} does not match length of hand dict list {len(dict_list)}"

        for i in range(self.batch_size):
            dict_list[i]["grasp_orientations"] = (
                self.grasp_orientations[i].matrix().detach().cpu().numpy()
            )

        return dict_list

    def set_grasp_orientations(self, grasp_orientations: pp.LieTensor):
        assert (
            grasp_orientations.shape == self.grasp_orientations.shape
        ), f"New grasp orientations, shape {grasp_orientations.shape}, do not match current grasp orientations shape {self.grasp_orientations.shape}"
        self.grasp_orientations.data = grasp_orientations.data.clone()

    def __getitem__(self, idxs):
        """
        Enables indexing/slicing into a batch of grasp configs.
        """
        return type(self).from_values(
            self.wrist_pose[idxs],
            self.joint_angles[idxs],
            self.grasp_orientations[idxs],
        )

    @property
    def wrist_pose(self) -> pp.LieTensor:
        return self.hand_config.wrist_pose

    @property
    def joint_angles(self) -> torch.Tensor:
        return self.hand_config.joint_angles

    @property
    def fingertip_transforms(self) -> pp.LieTensor:
        """Returns finger-to-world transforms."""
        return self.hand_config.get_fingertip_transforms()

    @property
    def grasp_frame_transforms(self) -> pp.LieTensor:
        """Returns SE(3) transforms for ``grasp frame'', i.e.,
        z-axis pointing along grasp direction."""
        fingertip_positions = self.fingertip_transforms.translation()
        assert fingertip_positions.shape == (
            self.batch_size,
            self.num_fingers,
            3,
        )

        grasp_orientations = self.grasp_orientations
        assert grasp_orientations.lshape == (self.batch_size, self.num_fingers)

        transforms = pp.SE3(
            torch.cat(
                [
                    fingertip_positions,
                    grasp_orientations,
                ],
                dim=-1,
            )
        )
        assert transforms.lshape == (self.batch_size, self.num_fingers)
        return transforms

    @property
    def grasp_dirs(self) -> torch.Tensor:  # shape [B, 4, 3].
        return self.grasp_frame_transforms.rotation() @ Z_AXIS.to(
            device=self.grasp_orientations.device, dtype=self.grasp_orientations.dtype
        ).unsqueeze(0).unsqueeze(0)


class GraspMetric(torch.nn.Module):
    """
    Wrapper for NeRF + grasp classifier to evaluate
    a particular AllegroGraspConfig.
    """

    def __init__(
        self,
        nerf_model: nerfstudio.models.base_model.Model,
        classifier_model: Classifier,
        fingertip_config: UnionFingertipConfig,
    ):
        super().__init__()
        self.nerf_model = nerf_model
        self.classifier_model = classifier_model
        self.fingertip_config = fingertip_config
        self.ray_origins_finger_frame = grasp_utils.get_ray_origins_finger_frame(
            fingertip_config
        )

    def forward(self, grasp_config: AllegroGraspConfig):
        # Generate RaySamples.
        ray_samples = grasp_utils.get_ray_samples(
            self.ray_origins_finger_frame, grasp_config.grasp_frame_transforms
        )

        # Query NeRF at RaySamples.
        densities = self.nerf_model.get_density(ray_samples.to("cuda"))[0][
            ..., 0
        ]  # Shape [B, 4, n_x, n_y, n_z]

        assert densities.shape == (
            grasp_config.batch_size,
            4,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )

        # TODO(pculbert): I think this doubles up some work since we've already computed
        # the sample positions -- check this isn't a big performance hit.
        batch_data_input = BatchDataInput(
            nerf_densities=densities,
            grasp_transforms=grasp_config.grasp_frame_transforms,
        )

        # Pass grasp transforms, densities into classifier.
        return self.classifier_model.get_failure_probability(batch_data_input)

    def get_failure_probability(self, grasp_config: AllegroGraspConfig):
        return self(grasp_config)


class IndexingDataset(torch.utils.data.Dataset):
    def __init__(self, num_datapoints: int):
        self.num_datapoints = num_datapoints

    def __getitem__(self, idx):
        return idx

    def __len__(self):
        return self.num_datapoints


def get_split_inds(
    num_datapoints: int, split: Iterable[Union[int, float]], random_seed: int
):
    indexing_dataset = IndexingDataset(num_datapoints)
    splits = torch.utils.data.random_split(
        indexing_dataset, split, generator=torch.Generator().manual_seed(random_seed)
    )

    return [split.indices for split in splits]


def SO3_to_SE3(R: pp.LieTensor):
    assert R.ltype == pp.SO3_type, f"R must be an SO3, not {R.ltype}"

    return pp.SE3(torch.cat((torch.zeros_like(R[..., :3]), R.tensor()), dim=-1))


# %%
