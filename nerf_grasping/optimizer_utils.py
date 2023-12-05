# %%
from __future__ import annotations
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
from nerf_grasping.nerf_utils import (
    get_cameras,
    render,
)
from nerf_grasping.config.grasp_metric_config import GraspMetricConfig
from nerf_grasping.config.fingertip_config import UnionFingertipConfig
from nerf_grasping.config.camera_config import CameraConfig
from nerf_grasping.config.classifier_config import ClassifierConfig

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


def load_allegro(allegro_path: pathlib.Path = ALLEGRO_URDF_PATH) -> Chain:
    return pk.build_chain_from_urdf(open(allegro_path).read())


class AllegroHandConfig(torch.nn.Module):
    """
    A container specifying a batch of configurations for an Allegro hand, i.e., the
    wrist pose and the joint configurations.
    """

    def __init__(
        self,
        batch_size: int = 1,  # TODO(pculbert): refactor for arbitrary batch sizes.
        chain: Chain = load_allegro(),
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
        chain: Chain = load_allegro(),
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
    def from_hand_config_dict(cls, hand_config_dict: Dict[str, Any]):
        trans = torch.from_numpy(hand_config_dict["trans"]).float()
        rot = torch.from_numpy(hand_config_dict["rot"]).float()
        joint_angles = torch.from_numpy(hand_config_dict["joint_angles"]).float()
        batch_size = trans.shape[0]
        assert trans.shape == (batch_size, 3)
        assert rot.shape == (batch_size, 3, 3)
        assert joint_angles.shape == (batch_size, 16)

        wrist_translation = trans
        wrist_quat = pp.from_matrix(rot, pp.SO3_type)
        wrist_pose = pp.SE3(torch.cat([wrist_translation, wrist_quat], dim=1))

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

    def as_dict(self):
        """
        Returns a hand config dict
        """
        trans = self.wrist_pose.translation().detach().cpu().numpy()
        rot = self.wrist_pose.rotation().matrix().detach().cpu().numpy()
        joint_angles = self.joint_angles.detach().cpu().numpy()
        return {
            "trans": trans,
            "rot": rot,
            "joint_angles": joint_angles,
        }

    def as_tensor(self):
        """
        Returns a tensor of shape [batch_size, 23]
        with all config parameters.
        """
        return torch.cat((self.wrist_pose.tensor(), self.joint_angles), dim=-1)

    def mean(self):
        """
        Returns the mean of the batch of hand configs.
        A bit hacky -- just works in the Lie algebra, which
        is hopefully ok.
        """
        mean_joint_angles = self.joint_angles.mean(dim=0, keepdim=True)
        mean_wrist_pose = pp.se3(self.wrist_pose.Log().mean(dim=0, keepdim=True)).Exp()

        return AllegroHandConfig.from_values(
            wrist_pose=mean_wrist_pose,
            joint_angles=mean_joint_angles,
            chain=self.chain,
        )

    def cov(self):
        """
        Returns the covariance of the batch of hand configs.
        A bit hacky -- just works in the Lie algebra, which
        is hopefully ok.

        Returns a tuple of covariance tensors for the wrist pose and joint angles.
        """
        cov_wrist_pose = batch_cov(
            self.wrist_pose.Log(), dim=0
        )  # Leave in tangent space.
        cov_joint_angles = batch_cov(self.joint_angles, dim=0)

        return (cov_wrist_pose, cov_joint_angles)


class AllegroGraspConfig(torch.nn.Module):
    """Container defining a batch of grasps -- both pre-grasps
    and grasping directions -- for use in grasp optimization."""

    def __init__(
        self,
        batch_size: int = 1,
        chain: Chain = load_allegro(),
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
    def from_grasp_config_dict(
        cls,
        grasp_config_dict: Dict[str, Any],
        num_fingers: int = 4,
    ):
        """
        Factory method get grasp configs from grasp config_dict
        """
        # Load grasp data + instantiate correctly-sized config object.
        batch_size = grasp_config_dict["trans"].shape[0]
        grasp_config = cls(batch_size, num_fingers=num_fingers)
        device = grasp_config.grasp_orientations.device
        dtype = grasp_config.grasp_orientations.dtype

        # Load hand config
        grasp_config.hand_config = AllegroHandConfig.from_hand_config_dict(
            grasp_config_dict
        )

        grasp_orientations = (
            torch.from_numpy(grasp_config_dict["grasp_orientations"])
            .to(device)
            .to(dtype)
        )
        assert grasp_orientations.shape == (batch_size, num_fingers, 3, 3)

        # Set the grasp config's data.
        grasp_config.set_grasp_orientations(
            # Set atol and rtol to be a bit larger than default to handle large matrices
            # (numerical errors larger affect the sanity checking)
            pp.from_matrix(grasp_orientations, pp.SO3_type, atol=1e-4, rtol=1e-4)
        )

        return grasp_config

    def as_dict(self):
        hand_config_dict = self.hand_config.as_dict()
        hand_config_dict_batch_size = hand_config_dict["trans"].shape[0]
        assert (
            hand_config_dict_batch_size == self.batch_size
        ), f"Batch size {self.batch_size} does not match hand_config_dict_batch_size of {hand_config_dict_batch_size}"

        hand_config_dict["grasp_orientations"] = (
            self.grasp_orientations.matrix().detach().cpu().numpy()
        )
        return hand_config_dict

    def as_tensor(self):
        """
        Returns a tensor of shape [batch_size, num_fingers, 7 + 16 + 4]
        with all config parameters.
        """
        return torch.cat(
            (
                self.hand_config.as_tensor()
                .unsqueeze(-2)
                .expand(-1, self.num_fingers, -1),
                self.grasp_orientations.tensor(),
            ),
            dim=-1,
        )

    def mean(self):
        """
        Returns the mean of the batch of grasp configs.
        """
        mean_hand_config = self.hand_config.mean()
        mean_grasp_orientations = pp.so3(
            self.grasp_orientations.Log().mean(dim=0, keepdim=True)
        ).Exp()

        return AllegroGraspConfig.from_values(
            wrist_pose=mean_hand_config.wrist_pose,
            joint_angles=mean_hand_config.joint_angles,
            grasp_orientations=mean_grasp_orientations,
        )

    def cov(self):
        """
        Returns the covariance of the batch of grasp configs.
        """
        cov_wrist_pose, cov_joint_angles = self.hand_config.cov()
        cov_grasp_orientations = batch_cov(self.grasp_orientations.Log(), dim=0)

        return (
            cov_wrist_pose,
            cov_joint_angles,
            cov_grasp_orientations,
        )

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
        nerf_field: Field,
        classifier_model: Classifier,
        fingertip_config: UnionFingertipConfig,
        return_type: str = "failure_probability",
    ):
        super().__init__()
        self.nerf_field = nerf_field
        self.classifier_model = classifier_model
        self.fingertip_config = fingertip_config
        self.ray_origins_finger_frame = grasp_utils.get_ray_origins_finger_frame(
            fingertip_config
        )
        self.return_type = return_type

    def forward(
        self,
        grasp_config: AllegroGraspConfig,
    ):
        # Generate RaySamples.
        ray_samples = grasp_utils.get_ray_samples(
            self.ray_origins_finger_frame,
            grasp_config.grasp_frame_transforms,
            self.fingertip_config,
        )

        # Query NeRF at RaySamples.
        densities = self.nerf_field.get_density(ray_samples.to("cuda"))[0][
            ..., 0
        ]  # Shape [B, 4, n_x, n_y, n_z]

        assert densities.shape == (
            grasp_config.batch_size,
            4,
            self.fingertip_config.num_pts_x,
            self.fingertip_config.num_pts_y,
            self.fingertip_config.num_pts_z,
        )

        batch_data_input = BatchDataInput(
            nerf_densities=densities,
            grasp_transforms=grasp_config.grasp_frame_transforms,
            fingertip_config=self.fingertip_config,
            grasp_configs=grasp_config.as_tensor(),
        )

        # Pass grasp transforms, densities into classifier.
        if self.return_type == "failure_probability":
            return self.classifier_model.get_failure_probability(batch_data_input)
        elif self.return_type == "failure_logits":
            return self.classifier_model(batch_data_input)[:, -1]

    def get_failure_probability(
        self,
        grasp_config: AllegroGraspConfig,
    ):
        return self(grasp_config)


    @classmethod
    def from_config(
        cls,
        GraspMetricConfig: GraspMetricConfig,
    ) -> GraspMetric:
        return cls.from_configs(
            GraspMetricConfig.nerf_checkpoint_path,
            GraspMetricConfig.classifier_config,
            GraspMetricConfig.classifier_checkpoint,
        )

    @classmethod
    def from_configs(
        cls,
        nerf_config: pathlib.Path,
        classifier_config: ClassifierConfig,
        classifier_checkpoint: int = -1,
    ) -> GraspMetric:
        assert not isinstance(classifier_config.nerfdata_config, DepthImageNerfDataConfig), f"classifier_config.nerfdata_config must not be a DepthImageNerfDataConfig, but is {classifier_config.nerfdata_config}

        # Load nerf
        nerf_field = grasp_utils.load_nerf_field(nerf_config)

        # Load classifier (should device thing be here? probably since saved on gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classifier = (
            classifier_config.model_config.get_classifier_from_fingertip_config(
                fingertip_config=classifier_config.nerfdata_config.fingertip_config,
                n_tasks=classifier_config.task_type.n_tasks,
            )
        ).to(device)

        # Load classifier weights
        assert (
            classifier_config.checkpoint_workspace.output_dir.exists()
        ), f"checkpoint_workspace.output_dir does not exist at {classifier_config.checkpoint_workspace.output_dir}"
        print(
            f"Loading checkpoint ({classifier_config.checkpoint_workspace.output_dir})..."
        )

        input_checkpoint_paths = (
            classifier_config.checkpoint_workspace.input_checkpoint_paths
        )
        assert (
            len(input_checkpoint_paths) > 0
        ), f"No checkpoints found in {classifier_config.checkpoint_workspace.input_checkpoint_paths}"
        assert classifier_checkpoint < len(
            input_checkpoint_paths
        ), f"Requested checkpoint {classifier_checkpoint} does not exist in {classifier_config.checkpoint_workspace.input_checkpoint_paths}"
        checkpoint_path = input_checkpoint_paths[classifier_checkpoint]

        checkpoint = torch.load(checkpoint_path)
        classifier.load_state_dict(checkpoint["classifier"])
        classifier.load_state_dict(torch.load(checkpoint_path)["classifier"])

        if not isinstance(classifier, Simple_CNN_LSTM_Classifier):
            classifier.eval()  # weird LSTM thing where cudnn hasn't implemented the backwards pass in eval (??)

        return cls(nerf_field, classifier, classifier_config.nerfdata_config.fingertip_config)


class DepthImageGraspMetric(torch.nn.Module):
    """
    Wrapper for NeRF + grasp classifier to evaluate
    a particular AllegroGraspConfig.
    """

    def __init__(
        self,
        nerf_model: Model,
        classifier_model: DepthImageClassifier,
        camera_config: CameraConfig,
        return_type: str = "failure_probability",
    ):
        super().__init__()
        self.nerf_model = nerf_model
        self.classifier_model = classifier_model
        self.camera_config = camera_config
        self.return_type = return_type

    def forward(
        self,
        grasp_config: AllegroGraspConfig,
    ):
        cameras = get_cameras(grasp_config.grasp_frame_transforms, self.camera_config)
        batch_size = cameras.shape[0]

        depth, uncertainty = render(
            cameras, self.nerf_model, "median", far_plane=0.15
        )
        assert (
            depth.shape
            == uncertainty.shape
            == (
                self.camera_config.H,
                self.camera_config.W,
                batch_size * grasp_config.num_fingers,
            )
        )

        depth = depth.permute(2, 0, 1).reshape(
            batch_size,
            grasp_config.num_fingers,
            self.camera_config.H,
            self.camera_config.W,
        )
        uncertainty = uncertainty.permute(2, 0, 1).reshape(
            batch_size,
            grasp_config.num_fingers,
            self.camera_config.H,
            self.camera_config.W,
        )

        depth_uncertainty_images = torch.stack(
                [
                    depth, uncertainty
                ],
                dim=-3,
            )
        assert depth_uncertainty_images.shape == (
            batch_size,
            grasp_config.num_fingers,
            2,
            self.camera_config.H,
            self.camera_config.W,
        )

        batch_data_input = DepthBatchDataInput(
            depth_uncertainty_images=depth_uncertainty_images,
            grasp_transforms=grasp_config.grasp_frame_transforms,
            fingertip_config=self.fingertip_config,
            grasp_configs=grasp_config.as_tensor(),
        )

        # Pass grasp transforms, densities into classifier.
        if self.return_type == "failure_probability":
            return self.classifier_model.get_failure_probability(batch_data_input)
        elif self.return_type == "failure_logits":
            return self.classifier_model(batch_data_input)[:, -1]

    def get_failure_probability(
        self,
        grasp_config: AllegroGraspConfig,
    ):
        return self(grasp_config)


    @classmethod
    def from_config(
        cls,
        GraspMetricConfig: GraspMetricConfig,
    ) -> GraspMetric:
        return cls.from_configs(
            GraspMetricConfig.nerf_checkpoint_path,
            GraspMetricConfig.classifier_config,
            GraspMetricConfig.classifier_checkpoint,
        )

    @classmethod
    def from_configs(
        cls,
        nerf_config: pathlib.Path,
        classifier_config: ClassifierConfig,
        classifier_checkpoint: int = -1,
    ) -> GraspMetric:

        assert isinstance(classifier_config.nerfdata_config, DepthImageNerfDataConfig), f"classifier_config.nerfdata_config must be a DepthImageNerfDataConfig, but is {classifier_config.nerfdata_config}

        # Load nerf
        nerf_model = grasp_utils.load_nerf_model(nerf_config)

        # Load classifier (should device thing be here? probably since saved on gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classifier: DepthImageClassifier = (
            classifier_config.model_config.get_classifier_from_camera_config(
                camera_config=classifier_config.nerfdata_config.fingertip_camera_config,
                n_tasks=classifier_config.task_type.n_tasks,
            ).to(device)
        )

        # Load classifier weights
        assert (
            classifier_config.checkpoint_workspace.output_dir.exists()
        ), f"checkpoint_workspace.output_dir does not exist at {classifier_config.checkpoint_workspace.output_dir}"
        print(
            f"Loading checkpoint ({classifier_config.checkpoint_workspace.output_dir})..."
        )

        input_checkpoint_paths = (
            classifier_config.checkpoint_workspace.input_checkpoint_paths
        )
        assert (
            len(input_checkpoint_paths) > 0
        ), f"No checkpoints found in {classifier_config.checkpoint_workspace.input_checkpoint_paths}"
        assert classifier_checkpoint < len(
            input_checkpoint_paths
        ), f"Requested checkpoint {classifier_checkpoint} does not exist in {classifier_config.checkpoint_workspace.input_checkpoint_paths}"
        checkpoint_path = input_checkpoint_paths[classifier_checkpoint]

        checkpoint = torch.load(checkpoint_path)
        classifier.load_state_dict(checkpoint["classifier"])
        classifier.load_state_dict(torch.load(checkpoint_path)["classifier"])

        if not isinstance(classifier, Simple_CNN_LSTM_Classifier):
            classifier.eval()  # weird LSTM thing where cudnn hasn't implemented the backwards pass in eval (??)

        return cls(nerf_model, classifier, classifier_config.nerfdata_config.fingertip_config)




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


def batch_cov(x: torch.Tensor, dim: int = 0, keepdim=False):
    n_dim = x.shape[dim]
    x_mean = x.mean(dim, keepdim=True)
    x_centered = x - x_mean

    return (x_centered.unsqueeze(-2) * x_centered.unsqueeze(-1)).sum(
        dim=dim, keepdim=keepdim
    ) / (n_dim - 1)


# %%
