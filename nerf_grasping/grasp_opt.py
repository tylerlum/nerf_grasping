# %%
import pathlib
import pytorch_kinematics as pk
import pypose as pp
import torch

import nerf_grasping
from nerf_grasping import grasp_utils

from typing import List

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
        super().__init__()
        self.chain = chain
        self.wrist_pose = pp.Parameter(
            pp.randn_SE3(batch_size), requires_grad=requires_grad
        )
        self.joint_angles = torch.nn.Parameter(
            torch.zeros(batch_size, 16), requires_grad=requires_grad
        )
        self.batch_size = batch_size

    def to(self, device=None, dtype=None):
        super().to(device=device, dtype=dtype)
        self.chain.to(device=torch.device(device), dtype=dtype)

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

    def get_fingertip_transforms(self) -> List[pp.LieTensor]:
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


class AllegroGraspConfig(torch.nn.Module):
    """Container defining a batch of grasps -- both pre-grasps
    and grasping directions -- for use in grasp optimization."""

    def __init__(
        self,
        batch_size: int = 1,
        chain: pk.chain.Chain = load_allegro(),
        requires_grad: bool = True,
    ):
        self.batch_size = batch_size
        super().__init__()
        self.hand_config = AllegroHandConfig(batch_size, chain, requires_grad)
        self.grasp_orientations = pp.Parameter(
            pp.identity_SO3(batch_size), requires_grad=requires_grad
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

        return self.fingertip_transforms @ pp.from_matrix(
            self.grasp_orientations.unsqueeze(1).matrix(), pp.SE3_type
        )

    @property
    def grasp_dirs(self) -> torch.Tensor:  # shape [B, 4, 3].
        return pp.from_matrix(
            self.grasp_frame_transforms.matrix(), pp.SO3_type
        ) @ Z_AXIS.to(
            device=self.grasp_orientations.device, dtype=self.grasp_orientations.dtype
        ).unsqueeze(
            0
        ).unsqueeze(
            0
        )


class GraspMetric(torch.nn.Module):
    """
    Wrapper for NeRF + grasp classifier to evaluate
    a particular AllegroGraspConfig.
    """

    def __init__(self, nerf_model, classifier_model):
        super().__init__()
        self.nerf_model = nerf_model
        self.classifier_model = classifier_model
        self.ray_origins_finger_frame = grasp_utils.get_ray_origins_finger_frame()

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
            grasp_utils.NUM_PTS_X,
            grasp_utils.NUM_PTS_Y,
            grasp_utils.NUM_PTS_Z,
        )

        # TODO(pculbert): fix this to match the classifier trace.
        # Pass ray_samples.get_positions(), densities into classifier.
        return self.classifier_model(densities, ray_samples.frustums.get_positions())


def dry_run():
    # Some semi-hardcoded unit tests to make sure the code runs.

    nerf_configs = grasp_utils.get_nerf_configs(
        nerf_grasping.get_package_root() + "/../nerfcheckpoints"
    )
    nerf_model = grasp_utils.load_nerf(nerf_configs[0])

    batch_size = 32
    grasp_config = AllegroGraspConfig(batch_size=batch_size)
    classifier = lambda x, y: torch.zeros(batch_size)

    grasp_metric = GraspMetric(nerf_model, classifier)

    grasp_metric(grasp_config)


# %%
