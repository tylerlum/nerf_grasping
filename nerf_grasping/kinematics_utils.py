# %%
# TODO(pculbert): remove this.
import pathlib
import pytorch_kinematics as pk
import pypose as pp
import torch

from nerf_grasping import grasp_utils

from typing import List

ALLEGRO_URDF_PATH = list(
    pathlib.Path(".").rglob("*allegro_hand_description_right.urdf")
)[0]

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
        batch_size: int = 1,
        chain: pk.chain.Chain = load_allegro(),
        requires_grad: bool = True,
    ):
        super().__init__()
        self.chain = chain
        self.wrist_pose = pp.Parameter(
            pp.randn_SE3(batch_size), requires_grad=requires_grad
        )
        self.hand_config = torch.nn.Parameter(
            torch.zeros(batch_size, 16), requires_grad=requires_grad
        )

    def to(self, device=None, dtype=None):
        super().to(device=device, dtype=dtype)
        self.chain.to(device=torch.device(device), dtype=dtype)

    def set_wrist_pose(self, wrist_pose: pp.LieTensor):
        assert (
            wrist_pose.shape == self.wrist_pose.shape
        ), f"New wrist pose, shape {wrist_pose.shape} does not match current wrist pose shape {self.wrist_pose.shape}"
        self.wrist_pose.data = wrist_pose.data.clone()

    def set_hand_config(self, hand_config: torch.Tensor):
        assert (
            hand_config.shape == self.hand_config.shape
        ), f"New hand config, shape {hand_config.shape}, does not match shape of current hand config, {self.current_hand_config.shape}."
        self.hand_config.data = hand_config

    def get_fingertip_transforms(self) -> List[pp.LieTensor]:
        # Run batched FK from current hand config.
        link_poses_hand_frame = self.chain.forward_kinematics(self.hand_config)

        # Pull out fingertip poses + cast to PyPose.
        fingertip_poses = [link_poses_hand_frame[ln] for ln in FINGERTIP_LINK_NAMES]
        fingertip_pyposes = [
            pp.from_matrix(fp.get_matrix(), pp.SE3_type) for fp in fingertip_poses
        ]

        # Apply wrist transformation to get world-frame fingertip poses.
        return torch.stack(
            [self.wrist_pose @ fp for fp in fingertip_pyposes]
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
        super().__init__()
        self.hand_config = AllegroHandConfig(batch_size, chain, requires_grad)
        self.grasp_orientations = pp.LieParameter(
            pp.identity_SO3(batch_size), requires_grad=requires_grad
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
        # Generate fingertip transforms.
        fingertip_poses = grasp_config.hand_config.get_fingertip_transforms()

        # Rotate frame by grasp_direction transform.
        fingertip_poses = fingertip_poses @ grasp_config.grasp_orientations.unsqueeze(0)

        # Generate RaySamples.
        ray_samples = grasp_utils.get_ray_samples(
            self.ray_origins_finger_frame, fingertip_poses
        )

        # Query NeRF at RaySamples.
        densities = self.nerf_model.get_density(ray_samples.to("cuda")).reshape(
            4, -1, 3
        )

        # TODO(pculbert): fix this to match the classifier trace.
        # Pass ray_samples.get_positions(), densities into classifier.
        return self.classifier(densities, ray_samples.frustums.get_positions())


# %%
