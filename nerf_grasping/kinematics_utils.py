# %%
import pathlib
import pytorch_kinematics as pk
import pypose as pp
import torch

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
        chain: pk.chain.Chain = load_allegro(),
        batch_size: int = 1,
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
        return [self.wrist_pose @ fp for fp in fingertip_pyposes]


# %%
