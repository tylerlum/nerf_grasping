# %%
import numpy as np
import pathlib
import pytorch_kinematics as pk
import pypose as pp
import torch

import nerf_grasping
from nerf_grasping import grasp_utils

from typing import List, Dict, Any

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


class AllegroGraspConfig(torch.nn.Module):
    """Container defining a batch of grasps -- both pre-grasps
    and grasping directions -- for use in grasp optimization."""

    def __init__(
        self,
        batch_size: int = 1,
        chain: pk.chain.Chain = load_allegro(),
        requires_grad: bool = True,
    ):
        # TODO(pculbert): refactor for arbitrary batch sizes.
        # TODO(pculbert): add device/dtype kwargs.

        self.batch_size = batch_size
        super().__init__()
        self.hand_config = AllegroHandConfig(batch_size, chain, requires_grad)

        # NOTE: grasp orientations has a batch dim for fingers,
        # since we choose one grasp dir / finger.
        self.grasp_orientations = pp.Parameter(
            pp.identity_SO3(batch_size, grasp_utils.NUM_FINGERS),
            requires_grad=requires_grad,
        )

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
        assert grasp_orientations.shape == (batch_size, grasp_utils.NUM_FINGERS, 4)

        grasp_config = cls(batch_size).to(
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
                grasp_utils.NUM_FINGERS,
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
    def from_grasp_data(cls, grasp_data_path: pathlib.Path, batch_size: int = 1):
        """
        Factory method to randomly sample a batch of grasp configs
        from a grasp data file.
        """
        # Load grasp data + instantiate correctly-sized config object.
        grasp_data = np.load(str(grasp_data_path), allow_pickle=True)
        grasp_config = cls(batch_size)

        # Sample (with replacement) random indices into grasp data.
        RANDOMIZE = False
        if RANDOMIZE:
            indices = np.random.choice(np.arange(len(grasp_data)), batch_size)
        else:
            indices = np.arange(batch_size)
        grasp_data = grasp_data[indices]

        # Assemble these samples into the data we need for the grasp config.
        grasp_data_tuples = [
            grasp_utils.get_grasp_config_from_grasp_data(gd) for gd in grasp_data
        ]

        # List of tuples -> tuple of lists.
        grasp_data_list = list(zip(*grasp_data_tuples))

        # Set the grasp config's data.
        state_dict = {}
        state_dict["hand_config.wrist_pose"] = torch.stack(
            grasp_data_list[0], dim=0
        ).to(
            device=grasp_config.grasp_orientations.device,
            dtype=grasp_config.grasp_orientations.dtype,
        )

        state_dict["hand_config.joint_angles"] = torch.stack(
            grasp_data_list[1], dim=0
        ).to(
            device=grasp_config.grasp_orientations.device,
            dtype=grasp_config.grasp_orientations.dtype,
        )

        state_dict["grasp_orientations"] = torch.stack(grasp_data_list[2], dim=0).to(
            device=grasp_config.grasp_orientations.device,
            dtype=grasp_config.grasp_orientations.dtype,
        )

        # Load state dict for module.
        grasp_config.load_state_dict(state_dict)

        return grasp_config

    def to_dexgraspnet_dicts(self) -> List[Dict[str, Any]]:
        qpos_list = grasp_utils.get_dexgraspnet_qpos_list(
            joint_angles=self.joint_angles.detach(),
            rotation_matrix=self.wrist_pose.rotation().matrix().detach(),
            translation=self.wrist_pose.translation().detach(),
        )
        grasp_dirs = self.grasp_dirs
        assert len(qpos_list) == self.batch_size
        n_fingers = 4
        assert grasp_dirs.shape == (self.batch_size, n_fingers, 3)

        dexgraspnet_dicts = []
        for i in range(self.batch_size):
            dexgraspnet_dict = {}
            dexgraspnet_dict["qpos"] = qpos_list[i]
            dexgraspnet_dict["grasp_dirs"] = grasp_dirs[i].tolist()
        return dexgraspnet_dicts

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

        return self.fingertip_transforms @ pp.from_matrix(
            self.grasp_orientations.matrix(), pp.SE3_type
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

        # TODO(pculbert): I think this doubles up some work since we've already computed
        # the sample positions -- check this isn't a big performance hit.

        # Pass grasp transforms, densities into classifier.
        return self.classifier_model(densities, grasp_config.grasp_frame_transforms)


def dry_run():
    # Some semi-hardcoded unit tests to make sure the code runs.

    nerf_configs = grasp_utils.get_nerf_configs(
        nerf_grasping.get_repo_root() + "/nerfcheckpoints"
    )
    nerf_model = grasp_utils.load_nerf(nerf_configs[0])

    batch_size = 32
    grasp_config = AllegroGraspConfig(batch_size=batch_size)
    classifier = lambda x, y: torch.zeros(batch_size)

    grasp_metric = GraspMetric(nerf_model, classifier)

    grasp_metric(grasp_config)


# %%
