from __future__ import annotations
from nerf_grasping.optimizer_utils import AllegroGraspConfig, GraspMetric
import pathlib
import grasp_utils
from nerf_grasping.grasp_utils import NUM_FINGERS
import torch
from nerf_grasping.classifier import Classifier
from typing import Tuple
import nerf_grasping
from functools import partial


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


if is_notebook():
    from tqdm.notebook import tqdm as std_tqdm
else:
    from tqdm import tqdm as std_tqdm

tqdm = partial(std_tqdm, dynamic_ncols=True)

# Hardcoding some values for now.
import functools
from localscope import localscope
import torch.nn as nn
from nerf_grasping.models.tyler_new_models import (
    ConvOutputTo1D,
    PoolType,
    conv_encoder,
    mlp,
)
from nerf_grasping.grasp_utils import (
    DIST_BTWN_PTS_MM,
    get_ray_samples,
    get_ray_origins_finger_frame,
    load_nerf,
    NUM_PTS_X,
    NUM_PTS_Y,
    NUM_PTS_Z,
    GRASP_DEPTH_MM,
    FINGER_WIDTH_MM,
    FINGER_HEIGHT_MM,
    NUM_FINGERS,
)

NUM_XYZ = 3


class CNN_3D_Classifier(nn.Module):
    # @localscope.mfc
    def __init__(self, input_shape: Tuple[int, int, int, int], n_fingers) -> None:
        # TODO: Make this not hardcoded
        super().__init__()
        self.input_shape = input_shape
        self.n_fingers = n_fingers

        self.conv = conv_encoder(
            input_shape=self.input_shape,
            conv_channels=[32, 64, 128],
            pool_type=PoolType.MAX,
            dropout_prob=0.1,
            conv_output_to_1d=ConvOutputTo1D.AVG_POOL_SPATIAL,
        )

        # Get conv output shape
        example_batch_size = 2
        example_input = torch.zeros(
            example_batch_size, self.n_fingers, *self.input_shape
        )
        example_input = example_input.reshape(
            example_batch_size * self.n_fingers, *self.input_shape
        )
        conv_output = self.conv(example_input)
        self.conv_output_dim = conv_output.shape[-1]
        assert conv_output.shape == (
            example_batch_size * self.n_fingers,
            self.conv_output_dim,
        )

        self.mlp = mlp(
            num_inputs=self.conv_output_dim * self.n_fingers,
            num_outputs=self.n_classes,
            hidden_layers=[256, 256],
        )

    @localscope.mfc
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        assert x.shape == (
            batch_size,
            self.n_fingers,
            *self.input_shape,
        ), f"{x.shape}"

        # Put n_fingers into batch dim
        x = x.reshape(batch_size * self.n_fingers, *self.input_shape)

        x = self.conv(x)
        assert x.shape == (
            batch_size * self.n_fingers,
            self.conv_output_dim,
        ), f"{x.shape}"
        x = x.reshape(batch_size, self.n_fingers, self.conv_output_dim)
        x = x.reshape(batch_size, self.n_fingers * self.conv_output_dim)

        x = self.mlp(x)
        assert x.shape == (batch_size, self.n_classes), f"{x.shape}"
        return x

    @localscope.mfc
    def get_success_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    @localscope.mfc
    def get_success_probability(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.softmax(self.get_success_logits(x), dim=-1)

    @property
    @functools.lru_cache
    def n_classes(self) -> int:
        return 2


class ClassifierWrapper(nn.Module):
    def __init__(self, classifier: CNN_3D_Classifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, densities: torch.Tensor, query_points: torch.Tensor):
        assert len(densities.shape) == 5, f"{densities.shape}"
        B, n_f, n_x, n_y, n_z = densities.shape

        assert query_points.shape == (B, n_f, n_x, n_y, n_z, 3), f"{query_points.shape}"

        cnn_inputs = torch.cat((densities.unsqueeze(-1), query_points), dim=-1)
        cnn_inputs = cnn_inputs.permute(0, 1, 5, 2, 3, 4)

        return self.classifier(cnn_inputs)  # Shape [B, 2]


class Optimizer:
    def __init__(self, init_grasps: AllegroGraspConfig, grasp_metric: GraspMetric):
        self.grasp_config = init_grasps
        self.grasp_metric = grasp_metric

    @classmethod
    def from_configs(
        cls,
        init_grasps: AllegroGraspConfig,
        nerf_config: pathlib.Path,
        classifier_config: pathlib.Path,
    ) -> Optimizer:
        nerf = grasp_utils.load_nerf(nerf_config)
        classifier = Classifier(classifier_config)
        grasp_metric = GraspMetric(nerf, classifier)
        return cls(init_grasps, grasp_metric)

    def optimize(self) -> Tuple[torch.tensor, AllegroGraspConfig]:
        pass


class AdamOptimizer(Optimizer):
    def __init__(
        self, init_grasps: AllegroGraspConfig, grasp_metric: GraspMetric, **kwargs
    ):
        """
        Constructor for AdamOptimizer.

        Args:
            init_grasps: Initial grasp configuration.
            grasp_metric: GraspMetric object defining the metric to optimize.
            **kwargs: Keyword arguments to pass to torch.optim.Adam.
        """
        super().__init__(init_grasps, grasp_metric)
        self.optimizer = torch.optim.SGD(self.grasp_config.parameters(), **kwargs)

    @classmethod
    def from_configs(
        cls,
        init_grasps: AllegroGraspConfig,
        nerf_config: pathlib.Path,
        classifier_config: pathlib.Path,
        **kwargs,
    ) -> AdamOptimizer:
        nerf = grasp_utils.load_nerf(nerf_config)

        # TODO(pculbert): BRITTLE! Support more classifiers etc.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        INPUT_SHAPE = (NUM_XYZ + 1, NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z)
        cnn = CNN_3D_Classifier(
            input_shape=INPUT_SHAPE,
            n_fingers=NUM_FINGERS,
        ).to(device)
        cnn.load_state_dict(
            torch.load(str(classifier_config))["nerf_to_grasp_success_model"]
        )
        cnn.eval()

        init_grasps = init_grasps.to(device=device)
        print(
            init_grasps.grasp_orientations.device,
            init_grasps.hand_config.wrist_pose.device,
            init_grasps.hand_config.joint_angles.device,
            init_grasps.hand_config.chain.device,
        )

        grasp_metric = GraspMetric(nerf, ClassifierWrapper(cnn))
        return cls(init_grasps, grasp_metric, **kwargs)

    def step(self):
        self.optimizer.zero_grad()
        loss = self.grasp_metric(self.grasp_config)
        assert loss.shape == (self.grasp_config.batch_size,)
        loss.mean().backward()
        self.optimizer.step()

    @property
    def grasp_scores(self) -> torch.tensor:
        return self.grasp_metric(self.grasp_config)


def run_optimizer_loop(
    optimizer: Optimizer, num_steps: int
) -> Tuple[torch.tensor, AllegroGraspConfig]:
    description = "Optimizing grasps"
    for iter in (
        pbar := tqdm(
            range(num_steps),
            desc=description,
        )
    ):
        optimizer.step()
        updated_description = (
            f"{description} | Loss: {optimizer.grasp_scores.mean():.3f}"
        )
        pbar.set_description(updated_description)
        # TODO(pculbert): Add logging for grasps and scores.
        # Likely want to log min/mean of scores, and store the grasp configs

        # TODO(pculbert): Track best grasps across steps.

    _, sort_indices = torch.sort(optimizer.grasp_scores, descending=True)
    return (optimizer.grasp_scores[sort_indices], optimizer.grasp_config[sort_indices])


def main() -> None:
    NERF_CONFIG = (
        pathlib.Path(nerf_grasping.get_repo_root())
        / "nerfcheckpoints/sem-Camera-7bff4fd4dc53de7496dece3f86cb5dd5_0_10/depth-nerfacto/2023-08-09_104724/config.yml"
    )
    CLASSIFIER_CHECKPOINT_PATH = (
        pathlib.Path(nerf_grasping.get_repo_root())
        / "Train_DexGraspNet_NeRF_Grasp_Metric_workspaces"
        / "2023-08-20_17-18-07"
        / "checkpoint_1000.pt"
    )
    GRASP_DATA_PATH = (
        pathlib.Path(nerf_grasping.get_repo_root())
        / "graspdata"
        / "sem-Camera-7bff4fd4dc53de7496dece3f86cb5dd5.npy"
    )

    init_grasps = AllegroGraspConfig.from_grasp_data(GRASP_DATA_PATH, batch_size=8)
    optimizer = AdamOptimizer.from_configs(
        init_grasps, NERF_CONFIG, CLASSIFIER_CHECKPOINT_PATH, lr=1e-2, momentum=0.9
    )

    scores, grasp_configs = run_optimizer_loop(optimizer, 100)

    assert (
        scores.shape[0] == grasp_configs.shape[0]
    ), f"{scores.shape[0]} != {grasp_configs.shape[0]}"
    assert all(
        x <= y for x, y in zip(scores[:-1], scores[1:])
    ), f"Scores are not sorted: {scores}"


if __name__ == "__main__":
    main()
