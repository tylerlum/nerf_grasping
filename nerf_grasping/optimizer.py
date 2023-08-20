from __future__ import annotations
from nerf_grasping.optimizer_utils import AllegroGraspConfig, GraspMetric
import pathlib
import grasp_utils
import torch
from nerf_grasping.classifier import Classifier
from typing import Tuple
import nerf_grasping


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
        self.optimizer = torch.optim.Adam(self.grasp_config.parameters(), **kwargs)

    @classmethod
    def from_configs(
        cls,
        init_grasps: AllegroGraspConfig,
        nerf_config: pathlib.Path,
        classifier_config: pathlib.Path,
        **kwargs,
    ) -> AdamOptimizer:
        nerf = grasp_utils.load_nerf(nerf_config)
        classifier = Classifier(classifier_config)
        grasp_metric = GraspMetric(nerf, classifier)
        return cls(init_grasps, grasp_metric, **kwargs)

    def step(self):
        self.optimizer.zero_grad()
        loss = self.grasp_metric(self.grasp_config)
        loss.backward()
        self.optimizer.step()

    @property
    def grasp_scores(self) -> torch.tensor:
        return self.grasp_metric(self.grasp_config)


def run_optimizer_loop(
    optimizer: Optimizer, num_steps: int
) -> Tuple[torch.tensor, AllegroGraspConfig]:
    for _ in range(num_steps):
        optimizer.step()
        # TODO(pculbert): Add logging for grasps and scores.
        # Likely want to log min/mean of scores, and store the grasp configs

        # TODO(pculbert): Track best grasps across steps.

    _, sort_indices = torch.sort(optimizer.grasp_scores, descending=True)
    return (optimizer.grasp_scores[sort_indices], optimizer.grasp_config[sort_indices])


def main() -> None:
    NERF_CONFIG = (
        pathlib.Path(nerf_grasping.get_repo_root())
        / "nerfcheckpoints"
        / "sem-Car-f9c2bc7b4ef896e7146ff63b4c7525d9_0_15"
    )
    CLASSIFIER_CONFIG = (
        pathlib.Path(nerf_grasping.get_package_root())
        / "models"
        / "3D_CNN_1_config.yaml"
    )
    init_grasps = AllegroGraspConfig(batch_size=256)
    optimizer = Optimizer.from_configs(init_grasps, NERF_CONFIG, CLASSIFIER_CONFIG)

    scores, grasp_configs = run_optimizer_loop(optimizer, 100)

    assert (
        scores.shape[0] == grasp_configs.shape[0]
    ), f"{scores.shape[0]} != {grasp_configs.shape[0]}"
    assert all(
        x <= y for x, y in zip(scores[:-1], scores[1:])
    ), f"Scores are not sorted: {scores}"


if __name__ == "__main__":
    main()
