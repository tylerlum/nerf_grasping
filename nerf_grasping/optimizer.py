from __future__ import annotations
from nerf_grasping.optimizer_utils import AllegroGraspConfig, GraspMetric
import pathlib
import grasp_utils
from nerf_grasping.grasp_utils import NUM_FINGERS
import torch
from nerf_grasping.classifier import Classifier, CNN_3D_Classifier
from typing import Tuple
import nerf_grasping
from functools import partial

from rich.console import Console
from rich.table import Table

from rich.progress import (
    Progress,
    BarColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)


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


# Switch to format the tqdm progress bar based on whether we're in a notebook or not.
if is_notebook():
    from tqdm.notebook import tqdm as std_tqdm
else:
    from tqdm import tqdm as std_tqdm

tqdm = partial(std_tqdm, dynamic_ncols=True)


class Optimizer:
    """
    A base class for grasp optimizers.
    """

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
        """
        Factory method for creating an Optimizer from configs.
        """
        nerf = grasp_utils.load_nerf(nerf_config)
        classifier = Classifier(classifier_config)
        grasp_metric = GraspMetric(nerf, classifier)
        return cls(init_grasps, grasp_metric)

    def optimize(self) -> Tuple[torch.tensor, AllegroGraspConfig]:
        raise NotImplementedError


class SGDOptimizer(Optimizer):
    def __init__(
        self, init_grasps: AllegroGraspConfig, grasp_metric: GraspMetric, **kwargs
    ):
        """
        Constructor for SGDOptimizer.

        Args:
            init_grasps: Initial grasp configuration.
            grasp_metric: GraspMetric object defining the metric to optimize.
            **kwargs: Keyword arguments to pass to torch.optim.SGD.
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
    ) -> SGDOptimizer:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
        ) as progress:
            task = progress.add_task("Loading NeRF", total=1)
            nerf = grasp_utils.load_nerf(nerf_config)
            progress.update(task, advance=1)

        # TODO(pculbert): BRITTLE! Support more classifiers etc.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
        ) as progress:
            task = progress.add_task("Loading CNN", total=1)
            cnn = CNN_3D_Classifier(classifier_config).to(device=device)
            progress.update(task, advance=1)

        cnn.eval()

        # Put grasps on the correc device.
        init_grasps = init_grasps.to(device=device)

        # Wrap Nerf and Classifier in GraspMetric.
        grasp_metric = GraspMetric(nerf, cnn)

        return cls(init_grasps, grasp_metric, **kwargs)

    def step(self):
        self.optimizer.zero_grad()
        loss = self.grasp_metric(self.grasp_config)
        assert loss.shape == (self.grasp_config.batch_size,)

        # TODO(pculbert): Think about clipping joint angles
        # to feasible range.
        loss.mean().backward()
        self.optimizer.step()

    @property
    def grasp_scores(self) -> torch.tensor:
        return self.grasp_metric(self.grasp_config)


def run_optimizer_loop(
    optimizer: Optimizer, num_steps: int, console=Console()
) -> Tuple[torch.tensor, AllegroGraspConfig]:
    """
    Convenience function for running the optimizer loop.
    """

    start_column = TextColumn("[bold green]{task.description}[/bold green]")
    end_column = TextColumn(
        "Min score: {task.fields[min_score]} | Mean score: {task.fields[mean_score]} | Std score: {task.fields[std_score]}"
    )
    progress = Progress(
        SpinnerColumn(), *Progress.get_default_columns(), end_column, console=console
    )
    with progress:
        task_id = progress.add_task(
            "Optimizing grasps...",
            min_score="?",
            mean_score="?",
            std_score="?",
            total=num_steps,
        )
        for iter in range(num_steps):
            optimizer.step()

            # Update progress bar.
            progress.update(
                task_id,
                min_score=f"{optimizer.grasp_scores.min():.5f}",
                mean_score=f"{optimizer.grasp_scores.mean():.5f}",
                std_score=f"{optimizer.grasp_scores.std():.5f}",
                advance=1,
            )

            # TODO(pculbert): Add logging for grasps and scores.
            # Likely want to log min/mean of scores, and store the grasp configs

            # TODO(pculbert): Track best grasps across steps.

    # Sort grasp scores and configs by score.
    _, sort_indices = torch.sort(optimizer.grasp_scores, descending=False)
    return (optimizer.grasp_scores[sort_indices], optimizer.grasp_config[sort_indices])


def main() -> None:
    NERF_CONFIG = (
        pathlib.Path(nerf_grasping.get_repo_root())
        / "nerfcheckpoints/sem-Camera-7bff4fd4dc53de7496dece3f86cb5dd5_0_10/depth-nerfacto/2023-08-09_104724/config.yml"
    )
    CLASSIFIER_CHECKPOINT_PATH = (
        pathlib.Path(nerf_grasping.get_package_root())
        / "learned_metric"
        / "Train_DexGraspNet_NeRF_Grasp_Metric_workspaces"
        / "2023-08-20_17-18-07"
        / "checkpoint_1000.pt"
    )
    GRASP_DATA_PATH = (
        pathlib.Path(nerf_grasping.get_repo_root())
        / "graspdata"
        / "sem-Camera-7bff4fd4dc53de7496dece3f86cb5dd5.npy"
    )

    # Sample a batch of grasps from the grasp data.
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
    ) as progress:
        task = progress.add_task("Loading grasp data", total=1)
        init_grasps = AllegroGraspConfig.from_grasp_data(GRASP_DATA_PATH, batch_size=64)
        progress.update(task, advance=1)

    # Create SGDOptimizer.
    optimizer = SGDOptimizer.from_configs(
        init_grasps, NERF_CONFIG, CLASSIFIER_CHECKPOINT_PATH, lr=1e-4, momentum=0.9
    )

    console = Console()

    table = Table(title="Grasp scores")
    table.add_column("Iteration", justify="right")
    table.add_column("Min value")
    table.add_column("Mean value")
    table.add_column("Max value")
    table.add_column("Std dev.")

    table.add_row(
        "0",
        f"{optimizer.grasp_scores.min():.5f}",
        f"{optimizer.grasp_scores.mean():.5f}",
        f"{optimizer.grasp_scores.max():.5f}",
        f"{optimizer.grasp_scores.std():.5f}",
    )

    scores, grasp_configs = run_optimizer_loop(optimizer, num_steps=35, console=console)

    assert (
        scores.shape[0] == grasp_configs.batch_size
    ), f"{scores.shape[0]} != {grasp_configs.shape[0]}"
    assert all(
        x <= y for x, y in zip(scores[:-1], scores[1:])
    ), f"Scores are not sorted: {scores}"

    table.add_row(
        "35",
        f"{scores.min():.5f}",
        f"{scores.mean():.5f}",
        f"{scores.max():.5f}",
        f"{scores.std():.5f}",
    )
    console.print(table)


if __name__ == "__main__":
    main()
