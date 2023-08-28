from __future__ import annotations
from nerf_grasping.optimizer_utils import (
    AllegroGraspConfig,
    GraspMetric,
    get_split_inds,
)
from dataclasses import asdict
from nerf_grasping.config.metric_config import GraspMetricConfig
import pathlib
import grasp_utils
import torch
from nerf_grasping.classifier import Classifier
from nerf_grasping.config.classifier_config import ClassifierConfig
from nerf_grasping.config.optimizer_config import GraspOptimizerConfig
from typing import Tuple
import nerf_grasping
from functools import partial
import numpy as np
import tyro
import wandb

from rich.console import Console
from rich.table import Table

from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from tap import Tap

PRINT_FREQ = 5


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
        self,
        init_grasps: AllegroGraspConfig,
        grasp_metric: GraspMetric,
        optimizer_config: GraspOptimizerConfig,
    ):
        """
        Constructor for SGDOptimizer.

        Args:
            init_grasps: Initial grasp configuration.
            grasp_metric: GraspMetric object defining the metric to optimize.
            **kwargs: Keyword arguments to pass to torch.optim.SGD.
        """
        super().__init__(init_grasps, grasp_metric)
        self.optimizer = torch.optim.SGD(
            self.grasp_config.parameters(),
            lr=optimizer_config.lr,
            momentum=optimizer_config.momentum,
        )

    # TODO: refactor to dispatch optimizer on config type.
    @classmethod
    def from_configs(
        cls,
        init_grasps: AllegroGraspConfig,
        nerf_config: pathlib.Path,
        classifier_config: ClassifierConfig,
        optimizer_config: GraspOptimizerConfig,
        classifier_checkpoint: int = -1,
        console=Console(width=120),
    ) -> SGDOptimizer:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description} "),
            console=console,
        ) as progress:
            task = progress.add_task("Loading NeRF", total=1)
            nerf = grasp_utils.load_nerf(nerf_config)
            progress.update(task, advance=1)

        # TODO(pculbert): BRITTLE! Support more classifiers etc.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description} "),
            console=console,
        ) as progress:
            task = progress.add_task("Loading CNN", total=1)
            cnn = classifier_config.model_config.get_classifier().to(device=device)

            # Load checkpoint if specified.
            checkpoint_path = (
                pathlib.Path(classifier_config.checkpoint_workspace.root_dir)
                / classifier_config.checkpoint_workspace.leaf_dir
            )

            if classifier_checkpoint == -1:
                # take latest checkpoint.
                all_checkpoints = checkpoint_path.glob("*.pt")
                checkpoint_path = max(all_checkpoints, key=pathlib.Path.stat)
            else:
                checkpoint_path = (
                    checkpoint_path / f"checkpoint_{classifier_checkpoint:04}.pt"
                )

            cnn.model.load_state_dict(
                torch.load(checkpoint_path)["nerf_to_grasp_success_model"]
            )

            progress.update(task, advance=1)

        cnn.eval()

        # Put grasps on the correc device.
        init_grasps = init_grasps.to(device=device)

        # Wrap Nerf and Classifier in GraspMetric.
        grasp_metric = GraspMetric(nerf, cnn)

        return cls(init_grasps, grasp_metric, optimizer_config)

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
    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TextColumn("=>"),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task_id = progress.add_task(
            "Optimizing grasps...",
            total=num_steps,
        )
        for iter in range(num_steps):
            wandb_log_dict = {}
            wandb_log_dict["optimization_step"] = iter
            optimizer.step()

            # Update progress bar.
            progress.update(
                task_id,
                advance=1,
            )

            if iter % PRINT_FREQ == 0:
                console.print(
                    f"Iter: {iter} | Min score: {optimizer.grasp_scores.min():.3f} | Max score: {optimizer.grasp_scores.max():.3f} | Mean score: {optimizer.grasp_scores.mean():.3f} | Std dev: {optimizer.grasp_scores.std():.3f}"
                )

            # Log to wandb.
            wandb_log_dict["scores"] = optimizer.grasp_scores.detach().cpu().numpy()
            wandb_log_dict["min_score"] = optimizer.grasp_scores.min().item()
            wandb_log_dict["max_score"] = optimizer.grasp_scores.max().item()
            wandb_log_dict["mean_score"] = optimizer.grasp_scores.mean().item()
            wandb_log_dict["std_score"] = optimizer.grasp_scores.std().item()

            if wandb.run is not None:
                wandb.log(wandb_log_dict)

    # Sort grasp scores and configs by score.
    _, sort_indices = torch.sort(optimizer.grasp_scores, descending=False)
    return (
        optimizer.grasp_scores[sort_indices],
        optimizer.grasp_config[sort_indices],
    )


def main(cfg: GraspMetricConfig) -> None:
    # Create rich.Console object.

    torch.random.manual_seed(0)
    np.random.seed(0)

    console = Console(width=120)

    if cfg.wandb:
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=asdict(cfg),
        )

    # Sample a batch of grasps from the grasp data.
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description} "),
        console=console,
    ) as progress:
        task = progress.add_task("Loading grasp data", total=1)

        # TODO: Find a way to load a particular split of the grasp_data.
        grasp_config_dicts = np.load(
            cfg.init_grasp_config_dicts_path, allow_pickle=True
        )
        init_grasps = AllegroGraspConfig.from_grasp_config_dicts(grasp_config_dicts)
        # Get indices of preferred split.
        train_inds, val_inds, test_inds = get_split_inds(
            init_grasps.batch_size,
            [
                cfg.classifier_config.data.frac_train,
                cfg.classifier_config.data.frac_val,
                cfg.classifier_config.data.frac_test,
            ],
            cfg.classifier_config.random_seed,
        )

        if cfg.grasp_split == "train":
            data_inds = train_inds
        elif cfg.grasp_split == "val":
            data_inds = val_inds
        elif cfg.grasp_split == "test":
            data_inds = test_inds
        else:
            raise ValueError(f"Invalid grasp_split: {cfg.grasp_split}")

        data_inds = np.random.choice(data_inds, size=cfg.optimizer.num_grasps)
        init_grasps = init_grasps[data_inds]
        progress.update(task, advance=1)

    # Create SGDOptimizer.
    optimizer = SGDOptimizer.from_configs(
        init_grasps,
        cfg.nerf_checkpoint_path,
        cfg.classifier_config,
        cfg.optimizer,
        cfg.classifier_checkpoint,
        console=console,
    )

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

    scores, grasp_configs = run_optimizer_loop(
        optimizer, num_steps=cfg.optimizer.num_steps, console=console
    )

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

    grasp_config_dicts = grasp_configs.as_dicts()
    for ii, dd in enumerate(grasp_config_dicts):
        dd["score"] = scores[ii].item()

    print(f"Saving sorted grasp configs to {cfg.output_path}")
    np.save(str(cfg.output_path), grasp_config_dicts)

    wandb.finish()


if __name__ == "__main__":
    cfg = tyro.cli(GraspMetricConfig)
    main(cfg)
