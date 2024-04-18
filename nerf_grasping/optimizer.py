from __future__ import annotations
import pypose as pp
from collections import defaultdict
from nerf_grasping.optimizer_utils import (
    AllegroGraspConfig,
    GraspMetric,
    DepthImageGraspMetric,
    predict_in_collision_with_object,
)
from dataclasses import asdict
from nerf_grasping.config.optimization_config import OptimizationConfig
import pathlib
import torch
from nerf_grasping.classifier import Classifier, Simple_CNN_LSTM_Classifier
from nerf_grasping.config.classifier_config import ClassifierConfig
from nerf_grasping.config.nerfdata_config import DepthImageNerfDataConfig
from nerf_grasping.config.optimizer_config import (
    SGDOptimizerConfig,
    CEMOptimizerConfig,
)
from typing import Tuple, Union, Dict
import nerf_grasping
from functools import partial
import numpy as np
import tyro
import wandb

from rich.console import Console
from rich.table import Table

from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from contextlib import nullcontext


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

    def __init__(
        self,
        init_grasp_config: AllegroGraspConfig,
        grasp_metric: Union[GraspMetric, DepthImageGraspMetric],
    ):
        # Put on the correct device. (TODO: DO WE NEED THIS?)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        grasp_metric = grasp_metric.to(device=device)
        init_grasp_config = init_grasp_config.to(device=device)

        self.grasp_config = init_grasp_config
        self.grasp_metric = grasp_metric

    @property
    def grasp_losses(self) -> torch.Tensor:
        return self.grasp_metric.get_failure_probability(self.grasp_config)

    def step(self):
        raise NotImplementedError()


class SGDOptimizer(Optimizer):
    def __init__(
        self,
        init_grasp_config: AllegroGraspConfig,
        grasp_metric: Union[GraspMetric, DepthImageGraspMetric],
        optimizer_config: SGDOptimizerConfig,
    ):
        """
        Constructor for SGDOptimizer.

        Args:
            init_grasp_config: Initial grasp configuration.
            grasp_metric: Union[GraspMetric, DepthImageGraspMetric] object defining the metric to optimize.
            optimizer_config: SGDOptimizerConfig object defining the optimizer configuration.
        """
        super().__init__(init_grasp_config, grasp_metric)
        self.optimizer_config = optimizer_config

        # Add requires_grad to grasp config.
        init_grasp_config.wrist_pose.requires_grad = optimizer_config.opt_wrist_pose
        init_grasp_config.grasp_orientations.requires_grad = (
            optimizer_config.opt_grasp_dirs
        )

        # TODO: Config this
        USE_ADAMW = True

        if optimizer_config.opt_wrist_pose:
            self.wrist_optimizer = (
                torch.optim.SGD(
                    [self.grasp_config.wrist_pose],
                    lr=optimizer_config.wrist_lr,
                    momentum=optimizer_config.momentum,
                )
                if not USE_ADAMW
                else torch.optim.AdamW(
                    [self.grasp_config.wrist_pose],
                    lr=optimizer_config.wrist_lr,
                )
            )

        self.joint_optimizer = (
            torch.optim.SGD(
                [self.grasp_config.joint_angles],
                lr=optimizer_config.finger_lr,
                momentum=optimizer_config.momentum,
            )
            if not USE_ADAMW
            else torch.optim.AdamW(
                [self.grasp_config.joint_angles],
                lr=optimizer_config.finger_lr,
            )
        )

        if optimizer_config.opt_grasp_dirs:
            self.grasp_dir_optimizer = (
                torch.optim.SGD(
                    [self.grasp_config.grasp_orientations],
                    lr=optimizer_config.grasp_dir_lr,
                    momentum=optimizer_config.momentum,
                )
                if not USE_ADAMW
                else torch.optim.AdamW(
                    [self.grasp_config.grasp_orientations],
                    lr=optimizer_config.grasp_dir_lr,
                )
            )

    def step(self):
        self.joint_optimizer.zero_grad()
        if self.optimizer_config.opt_wrist_pose:
            self.wrist_optimizer.zero_grad()
        if self.optimizer_config.opt_grasp_dirs:
            self.grasp_dir_optimizer.zero_grad()
        losses = self.grasp_losses
        assert losses.shape == (self.grasp_config.batch_size,)

        # TODO(pculbert): Think about clipping joint angles
        # to feasible range.
        losses.sum().backward()  # Should be sum so gradient magnitude per parameter is invariant to batch size.

        self.joint_optimizer.step()
        if self.optimizer_config.opt_wrist_pose:
            self.wrist_optimizer.step()
        if self.optimizer_config.opt_grasp_dirs:
            self.grasp_dir_optimizer.step()


class CEMOptimizer(Optimizer):
    def __init__(
        self,
        grasp_config: AllegroGraspConfig,
        grasp_metric: Union[GraspMetric, DepthImageGraspMetric],
        optimizer_config: CEMOptimizerConfig,
    ):
        """
        Constructor for SGDOptimizer.

        Args:
            init_grasp_config: Initial grasp configuration.
            grasp_metric: Union[GraspMetric, DepthImageGraspMetric] object defining the metric to optimize.
            optimizer_config: SGDOptimizerConfig object defining the optimizer configuration.
        """
        super().__init__(grasp_config, grasp_metric)
        self.optimizer_config = optimizer_config

    def step(self):
        # Find the elite fraction of samples.
        elite_inds = torch.argsort(self.grasp_losses)[: self.optimizer_config.num_elite]
        elite_grasps = self.grasp_config[elite_inds]

        # Compute the mean and covariance of the grasp config.
        elite_mean = elite_grasps.mean()
        (
            elite_cov_wrist_pose,
            elite_cov_joint_angles,
            elite_cov_grasp_orientations,
        ) = elite_grasps.cov()

        elite_chol_wrist_pose = (
            torch.linalg.cholesky(
                elite_cov_wrist_pose
                + self.optimizer_config.min_cov_std**2
                * torch.eye(6, device=elite_cov_wrist_pose.device)
            )
        ).unsqueeze(0)

        wrist_pose_perturbations = torch.randn_like(
            elite_mean.wrist_pose.Log()
            .expand(self.optimizer_config.num_samples, -1)
            .unsqueeze(-1)
        )

        wrist_pose_innovations = (
            elite_chol_wrist_pose @ wrist_pose_perturbations
        ).squeeze(-1)

        elite_chol_joint_angles = (
            torch.linalg.cholesky(
                elite_cov_joint_angles
                + self.optimizer_config.min_cov_std**2
                * torch.eye(16, device=elite_cov_joint_angles.device)
            )
        ).unsqueeze(0)

        joint_angle_perturbations = torch.randn_like(
            elite_mean.joint_angles.expand(self.optimizer_config.num_samples, -1)
        ).unsqueeze(-1)

        joint_angle_innovations = (
            elite_chol_joint_angles @ joint_angle_perturbations
        ).squeeze(-1)

        elite_chol_grasp_orientations = (
            torch.linalg.cholesky(
                elite_cov_grasp_orientations
                + self.optimizer_config.min_cov_std**2
                * torch.eye(3, device=elite_cov_grasp_orientations.device).unsqueeze(0)
            )
        ).unsqueeze(0)

        grasp_orientation_perturbations = (
            torch.randn_like(elite_mean.grasp_orientations.Log())
            .expand(self.optimizer_config.num_samples, -1, -1)
            .unsqueeze(-1)
        )

        grasp_orientation_innovations = (
            elite_chol_grasp_orientations @ grasp_orientation_perturbations
        ).squeeze(-1)

        # Sample grasp configs from the current mean and covariance.
        self.grasp_config = AllegroGraspConfig.from_values(
            wrist_pose=elite_mean.wrist_pose.expand(
                self.optimizer_config.num_samples, -1
            )
            + wrist_pose_innovations,
            joint_angles=elite_mean.joint_angles.expand(
                self.optimizer_config.num_samples, -1
            )
            + joint_angle_innovations,
            grasp_orientations=elite_mean.grasp_orientations.expand(
                self.optimizer_config.num_samples, -1, -1
            )
            + grasp_orientation_innovations,
        )


def run_optimizer_loop(
    optimizer: Optimizer,
    optimizer_config: Union[SGDOptimizerConfig, CEMOptimizerConfig],
    print_freq: int,
    save_grasps_freq: int,
    output_path: pathlib.Path,
    use_rich: bool = False,
    console=Console(),
) -> Tuple[torch.Tensor, AllegroGraspConfig]:
    """
    Convenience function for running the optimizer loop.
    """

    with (
        Progress(
            TextColumn("[bold green]{task.description}[/bold green]"),
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TextColumn("=>"),
            TimeElapsedColumn(),
            console=console,
        )
        if use_rich
        else nullcontext()
    ) as progress:
        task_id = (
            progress.add_task(
                "Optimizing grasps...",
                total=optimizer_config.num_steps,
            )
            if progress is not None
            else None
        )

        for iter in range(optimizer_config.num_steps):
            wandb_log_dict = {}
            wandb_log_dict["optimization_step"] = iter

            if iter % print_freq == 0:
                losses_np = optimizer.grasp_losses.detach().cpu().numpy()
                # console.print(
                #     f"Iter: {iter} | Min loss: {grasp_loss.min():.3f} | Max loss: {grasp_loss.max():.3f} | Mean loss: {grasp_loss.mean():.3f} | Std dev: {grasp_loss.std():.3f}"
                # )
                print(
                    f"Iter: {iter} | Losses: {np.round(losses_np.tolist(), decimals=3)} | Min loss: {losses_np.min():.3f} | Max loss: {losses_np.max():.3f} | Mean loss: {losses_np.mean():.3f} | Std dev: {losses_np.std():.3f}"
                )

            optimizer.step()

            # Update progress bar.
            if progress is not None and task_id is not None:
                progress.update(
                    task_id,
                    advance=1,
                )

            # Log to wandb.
            grasp_losses_np = optimizer.grasp_losses.detach().cpu().numpy()
            for i, loss in enumerate(grasp_losses_np.tolist()):
                wandb_log_dict[f"loss_{i}"] = loss
            wandb_log_dict["min_loss"] = grasp_losses_np.min().item()
            wandb_log_dict["max_loss"] = grasp_losses_np.max().item()
            wandb_log_dict["mean_loss"] = grasp_losses_np.mean().item()
            wandb_log_dict["std_loss"] = grasp_losses_np.std().item()

            if wandb.run is not None:
                wandb.log(wandb_log_dict)

            if iter % save_grasps_freq == 0:
                # Save mid optimization grasps to file
                grasp_config_dict = optimizer.grasp_config.as_dict()
                grasp_config_dict["loss"] = grasp_losses_np

                # To interface with mid optimization visualizer, need to create new folder (mid_optimization_folder_path)
                # that has folders with iteration number
                # TODO: Decide if this should just store in output_path.parent (does this cause issues?) or store in new folder
                # <mid_optimization_folder_path>
                #    - 0
                #        - <object_code_and_scale_str>.py
                #    - x
                #        - <object_code_and_scale_str>.py
                #    - 2x
                #        - <object_code_and_scale_str>.py
                #    - 3x
                #        - <object_code_and_scale_str>.py
                main_output_folder_path, filename = (
                    output_path.parent,
                    output_path.name,
                )
                mid_optimization_folder_path = (
                    main_output_folder_path / "mid_optimization"
                )
                this_iter_folder_path = mid_optimization_folder_path / f"{iter}"
                this_iter_folder_path.mkdir(parents=True, exist_ok=True)
                print(f"Saving mid opt grasp config dict to {this_iter_folder_path}")
                np.save(
                    this_iter_folder_path / filename,
                    grasp_config_dict,
                    allow_pickle=True,
                )

    optimizer.grasp_metric.eval()

    return (
        optimizer.grasp_losses,
        optimizer.grasp_config,
    )


def get_optimized_grasps(
    cfg: OptimizationConfig,
    grasp_metric: Union[GraspMetric, DepthImageGraspMetric] = None,
) -> Dict[str, np.ndarray]:
    # print("=" * 80)
    # print(f"Config:\n{tyro.extras.to_yaml(cfg)}")
    # print("=" * 80 + "\n")

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

    with (
        Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description} "),
            console=console,
        )
        if cfg.use_rich
        else nullcontext()
    ) as progress:
        task = (
            progress.add_task("Loading grasp data", total=1)
            if progress is not None
            else None
        )

        # TODO: Find a way to load a particular split of the grasp_data.
        init_grasp_config_dict = np.load(
            cfg.init_grasp_config_dict_path, allow_pickle=True
        ).item()

        # HACK: For now, just take every 400th grasp.
        # for key in init_grasp_config_dict.keys():
        #     init_grasp_config_dict[key] = init_grasp_config_dict[key][::400]

        init_grasp_configs = AllegroGraspConfig.from_grasp_config_dict(
            init_grasp_config_dict
        )

        # HACK: For now, just take the first num_grasps.
        # init_grasp_configs = init_grasp_configs[: cfg.optimizer.num_grasps]

        if progress is not None and task is not None:
            progress.update(task, advance=1)

    # Create grasp metric
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if grasp_metric is None:
        print(
            f"Loading classifier config from {cfg.grasp_metric.classifier_config_path}"
        )
        USE_DEPTH_IMAGES = isinstance(
            cfg.grasp_metric.classifier_config.nerfdata_config, DepthImageNerfDataConfig
        )
        if USE_DEPTH_IMAGES:
            grasp_metric = DepthImageGraspMetric.from_config(
                cfg.grasp_metric,
                console=console,
            )
        else:
            grasp_metric = GraspMetric.from_config(
                cfg.grasp_metric,
                console=console,
            )
    else:
        print("Using provided grasp metric.")
    grasp_metric = grasp_metric.to(device=device)

    GET_BEST_GRASPS = True
    if GET_BEST_GRASPS:
        BATCH_SIZE = 64
        n_batches = init_grasp_configs.batch_size // BATCH_SIZE
        all_preds = []
        all_grasp_configs = []
        all_predicted_in_collision = []
        with torch.no_grad():
            N_SAMPLES = 1
            for i in range(N_SAMPLES):
                temp_preds = []

                original_grasp_configs = AllegroGraspConfig.from_grasp_config_dict(
                    init_grasp_config_dict
                )
                if i != 0:
                    random_rotate_transforms = (
                        sample_random_rotate_transforms_only_around_y(
                            original_grasp_configs.batch_size
                        )
                    )
                    original_grasp_configs.hand_config.set_wrist_pose(
                        random_rotate_transforms
                        @ original_grasp_configs.hand_config.wrist_pose
                    )

                for batch_i in tqdm(range(n_batches)):
                    preds = grasp_metric.get_failure_probability(
                        original_grasp_configs[
                            batch_i * BATCH_SIZE : (batch_i + 1) * BATCH_SIZE
                        ].to(device=device)
                    )
                    temp_preds.append(1 - preds.detach().cpu().numpy())
                if n_batches * BATCH_SIZE < original_grasp_configs.batch_size:
                    preds = grasp_metric.get_failure_probability(
                        original_grasp_configs[n_batches * BATCH_SIZE :].to(
                            device=device
                        )
                    )
                    temp_preds.append(1 - preds.detach().cpu().numpy())
                all_grasp_configs.append(original_grasp_configs)
                all_preds.append(np.concatenate(temp_preds, axis=0))

                predicted_in_collision = predict_in_collision_with_object(
                    nerf_field=grasp_metric.nerf_field,
                    grasp_config=original_grasp_configs.to(device),
                )
                all_predicted_in_collision.append(predicted_in_collision)

            all_preds = np.array(all_preds)
            assert all_preds.shape == (N_SAMPLES, original_grasp_configs.batch_size)
            all_preds = all_preds.reshape(-1)

            all_predicted_in_collision = np.array(all_predicted_in_collision)
            assert all_predicted_in_collision.shape == (
                N_SAMPLES,
                original_grasp_configs.batch_size,
            )
            all_predicted_in_collision = all_predicted_in_collision.reshape(-1)

            all_grasp_config_dicts = defaultdict(list)
            for i in range(N_SAMPLES):
                config_dict = all_grasp_configs[i].as_dict()
                for k, v in config_dict.items():
                    all_grasp_config_dicts[k].append(v)
            for k, v in all_grasp_config_dicts.items():
                all_grasp_config_dicts[k] = np.concatenate(v, axis=0)
            all_grasp_configs = AllegroGraspConfig.from_grasp_config_dict(
                all_grasp_config_dicts
            )
            assert (
                all_grasp_configs.batch_size
                == original_grasp_configs.batch_size * N_SAMPLES
            )
            CHECK_COLLISION = True
            if CHECK_COLLISION:
                new_all_preds = np.where(
                    all_predicted_in_collision,
                    np.zeros_like(all_preds),
                    all_preds,
                )
            else:
                new_all_preds = all_preds
            ordered_idxs_best_first = np.argsort(new_all_preds)[::-1].copy()
            breakpoint()  # TODO: Debug here
            all_grasp_configs = all_grasp_configs[ordered_idxs_best_first]
    init_grasp_configs = all_grasp_configs[: cfg.optimizer.num_grasps]

    # Create Optimizer.
    if isinstance(cfg.optimizer, SGDOptimizerConfig):
        optimizer = SGDOptimizer(
            init_grasp_configs,
            grasp_metric,
            cfg.optimizer,
        )
    elif isinstance(cfg.optimizer, CEMOptimizerConfig):
        optimizer = CEMOptimizer(
            init_grasp_configs,
            grasp_metric,
            cfg.optimizer,
        )
    else:
        raise ValueError(f"Invalid optimizer config: {cfg.optimizer}")

    init_losses = optimizer.grasp_losses

    table = Table(title="Grasp loss")
    table.add_column("Iteration", justify="right")
    table.add_column("Min loss")
    table.add_column("Mean loss")
    table.add_column("Max loss")
    table.add_column("Std dev.")

    table.add_row(
        "0",
        f"{init_losses.min():.5f}",
        f"{init_losses.mean():.5f}",
        f"{init_losses.max():.5f}",
        f"{init_losses.std():.5f}",
    )

    final_losses, final_grasp_configs = run_optimizer_loop(
        optimizer,
        optimizer_config=cfg.optimizer,
        print_freq=cfg.print_freq,
        save_grasps_freq=cfg.save_grasps_freq,
        output_path=cfg.output_path,
        use_rich=cfg.use_rich,
        console=console,
    )

    assert (
        final_losses.shape[0] == final_grasp_configs.batch_size
    ), f"{final_losses.shape[0]} != {final_grasp_configs.batch_size}"

    print(f"Initial grasp loss: {np.round(init_losses.tolist(), decimals=3)}")
    print(f"Final grasp loss: {np.round(final_losses.tolist(), decimals=3)}")

    table.add_row(
        f"{cfg.optimizer.num_steps}",
        f"{final_losses.min():.5f}",
        f"{final_losses.mean():.5f}",
        f"{final_losses.max():.5f}",
        f"{final_losses.std():.5f}",
    )
    console.print(table)

    # HACK
    # grasp_config_dict = COPY.as_dict()
    grasp_config_dict = final_grasp_configs.as_dict()
    grasp_config_dict["loss"] = final_losses.detach().cpu().numpy()

    print(f"Saving final grasp config dict to {cfg.output_path}")
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cfg.output_path), grasp_config_dict, allow_pickle=True)

    wandb.finish()
    return grasp_config_dict


def main() -> None:
    cfg = tyro.cli(OptimizationConfig)
    get_optimized_grasps(cfg)


def sample_random_rotate_transforms_only_around_y(N: int) -> pp.LieTensor:
    PP_MATRIX_ATOL, PP_MATRIX_RTOL = 1e-4, 1e-4
    # Sample big rotations in tangent space of SO(3).
    # Choose 4 * \pi as a heuristic to get pretty evenly spaced rotations.
    # TODO(pculbert): Figure out better uniform sampling on SO(3).
    x_rotations = torch.zeros(N)
    y_rotations = 4 * torch.pi * (2 * torch.rand(N) - 1)
    z_rotations = torch.zeros(N)
    xyz_rotations = torch.stack([x_rotations, y_rotations, z_rotations], dim=-1)
    log_random_rotations = pp.so3(xyz_rotations)

    # Return exponentiated rotations.
    random_SO3_rotations = log_random_rotations.Exp()

    # A bit annoying -- need to cast SO(3) -> SE(3).
    random_rotate_transforms = pp.from_matrix(
        random_SO3_rotations.matrix(),
        pp.SE3_type,
        atol=PP_MATRIX_ATOL,
        rtol=PP_MATRIX_RTOL,
    )

    return random_rotate_transforms


if __name__ == "__main__":
    main()
