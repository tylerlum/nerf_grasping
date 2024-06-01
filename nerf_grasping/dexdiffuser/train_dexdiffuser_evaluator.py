from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from wandb.util import generate_id

from nerf_grasping.dexdiffuser.dex_evaluator import DexEvaluator
from nerf_grasping.dexdiffuser.diffusion import get_dataset


@dataclass
class DexEvaluatorTrainingConfig:
    # training parameters
    batch_size: int = 32768
    learning_rate: float = 1e-4
    num_epochs: int = 10000
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_seed: int = 42

    # validation, printing, and saving
    print_freq: int = 100
    snapshot_freq: int = 5000
    log_path: Path = Path("logs/dexdiffuser_evaluator")

    # wandb
    wandb_project: str = "dexdiffuser-evaluator"
    wandb_log: bool = True


def setup(cfg: DexEvaluatorTrainingConfig):
    """Sets up the training loop."""
    # set random seed
    torch.cuda.manual_seed_all(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    # get datasets
    train_dataset, test_dataset, _ = get_dataset()

    # make dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    # make other stuff we need
    model = DexEvaluator(in_grasp=25).to(cfg.device)  # 25 for dexdiffuser baseline
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)

    # logging
    wandb_id = generate_id()
    if cfg.wandb_log:
        wandb.init(project=cfg.wandb_project, config=cfg, id=wandb_id, resume="allow")

    return train_loader, test_loader, model, optimizer, scheduler, wandb_id

def main(cfg: DexEvaluatorTrainingConfig) -> None:
    """Main function."""
    train_loader, test_loader, model, optimizer, scheduler, wandb_id = setup(cfg)

    # make log path
    cfg.log_path.mkdir(parents=True, exist_ok=True)

    # update tqdm bar with train and val loss
    for epoch in tqdm(range(cfg.num_epochs), desc="Epoch", leave=True):
        model.train()
        for i, (f_O, _g_O, y) in enumerate(train_loader):
            f_O, _g_O, y = f_O.to(cfg.device), _g_O.to(cfg.device), y.to(cfg.device)
            g_O = _g_O[..., :-12]  # don't include the grasp dirs for dexdiffuser
            optimizer.zero_grad()
            y_pred = model(f_O, g_O).squeeze(-1)
            assert y_pred.shape == y.shape
            loss = torch.nn.functional.mse_loss(y_pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

        model.eval()
        with torch.no_grad():
            for i, (f_O, _g_O, y) in enumerate(test_loader):
                f_O, _g_O, y = f_O.to(cfg.device), _g_O.to(cfg.device), y.to(cfg.device)
                g_O = _g_O[..., :-12]  # don't include the grasp dirs for dexdiffuser
                y_pred = model(f_O, g_O).squeeze(-1)
                assert y_pred.shape == y.shape
                val_loss = torch.nn.functional.mse_loss(y_pred, y)
        
        if i % cfg.print_freq == 0:
            print(f"Epoch {epoch}, batch {i}, train_loss: {loss.item()}, val_loss: {val_loss.item()}")

        if epoch % cfg.snapshot_freq == 0:
            print(f"Saving model at epoch {epoch}!")
            torch.save(model.state_dict(), cfg.log_path / f"ckpt-{wandb_id}-step-{epoch}.pth")
            

if __name__ == "__main__":
    cfg = DexEvaluatorTrainingConfig(
        num_epochs=10000,
        batch_size=32768,
        learning_rate=1e-4,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        random_seed=42,
        print_freq=100,
        snapshot_freq=1000,
        log_path=Path("logs/dexdiffuser_evaluator"),
        wandb_project="dexdiffuser-evaluator",
        wandb_log=True,
    )
    main(cfg)
