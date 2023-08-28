from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

WANDB_DATETIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


@dataclass(frozen=True)
class WandbConfig:
    """Parameters for logging to wandb."""

    project: str
    """Name of the wandb project."""

    entity: Optional[str] = None
    """Account associated with the wandb project."""

    name: str = field(default_factory=lambda: WANDB_DATETIME_STR)
    """Name of the run."""

    group: str = ""
    """Name of the run group."""

    job_type: str = ""
    """Name of the job type."""
