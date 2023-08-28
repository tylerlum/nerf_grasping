from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

# A general date-time string for naming runs -- shared across all config modules.
CONFIG_DATETIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


@dataclass(frozen=True)
class WandbConfig:
    """Parameters for logging to wandb."""

    project: str
    """Name of the wandb project."""

    entity: Optional[str] = None
    """Account associated with the wandb project."""

    name: str = field(default_factory=lambda: CONFIG_DATETIME_STR)
    """Name of the run."""

    group: str = ""
    """Name of the run group."""

    job_type: str = ""
    """Name of the job type."""
