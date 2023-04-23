from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
OmegaConf.register_new_resolver("eval", eval, replace=True)

# Define your data classes
@dataclass
class WandbConfig:
    entity: str
    project: str
    name: str
    group: str
    job_type: str


@dataclass
class DataConfig:
    frac_val: float
    frac_test: float
    frac_train: float
    input_dataset_dir: str


@dataclass
class Config:
    wandb: WandbConfig
    data: DataConfig

# Add your config to the ConfigStore
config_store = ConfigStore.instance()
config_store.store(name="config", node=Config)

# Initialize Hydra
with initialize(config_path="./Train_NeRF_Grasp_Metric_cfg"):
    # Load your config
    cfg: DictConfig = compose(config_name="config")
    # Convert it to a structured config using OmegaConf
    structured_cfg: Config = OmegaConf.create(cfg)

print(OmegaConf.to_yaml(structured_cfg))
print(structured_cfg.data.frac_train)
print(type(structured_cfg))