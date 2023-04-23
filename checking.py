from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf, MISSING
from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
OmegaConf.register_new_resolver("eval", eval, replace=True)

# Define your data classes
@dataclass
class WandbConfig:
    entity: str = MISSING
    project: str = MISSING
    name: str = MISSING
    group: str = MISSING
    job_type: str = MISSING


@dataclass
class DataConfig:
    frac_val: float = MISSING
    frac_test: float = MISSING
    frac_train: float = MISSING
    input_dataset_dir: str = MISSING


@dataclass
class Config:
    wandb: WandbConfig = MISSING
    data: DataConfig = MISSING

# Add your config to the ConfigStore
config_store = ConfigStore.instance()
config_store.store(name="config", node=Config)

# Initialize Hydra
with initialize(version_base="1.2", config_path="./Train_NeRF_Grasp_Metric_cfg"):
    # Load your config
    cfg: DictConfig = compose(config_name="config")
    # Convert it to a structured config using OmegaConf
    structured_cfg: Config = OmegaConf.create(cfg)

print(OmegaConf.to_yaml(structured_cfg))
print(structured_cfg.data.frac_train)
print(type(structured_cfg))