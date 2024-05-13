from typing import Tuple
import torch
from torch.utils import data


class GraspBPSDataset(data.Dataset):
    def __init__(self, grasps: torch.Tensor, bpss: torch.Tensor) -> None:
        self.grasps = grasps
        self.bpss = bpss

    def __len__(self) -> int:
        return len(self.grasps)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.grasps[idx], self.bpss[idx]
