import torch
import torch.nn as nn


class FCResBlock(nn.Module):
    """FFHNet: https://ieeexplore.ieee.org/document/9811666:
    The core building block of both models is the FC ResBlock, which consists of two parallel paths from input to output.
    One path consists of a single FC layer, the other path has two FC layers. Each is followed by a layer of batch norm (BN).
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Path 1
        self.fc_path1 = nn.Linear(in_features, out_features)
        self.bn_path1 = nn.BatchNorm1d(out_features)

        # Path 2
        self.fc_path2_1 = nn.Linear(in_features, out_features)
        self.fc_path2_2 = nn.Linear(out_features, out_features)
        self.bn_path2 = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        assert x.shape == (
            B,
            self.in_features,
        ), f"Expected shape ({B}, {self.in_features}), got {x.shape}"

        # Path 1
        x1 = self.fc_path1(x)
        x1 = self.bn_path1(x1)

        # Path 2
        x2 = self.fc_path2_1(x)
        x2 = self.fc_path2_2(x2)
        x2 = self.bn_path2(x2)

        x = x1 + x2
        assert x.shape == (
            B,
            self.out_features,
        ), f"Expected shape ({B}, {self.out_features}), got {x.shape}"
        return x
