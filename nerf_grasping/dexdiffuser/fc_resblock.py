import torch
import torch.nn as nn


# class FCResBlock(nn.Module):
#     """FFHNet: https://ieeexplore.ieee.org/document/9811666:
#     The core building block of both models is the FC ResBlock, which consists of two parallel paths from input to output.
#     One path consists of a single FC layer, the other path has two FC layers. Each is followed by a layer of batch norm (BN).
#     It uses a leaky ReLU activation function.
#     """

#     def __init__(
#         self,
#         in_features: int,
#         out_features: int,
#         path2_hidden_features: int | None = None,
#     ) -> None:
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.path2_hidden_features = (
#             path2_hidden_features if path2_hidden_features else out_features
#         )

#         # leaky relu
#         self.relu = nn.LeakyReLU()  # no params

#         # Path 1
#         self.fc_path1 = nn.Linear(in_features, out_features)
#         self.bn_path1 = nn.BatchNorm1d(out_features)

#         # Path 2
#         self.fc_path2_1 = nn.Linear(in_features, path2_hidden_features)
#         self.bn_path2_1 = nn.BatchNorm1d(path2_hidden_features)
#         self.fc_path2_2 = nn.Linear(path2_hidden_features, out_features)
#         self.bn_path2_2 = nn.BatchNorm1d(out_features)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B = x.shape[0]
#         assert x.shape == (
#             B,
#             self.in_features,
#         ), f"Expected shape ({B}, {self.in_features}), got {x.shape}"

#         # Path 1
#         x1 = self.fc_path1(x)
#         x1 = self.bn_path1(x1)
#         x1 = self.relu(x1)

#         # Path 2
#         x2 = self.fc_path2_1(x)
#         x2 = self.bn_path2_1(x2)
#         x2 = self.relu(x2)

#         x2 = self.fc_path2_2(x2)
#         x2 = self.bn_path2_2(x2)
#         x2 = self.relu(x2)

#         x = x1 + x2
#         assert x.shape == (
#             B,
#             self.out_features,
#         ), f"Expected shape ({B}, {self.out_features}), got {x.shape}"
#         return x

class ResBlock(nn.Module):
    """The FFHNet ResBlock.
    
    See: github.com/qianbot/FFHNet/blob/main/FFHNet/models/networks.py#L78
    """

    def __init__(self, Fin, Fout, n_neurons=256):
        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout
