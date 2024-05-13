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


class DexEvaluator(nn.Module):
    """DexDiffuser: https://arxiv.org/pdf/2402.02989
    The architecture of DexEvaluator is adopted from [20] (FFHNet)

    FFHNet: https://ieeexplore.ieee.org/document/9811666:
    three FC Resblocks for the FFHEvaluator achieved the best performance
    The inputs of the encoder (xb,θ,R,t) and the conditional input xb are also pre-processed by BN.
    The output of the FFHEvaluator's final layer is fed through a sigmoid activation function.
    """

    def __init__(self, n_pts: int, grasp_dim: int) -> None:
        super().__init__()
        self.n_pts = n_pts
        self.grasp_dim = grasp_dim

        self.n_inputs = n_pts + grasp_dim
        n_hidden = self.n_inputs  # Must be same for skip connections

        self.bn = nn.BatchNorm1d(self.n_inputs)
        self.fc_resblock_1 = FCResBlock(
            in_features=self.n_inputs, out_features=n_hidden
        )
        self.fc_resblock_2 = FCResBlock(in_features=n_hidden, out_features=n_hidden)
        self.fc_resblock_3 = FCResBlock(in_features=n_hidden, out_features=n_hidden)
        self.fc_out = nn.Linear(n_hidden, 1)

    def forward(self, f_O: torch.Tensor, g_0: torch.Tensor) -> torch.Tensor:
        B = f_O.shape[0]
        assert f_O.shape == (
            B,
            self.n_pts,
        ), f"Expected shape ({B}, {self.n_pts}), got {f_O.shape}"
        assert g_0.shape == (
            B,
            self.grasp_dim,
        ), f"Expected shape ({B}, {self.grasp_dim}), got {g_0.shape}"

        # Concat and batch norm
        x = torch.cat([f_O, g_0], dim=1)
        x = self.bn(x)

        # Resblocks
        x = self.fc_resblock_1(x) + x
        x = self.fc_resblock_2(x) + x
        x = self.fc_resblock_3(x) + x

        # Output
        x = self.fc_out(x)
        x = torch.sigmoid(x)
        assert x.shape == (B, 1), f"Expected shape ({B}, 1), got {x.shape}"
        return x


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "-" * 80)
    print("Testing DexEvaluator...")
    print("-" * 80)
    dex_evaluator = DexEvaluator(n_pts=4096, grasp_dim=3 + 6 + 16).to(device)

    batch_size = 2
    f_O = torch.rand(batch_size, 4096).to(device)
    g_0 = torch.rand(batch_size, 3 + 6 + 16).to(device)

    output = dex_evaluator(f_O=f_O, g_0=g_0)

    assert output.shape == (
        batch_size,
        1,
    ), f"Expected shape ({batch_size}, 1), got {output.shape}"
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
