import torch
import torch.nn as nn
from nerf_grasping.dexdiffuser.fc_resblock import FCResBlock


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
