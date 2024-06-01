import torch
import torch.nn as nn
from nerf_grasping.dexdiffuser.fc_resblock import ResBlock


class DexEvaluator(nn.Module):
    """The DexDiffuser evaluator module.
    
    Adapted for use in our repo.

    See: https://github.com/qianbot/FFHNet/blob/4aa38dd6bd59bcf4b794ca872f409844579afa9f/FFHNet/models/networks.py#L243
    """

    def __init__(
        self,
        in_grasp,
        n_neurons=512,
        in_bps=4096,
        dtype=torch.float32,
    ) -> None:
        super().__init__()

        self.bn1 = nn.BatchNorm1d(in_bps + in_grasp)
        self.rb1 = ResBlock(in_bps + in_grasp, n_neurons)
        self.rb2 = ResBlock(in_bps + in_grasp + n_neurons, n_neurons)
        self.rb3 = ResBlock(in_bps + in_grasp + n_neurons, n_neurons)
        self.out_success = nn.Linear(n_neurons, 1)
        self.dout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

        self.dtype = torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, f_O: torch.Tensor, g_O: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            f_O: The basis point set. Shape: (B, dim_BPS)
            g_O: The grasp features. Shape: (B, dim_grasp)
                For the DexDiffuser baseline, dim_grasp = 3 + 6 + 16 = 25.
                For the GG-NeRF ablation, dim_grasp = 3 + 6 + 16 + 3 * 4 = 37.
                The 6 rotation dims are the first two cols of the rot matrix.
                The extra 12 inputs for GG-NeRF are the grasp directions.
        """
        X = torch.cat([f_O, g_O], dim=-1)

        X0 = self.bn1(X)
        X = self.rb1(X0)
        X = self.dout(X)
        X = self.rb2(torch.cat([X, X0], dim=1))
        X = self.dout(X)
        X = self.rb3(torch.cat([X, X0], dim=1))
        X = self.dout(X)
        X = self.out_success(X)
        p_success = self.sigmoid(X)
        return p_success


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
