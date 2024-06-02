import torch
import torch.nn as nn
from nerf_grasping.dexdiffuser.fc_resblock import FCResBlock


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
        cov_mcmc=None,
        dtype=torch.float32,
    ) -> None:
        super().__init__()

        self.bn1 = nn.BatchNorm1d(in_bps + in_grasp)
        self.rb1 = FCResBlock(in_bps + in_grasp, n_neurons)
        self.rb2 = FCResBlock(in_bps + in_grasp + n_neurons, n_neurons)
        self.rb3 = FCResBlock(in_bps + in_grasp + n_neurons, n_neurons)
        self.out_success = nn.Linear(n_neurons, 3)
        self.dout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

        self.dtype = torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if cov_mcmc is None:
            self.cov_mcmc = torch.diag(
                torch.tensor(
                    [0.005 ** 2] * 3  # translations
                    + [0.025 ** 2] * 6  # x and y axes
                    + [0.025 ** 2] * 16  # joint angles
                    + [0.025 ** 2] * 12,  # grasp directions
                    device=self.device,
                )
            )
        else:
            self.cov_mcmc = cov_mcmc

    def forward(self, f_O: torch.Tensor, g_O: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            f_O: The basis point set. Shape: (B, dim_BPS)
            g_O: The grasp features. Shape: (B, dim_grasp)
                We have dim_grasp = 3 + 6 + 16 + 3 * 4 = 37.
                The 6 rotation dims are the first two cols of the rot matrix.
                The extra 12 inputs are the grasp directions, which we provide to all.

        Returns:
            ys: The three labels for the grasp: y_coll, y_pick, y_eval.
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

    def _sample_proposal(self, g: torch.Tensor, cov: torch.Tensor, stage: str = "all") -> torch.Tensor:
        """Helper function to sample from the proposal distribution for MCMC.

        Going to use a Gaussian proposal distribution, but some of the grasp parameters
        are non-Euclidean and have compact support, so we need to do some projections.

        The grasp g is parameterized by:
        * trans (3D): the translation of the wrist
        * rot (6D): the first two columns of the rotation matrix of the wrist
        * q (16D): the joint angles of the Allegro hand
        * dirs (4 * 3D): the four closing directions of the hand

        Args:
            g: The current grasp. Shape: (B, dim_grasp)
            cov: The covariance matrix. This is the noise BEFORE adjustments are done to
                take into account support/topology. Shape: (B, dim_grasp, dim_grasp)
        """
        assert stage in ["all", "wrist_pose", "joint_angles", "dirs"], f"Invalid stage: {stage}"

        L = torch.linalg.cholesky(cov)  # (dim_grasp, dim_grasp)
        _g_prime = g + (L @ torch.randn_like(g)[..., None]).squeeze(-1)  # Shape: (B, dim_grasp)

        # fix the rotations such that the x and y vecs are orthogonal and unit norm
        if stage in ["all", "wrist_pose"]:
            _x_col = _g_prime[..., 3:6]  # Shape: (B, 3)
            x_col = _x_col / torch.norm(_x_col, dim=-1, keepdim=True)
            _y_col = _g_prime[..., 6:9]
            y_col = _y_col - torch.sum(_y_col * x_col, dim=-1, keepdim=True) * x_col
            y_col = y_col / torch.norm(y_col, dim=-1, keepdim=True)

            new_pose = torch.cat([_g_prime[..., :3], x_col, y_col], dim=-1)
        else:
            new_pose = g[..., :9]

        # fix the joint angles such that they are in the limits of the Allegro hand
        if stage in ["all", "joint_angles"]:
            lb = torch.tensor(
                [
                    -0.47,
                    -0.196,
                    -0.174,
                    -0.227,
                    -0.47,
                    -0.196,
                    -0.174,
                    -0.227,
                    -0.47,
                    -0.196,
                    -0.174,
                    -0.227,
                    0.263,
                    -0.105,
                    -0.189,
                    -0.162,
                ],
                device=self.device,
            )
            ub = torch.tensor(
                [
                    0.47,
                    1.61,
                    1.709,
                    1.618,
                    0.47,
                    1.61,
                    1.709,
                    1.618,
                    0.47,
                    1.61,
                    1.709,
                    1.618,
                    1.396,
                    1.163,
                    1.644,
                    1.719,
                ],
                device=self.device,
            )
            _q = _g_prime[..., 9:25]
            q = torch.clamp(_q, lb, ub)
        else:
            q = g[..., 9:25]

        # fix the dirs to all be unit norm
        if stage in ["all", "dirs"]:
            _dirs = _g_prime[..., 25:].reshape(-1, 4, 3)
            dirs = (_dirs / torch.norm(_dirs, dim=-1, keepdim=True)).view(-1, 12)
        else:
            dirs = g[..., 25:]

        # re-assign these to g_prime
        g_prime = torch.cat([new_pose, q, dirs.view(-1, 12)], dim=-1)
        return g_prime

    def refine(self, f_O: torch.Tensor, g_O: torch.Tensor, num_steps: int = 100, stage: str = "all") -> torch.Tensor:
        """Refine the grasp prediction using MCMC.

        Args:
            f_O: The basis point set. Shape: (B, dim_BPS)
            g_O: The grasp features. Shape: (B, dim_grasp)
                We have dim_grasp = 3 + 6 + 16 + 3 * 4 = 37.
                The 6 rotation dims are the first two cols of the rot matrix.
                The extra 12 inputs are the grasp directions, which we provide to all.

        Returns:
            g: The refined grasp prediction. Shape: (B, dim_grasp)
        """
        assert stage in ["all", "wrist_pose", "joint_angles", "dirs"], f"Invalid stage: {stage}"
        self.eval()  # removes non-deterministic effects from the evaluator

        g = g_O
        for _ in range(num_steps):
            g_prime = self._sample_proposal(g, cov=self.cov_mcmc, stage=stage)
            alpha = self(f_O, g_prime)[..., -1] / self(f_O, g)[..., -1]  # last label is the PGS
            u = torch.rand_like(alpha, device=self.device)
            mask = u <= alpha
            g = torch.where(mask[..., None], g_prime, g)
        return g

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "-" * 80)
    print("Testing DexEvaluator...")
    print("-" * 80)
    dex_evaluator = DexEvaluator(in_grasp=3 + 6 + 16 + 12, in_bps=4096).to(device)

    batch_size = 2
    f_O = torch.rand(batch_size, 4096).to(device)
    g_O = torch.rand(batch_size, 3 + 6 + 16 + 12).to(device)

    labels = dex_evaluator(f_O=f_O, g_O=g_O)
    PGS_0 = labels[..., -1]
    g_new = dex_evaluator.refine(f_O=f_O, g_O=g_O, num_steps=100, stage="all")
    PGS_new = dex_evaluator(f_O=f_O, g_O=g_new)[..., -1]
    print(f"PGS_0: {PGS_0}")
    print(f"PGS_new: {PGS_new}")
    breakpoint()

    assert output.shape == (
        batch_size,
        1,
    ), f"Expected shape ({batch_size}, 1), got {output.shape}"
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
