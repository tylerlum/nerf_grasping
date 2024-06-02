import torch
import torch.nn as nn

from nerf_grasping.dexdiffuser.fc_resblock import FCResBlock
from torch.nn.modules.activation import MultiheadAttention



class ResBlock(nn.Module):
    """A residual block that can optionally change the number of channels.

    Haofei said their ResBlock impl was similar to this:
    github.com/scenediffuser/Scene-Diffuser/blob/4a62ca30a4b37bb6d7b538e512905c570c4ded7c/models/model/utils.py#L32
    """

    def __init__(
        self,
        in_channels,
        emb_channels,
        dropout,
        out_channels=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = in_channels if out_channels is None else out_channels

        self.in_layers = nn.Sequential(
            nn.LayerNorm((self.in_channels, 1)),
            nn.SiLU(),
            nn.Conv1d(self.in_channels, self.out_channels, 1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.emb_channels, self.out_channels)
        )

        self.out_layers = nn.Sequential(
            nn.LayerNorm((self.out_channels, 1)),
            nn.SiLU(),
            nn.Dropout(p=self.dropout),
            nn.Conv1d(self.out_channels, self.out_channels, 1)
        )

        if self.out_channels == self.in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv1d(self.in_channels, self.out_channels, 1)

    def forward(self, x, emb):
        """Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        h = h + emb_out.unsqueeze(-1)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class DexSampler(nn.Module):
    """The DexSampler from the DexDiffuser paper.

    See: https://arxiv.org/pdf/2402.02989.
    """

    def __init__(self, dim_grasp: int = 37) -> None:
        """Initialize the sampler."""
        super().__init__()

        # query path
        self.res_block = ResBlock(
            in_channels=dim_grasp,  # grasp features
            emb_channels=1,  # time
            dropout=0.1,
            out_channels=512,  # we increase the channel dims to 512
        )
        self.sa = MultiheadAttention(
            embed_dim=512,
            num_heads=1,
            batch_first=True,
        )

        # key/value path
        self.fc_key = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
        )
        self.fc_value = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
        )

        # output path
        self.ca = MultiheadAttention(
            embed_dim=512,
            num_heads=1,
            batch_first=True,
        )
        self.fc_output = nn.Linear(512, dim_grasp)  # output the noise for the diffusion model

    def forward(self, f_O: torch.Tensor, g_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the sampler.

        :param f_O: the basis point set. Shape: (B, 4096).
        :param g_t: the grasp features. Shape: (B, 37).
        :param t: the timestep. Shape: (B, 1).
        :return: the noise for the diffusion model. Shape: (B, 37).
        """
        # key/value path
        f_O_seq = f_O.reshape((*f_O.shape[:-1], 8, 512))  # (B, 4096) -> (B, 8, 512)
        x_k = self.fc_key(f_O_seq)  # (B, 8, 512)
        x_v = self.fc_value(f_O_seq)  # (B, 8, 512)

        # query path
        x_q = self.res_block(g_t.unsqueeze(-1), t)  # (B, 512, 1)
        x_q = x_q.transpose(-1, -2)  # (B, 1, 512)

        # output path
        x, _ = self.sa(query=x_q, key=x_k, value=x_v)  # (B, 1, 512)
        x = x.squeeze(-2)  # (B, 512)
        eps = self.fc_output(x)  # (B, 37)  # output noise
        return eps


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "-" * 80)
    print("Testing DexSampler...")
    print("-" * 80)
    dex_sampler = DexSampler().to(device)
    batch_size = 2
    f_O = torch.rand(batch_size, 4096).to(device)
    g_t = torch.rand(batch_size, 37).to(device)
    t = torch.rand(batch_size, 1).to(device)
    output = dex_sampler(f_O=f_O, g_t=g_t, t=t)


if __name__ == "__main__":
    main()
