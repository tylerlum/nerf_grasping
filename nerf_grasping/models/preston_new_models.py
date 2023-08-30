import torch
import torch.nn as nn
from enum import Enum, auto
from functools import partial
from typing import List, Tuple


class ConvOutputTo1D(Enum):
    FLATTEN = auto()  # (N, C, H, W) -> (N, C*H*W)
    AVG_POOL_SPATIAL = auto()  # (N, C, H, W) -> (N, C, 1, 1) -> (N, C)
    AVG_POOL_CHANNEL = auto()  # (N, C, H, W) -> (N, 1, H, W) -> (N, H*W)
    MAX_POOL_SPATIAL = auto()  # (N, C, H, W) -> (N, C, 1, 1) -> (N, C)
    MAX_POOL_CHANNEL = auto()  # (N, C, H, W) -> (N, 1, H, W) -> (N, H*W)
    SPATIAL_SOFTMAX = auto()  # (N, C, H, W) -> (N, C, H, W) -> (N, 2*C)


class PoolType(Enum):
    MAX = auto()
    AVG = auto()


### Small Modules ###
class Mean(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=self.dim)


class Max(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(x, dim=self.dim)


class SpatialSoftmax(nn.Module):
    def __init__(self, temperature: float = 1.0, output_variance: bool = False) -> None:
        super().__init__()
        self.temperature = temperature
        self.output_variance = output_variance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Either (batch_size, n_channels, width) or (batch_size, n_channels, height, width)
        assert len(x.shape) in [3, 4]
        batch_size, n_channels = x.shape[:2]
        spatial_indices = [i for i in range(2, len(x.shape))]
        spatial_dims = x.shape[2:]

        # Softmax over spatial dimensions
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = nn.Softmax(dim=-1)(x / self.temperature)
        x = x.reshape(batch_size, n_channels, *spatial_dims)

        # Create spatial grid
        mesh_grids = torch.meshgrid(
            *[torch.linspace(-1, 1, dim, device=x.device) for dim in spatial_dims]
        )

        # Sanity check
        for mesh_grid in mesh_grids:
            assert mesh_grid.shape == spatial_dims

        # Get coords
        outputs = []
        for mesh_grid in mesh_grids:
            mesh_grid = mesh_grid.reshape(1, 1, *mesh_grid.shape)
            coord = torch.sum(x * mesh_grid, dim=spatial_indices)
            outputs.append(coord)

        # Get variance
        if self.output_variance:
            for mesh_grid in mesh_grids:
                mesh_grid = mesh_grid.reshape(1, 1, *mesh_grid.shape)
                coord = torch.sum(x * (mesh_grid**2), dim=spatial_indices)
                outputs.append(coord)

        # Stack
        outputs = torch.stack(outputs, dim=-1)
        expected_output_shape = (
            (batch_size, n_channels, len(spatial_dims))
            if not self.output_variance
            else (batch_size, n_channels, len(spatial_dims) * 2)
        )
        assert outputs.shape == expected_output_shape

        return outputs


CONV_2D_OUTPUT_TO_1D_MAP = {
    ConvOutputTo1D.FLATTEN: nn.Identity,
    ConvOutputTo1D.AVG_POOL_SPATIAL: partial(nn.AdaptiveAvgPool2d, output_size=(1, 1)),
    ConvOutputTo1D.AVG_POOL_CHANNEL: partial(Mean, dim=-3),
    ConvOutputTo1D.MAX_POOL_SPATIAL: partial(nn.AdaptiveMaxPool2d, output_size=(1, 1)),
    ConvOutputTo1D.MAX_POOL_CHANNEL: partial(Max, dim=-3),
    ConvOutputTo1D.SPATIAL_SOFTMAX: partial(
        SpatialSoftmax, temperature=1.0, output_variance=False
    ),
}
CONV_1D_OUTPUT_TO_1D_MAP = {
    ConvOutputTo1D.FLATTEN: nn.Identity,
    ConvOutputTo1D.AVG_POOL_SPATIAL: partial(nn.AdaptiveAvgPool1d, output_size=1),
    ConvOutputTo1D.AVG_POOL_CHANNEL: partial(Mean, dim=-2),
    ConvOutputTo1D.MAX_POOL_SPATIAL: partial(nn.AdaptiveMaxPool1d, output_size=1),
    ConvOutputTo1D.MAX_POOL_CHANNEL: partial(Max, dim=-2),
    ConvOutputTo1D.SPATIAL_SOFTMAX: partial(
        SpatialSoftmax, temperature=1.0, output_variance=False
    ),
}


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Build model
        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(
            zip([input_dim] + hidden_dims[:-1], hidden_dims)
        ):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, x):
        assert x.shape[-1] == self.input_dim
        for layer in self.layers:
            x = layer(x)
        return x


class FiLMLayer(nn.Module):
    """
    A PyTorch implementation of a FiLM layer.
    """

    def __init__(
        self,
        num_output_dims: int,
        in_channels: int,
        conditioning_dim: int,
        hidden_dims: List[int] = [32],
    ):
        super().__init__()
        self.num_output_dims = num_output_dims
        self.in_channels = in_channels
        self.conditioning_dim = conditioning_dim
        self.hidden_dims = hidden_dims

        # Build model
        self.gamma = MLP(
            conditioning_dim, hidden_dims, in_channels
        )  # Map conditioning dimension to scaling for each channel.
        self.beta = MLP(conditioning_dim, hidden_dims, in_channels)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor):
        # Compute FiLM parameters
        assert conditioning.shape[-1] == self.conditioning_dim
        batch_dims = conditioning.shape[:-1]

        assert x.shape[: -(self.num_output_dims + 1)] == batch_dims
        assert x.shape[-(self.num_output_dims + 1)] == self.in_channels

        gamma = self.gamma(conditioning)
        beta = self.beta(conditioning)

        # Do unsqueezing to make sure dimensions match; run e.g., twice for 2D FiLM.
        for _ in range(self.num_output_dims):
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)

        assert (
            gamma.shape
            == beta.shape
            == batch_dims + (self.in_channels,) + (1,) * self.num_output_dims
        )

        # Apply FiLM
        return gamma * x + beta


class CNN2DFiLM(nn.Module):
    """
    A vanilla 2D CNN with FiLM conditioning layers
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        conv_channels: List[int],
        conditioning_dim: int,
        num_in_channels: int,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.conv_channels = conv_channels
        self.conditioning_dim = conditioning_dim  # Note conditioning can only be 1D.
        self.num_in_channels = num_in_channels

        # Build model
        self.conv_layers = nn.ModuleList()
        for i, (in_channels, out_channels) in enumerate(
            zip([self.num_in_channels] + conv_channels[:-1], conv_channels)
        ):
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.BatchNorm2d(out_channels))
            self.conv_layers.append(FiLMLayer(2, out_channels, conditioning_dim))

        # Compute output shape
        with torch.no_grad():
            self.output_shape = self.get_output_shape()

    def get_output_shape(self):
        # Compute output shape
        x = torch.zeros((1, self.num_in_channels, *self.input_shape))
        for layer in self.conv_layers:
            if isinstance(layer, FiLMLayer):
                x = layer(x, torch.zeros((1, self.conditioning_dim)))
            else:
                x = layer(x)
        return x.shape[1:]

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor):
        assert x.shape[-2:] == self.input_shape
        assert x.shape[-3] == self.num_in_channels
        assert conditioning.shape[-1] == self.conditioning_dim

        # Reshape batch dims
        if len(x.shape) >= 4:
            batch_dims = x.shape[:-3]
            assert batch_dims == conditioning.shape[:-1]

            x = x.reshape(-1, *x.shape[-3:])
            conditioning = conditioning.reshape(-1, *conditioning.shape[-1:])

        for layer in self.conv_layers:
            if isinstance(layer, FiLMLayer):
                x = layer(x, conditioning)
            else:
                x = layer(x)

        # Reshape batch dims
        if len(x.shape) > 3:
            x = x.reshape(*batch_dims, *x.shape[-3:])

        return x


class CNN1DFiLM(nn.Module):
    def __init__(
        self,
        seq_len: int,
        conv_channels: List[int],
        conditioning_dim: int,
        num_in_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.conv_channels = conv_channels
        self.conditioning_dim = conditioning_dim
        self.kernel_size = kernel_size
        self.num_in_channels = num_in_channels

        # Build model
        self.conv_layers = nn.ModuleList()
        for i, (in_channels, out_channels) in enumerate(
            zip([self.num_in_channels] + conv_channels[:-1], conv_channels)
        ):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=self.kernel_size, padding=1
                )
            )
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.BatchNorm1d(out_channels))
            self.conv_layers.append(FiLMLayer(1, out_channels, conditioning_dim))
            breakpoint()

        # Compute output shape
        with torch.no_grad():
            self.output_shape = self.get_output_shape()

    def get_output_shape(self):
        # Compute output shape
        breakpoint()
        x = torch.zeros((1, self.num_in_channels, self.seq_len))
        for layer in self.conv_layers:
            if isinstance(layer, FiLMLayer):
                x = layer(x, torch.zeros((1, self.conditioning_dim)))
            else:
                x = layer(x)
        return x.shape[1:]

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor):
        assert x.shape[-1] == self.seq_len
        assert x.shape[-2] == self.num_in_channels
        assert conditioning.shape[-1] == self.conditioning_dim

        # Reshape batch dims
        if len(x.shape) >= 3:
            batch_dims = x.shape[:-2]
            assert batch_dims == conditioning.shape[:-1]
            x = x.reshape(-1, *x.shape[-2:])
            conditioning = conditioning.reshape(-1, *conditioning.shape[-1:])

        for layer in self.conv_layers:
            if isinstance(layer, FiLMLayer):
                x = layer(x, conditioning)
            else:
                x = layer(x)

        # Reshape batch dims
        if len(x.shape) > 2:
            x = x.reshape(*batch_dims, *x.shape[-2:])

        return x


if __name__ == "__main__":
    conditioning_dim = 7
    input_shape = (32, 32)
    conv2d_channels = [32, 64, 128]
    in_channels_2d = 1
    cnn2d_film = CNN2DFiLM(
        input_shape, conv2d_channels, conditioning_dim, in_channels_2d
    )
    print(cnn2d_film)

    cnn2d_output_shape = cnn2d_film.get_output_shape()
    assert cnn2d_output_shape == (conv2d_channels[-1], *input_shape)

    print(f"cnn2d_output_shape: {cnn2d_output_shape}")

    batch_size = (5, 10)

    conditioning = torch.rand(*batch_size, conditioning_dim)
    x = torch.rand(*batch_size, 1, *input_shape)
    print(f"x.shape: {x.shape}")
    print(f"conditioning.shape: {conditioning.shape}")

    out_2d = cnn2d_film(x, conditioning)
    print(f"out.shape: {out_2d.shape}")
