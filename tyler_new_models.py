import torch
import torch.nn as nn
from typing import List, Tuple, Optional

# from torchvision.models import resnet18, ResNet18_Weights
from FiLM_resnet import resnet18, ResNet18_Weights
from FiLM_resnet_1d import ResNet1D
from torchvision.transforms import Lambda, Compose
from enum import Enum, auto


class ConvOutputTo1D(Enum):
    FLATTEN = auto()  # (N, C, H, W) -> (N, C*H*W)
    AVG_POOL_SPATIAL = auto()  # (N, C, H, W) -> (N, C, 1, 1) -> (N, C)
    AVG_POOL_CHANNEL = auto()  # (N, C, H, W) -> (N, 1, H, W) -> (N, H*W)
    MAX_POOL_SPATIAL = auto()  # (N, C, H, W) -> (N, C, 1, 1) -> (N, C)
    MAX_POOL_CHANNEL = auto()  # (N, C, H, W) -> (N, 1, H, W) -> (N, H*W)


class PoolType(Enum):
    MAX = auto()
    AVG = auto()


def mlp(
    num_inputs: int,
    num_outputs: int,
    hidden_layers: List[int],
    activation=nn.ReLU,
    output_activation=nn.Identity,
) -> nn.Sequential:
    layers = []
    layer_sizes = [num_inputs] + hidden_layers + [num_outputs]
    for i in range(len(layer_sizes) - 1):
        act = activation if i < len(layer_sizes) - 2 else output_activation
        layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), act()]
    return nn.Sequential(*layers)


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


def conv_encoder(
    input_shape: Tuple[int, ...],
    conv_channels: List[int],
    pool_type: PoolType = PoolType.MAX,
    dropout_prob: float = 0.0,
    conv_output_to_1d: ConvOutputTo1D = ConvOutputTo1D.FLATTEN,
    activation=nn.ReLU,
) -> nn.Module:
    # Input: Either (n_channels, n_dims) or (n_channels, height, width) or (n_channels, depth, height, width)

    # Validate input
    assert 2 <= len(input_shape) <= 4
    n_input_channels = input_shape[0]
    n_spatial_dims = len(input_shape[1:])

    # Layers for different input sizes
    n_spatial_dims_to_conv_layer_map = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
    n_spatial_dims_to_maxpool_layer_map = {
        1: nn.MaxPool1d,
        2: nn.MaxPool2d,
        3: nn.MaxPool3d,
    }
    n_spatial_dims_to_avgpool_layer_map = {
        1: nn.AvgPool1d,
        2: nn.AvgPool2d,
        3: nn.AvgPool3d,
    }
    n_spatial_dims_to_dropout_layer_map = {
        # 1: nn.Dropout1d,  # Not in some versions of torch
        2: nn.Dropout2d,
        3: nn.Dropout3d,
    }
    n_spatial_dims_to_adaptivemaxpool_layer_map = {
        1: nn.AdaptiveMaxPool1d,
        2: nn.AdaptiveMaxPool2d,
        3: nn.AdaptiveMaxPool3d,
    }
    n_spatial_dims_to_adaptiveavgpool_layer_map = {
        1: nn.AdaptiveMaxPool1d,
        2: nn.AdaptiveMaxPool2d,
        3: nn.AdaptiveMaxPool3d,
    }

    # Setup layer types
    conv_layer = n_spatial_dims_to_conv_layer_map[n_spatial_dims]
    if pool_type == PoolType.MAX:
        pool_layer = n_spatial_dims_to_maxpool_layer_map[n_spatial_dims]
    elif pool_type == PoolType.AVG:
        pool_layer = n_spatial_dims_to_avgpool_layer_map[n_spatial_dims]
    else:
        raise ValueError(f"Invalid pool_type = {pool_type}")
    dropout_layer = n_spatial_dims_to_dropout_layer_map[n_spatial_dims]

    # Conv layers
    layers = []
    n_channels = [n_input_channels] + conv_channels
    for i in range(len(n_channels) - 1):
        layers += [
            conv_layer(
                in_channels=n_channels[i],
                out_channels=n_channels[i + 1],
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            activation(),
            pool_layer(kernel_size=2, stride=2),
        ]
        if dropout_prob != 0.0:
            layers += [dropout_layer(p=dropout_prob)]

    # Convert from (n_channels, X) => (Y,)
    if conv_output_to_1d == ConvOutputTo1D.FLATTEN:
        layers.append(nn.Flatten(start_dim=1))
    elif conv_output_to_1d == ConvOutputTo1D.AVG_POOL_SPATIAL:
        adaptiveavgpool_layer = n_spatial_dims_to_adaptiveavgpool_layer_map[
            n_spatial_dims
        ]
        layers.append(
            adaptiveavgpool_layer(output_size=tuple([1 for _ in range(n_spatial_dims)]))
        )
        layers.append(nn.Flatten(start_dim=1))
    elif conv_output_to_1d == ConvOutputTo1D.MAX_POOL_SPATIAL:
        adaptivemaxpool_layer = n_spatial_dims_to_adaptivemaxpool_layer_map[
            n_spatial_dims
        ]
        layers.append(
            adaptivemaxpool_layer(output_size=tuple([1 for _ in range(n_spatial_dims)]))
        )
        layers.append(nn.Flatten(start_dim=1))
    elif conv_output_to_1d == ConvOutputTo1D.AVG_POOL_CHANNEL:
        channel_dim = 1
        layers.append(Mean(dim=channel_dim))
        layers.append(nn.Flatten(start_dim=1))
    elif conv_output_to_1d == ConvOutputTo1D.MAX_POOL_CHANNEL:
        channel_dim = 1
        layers.append(Max(dim=channel_dim))
        layers.append(nn.Flatten(start_dim=1))
    else:
        raise ValueError(f"Invalid conv_output_to_1d = {conv_output_to_1d}")

    return nn.Sequential(*layers)


class FiLMGenerator(nn.Module):
    num_beta_gamma = 2  # one scale and one bias

    def __init__(
        self, film_input_dim: int, num_params_to_film: int, hidden_layers: List[int]
    ) -> None:
        super().__init__()
        self.film_input_dim = film_input_dim
        self.num_params_to_film = num_params_to_film
        self.film_output_dim = self.num_beta_gamma * num_params_to_film

        self.mlp = mlp(
            num_inputs=self.film_input_dim,
            num_outputs=self.film_output_dim,
            hidden_layers=hidden_layers,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(x.shape) == 2
        assert x.shape[1] == self.film_input_dim
        batch_size = x.shape[0]

        # Use delta-gamma so baseline is gamma=1
        film_output = self.mlp(x)
        beta, delta_gamma = torch.chunk(film_output, chunks=self.num_beta_gamma, dim=1)
        gamma = delta_gamma + 1.0
        assert beta.shape == gamma.shape == (batch_size, self.num_params_to_film)

        return beta, gamma


class ConvEncoder2D(nn.Module):
    """
    2D convolutional neural network for encoding 2D images. Given an input tensor of shape
    (batch_size, n_channels, height, width), the network applies a series of convolutional and pooling
    layers to extract features, and produces an output tensor of shape (batch_size, output_dim).

    Args:
        input_shape (tuple): Tuple of integers representing the shape of the input tensor
            (n_channels, height, width).

    Attributes:
        input_shape (tuple): Tuple of integers representing the shape of the input tensor
            (n_channels, height, width).
        output_dim (int): Number of output features produced by the network.

    Raises:
        AssertionError: If the length of input_shape is not 3 or if the number of input channels
            is not 1.

    Note:
        The final pooling method can be customized by modifying the network architecture. The
        current implementation uses ResNet18 or a simple convolutional encoder with max pooling.

    Examples:
        >>> encoder = ConvEncoder2D(input_shape=(1, 28, 28))
        >>> x = torch.randn(32, 1, 28, 28)
        >>> out = encoder(x)
        >>> print(out.shape)
        torch.Size([32, 512])
    """

    def __init__(self, input_shape: Tuple[int, int, int]) -> None:
        super().__init__()

        # input_shape: (n_channels, height, width)
        self.input_shape = input_shape

        assert len(input_shape) == 3
        n_channels, height, width = input_shape
        assert n_channels == 1

        # Create architecture
        # TODO: Customize final pooling method (avg pool, channel/spatial pool, spatial softmax, etc.)
        self.USE_RESNET = True
        self.USE_PRETRAINED = True
        if self.USE_RESNET:
            weights = ResNet18_Weights.DEFAULT if self.USE_PRETRAINED else None
            weights_transforms = weights.transforms() if self.USE_PRETRAINED else []
            # TODO: ensure transform is correct
            self.img_preprocess = Compose(
                [
                    Lambda(lambda x: x.repeat(1, 3, 1, 1)),
                    weights_transforms,
                ]
            )
            self.conv_2d = resnet18(weights=weights)
            self.conv_2d.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            self.conv_2d.fc = nn.Identity()
        else:
            self.img_preprocess = nn.Identity()
            self.conv_2d = conv_encoder(
                input_shape=input_shape,
                conv_channels=[32, 64, 128, 256],
                pool_type=PoolType.MAX,
                dropout_prob=0.0,
                conv_output_to_1d=ConvOutputTo1D.FLATTEN,
                activation=nn.ReLU,
            )

        # Compute output shape
        example_input = torch.randn(1, *input_shape)
        example_output = self.conv_2d(self.img_preprocess(example_input))
        assert len(example_output.shape) == 2
        self.output_dim = example_output.shape[1]
        self.num_film_params = self.conv_2d.num_film_params if self.USE_RESNET else None

    def forward(
        self,
        x: torch.Tensor,
        beta: Optional[torch.Tensor] = None,
        gamma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (batch_size, n_channels, height, width)
        assert len(x.shape) == 4
        batch_size, _, _, _ = x.shape

        x = self.img_preprocess(x)

        if beta is not None or gamma is not None:
            if self.num_film_params is None:
                raise ValueError("FiLM not supported for non-ResNet architecture")
            assert beta is None or beta.shape == (batch_size, self.num_film_params)
            assert gamma is None or gamma.shape == (batch_size, self.num_film_params)

        x = self.conv_2d(x, beta=beta, gamma=gamma)
        assert x.shape == (batch_size, self.output_dim)

        return x


class ConvEncoder1D(nn.Module):
    def __init__(self, input_shape: Tuple[int, int]) -> None:
        super().__init__()

        # input_shape: (n_channels, seq_len)
        self.input_shape = input_shape

        assert len(input_shape) == 2
        n_channels, seq_len = input_shape

        # Create architecture
        # TODO: Customize final pooling method (avg pool, channel/spatial pool, spatial softmax, etc.)
        self.USE_RESNET = True
        if self.USE_RESNET:
            self.conv_1d = ResNet1D(
                in_channels=n_channels,
                seq_len=seq_len,
                base_filters=64,
                kernel_size=3,
                stride=1,
                groups=1,
                n_block=4,
                n_classes=1,
                downsample_gap=2,
                increasefilter_gap=2,
                use_do=False,
                verbose=False,
            )
            # Set equivalent pooling setting
            self.conv_1d.avgpool = nn.AdaptiveAvgPool1d(output_size=1)
            self.conv_1d.fc = nn.Identity()
        else:
            self.conv_1d = conv_encoder(
                input_shape=input_shape,
                conv_channels=[32, 64, 128, 256],
                pool_type=PoolType.MAX,
                dropout_prob=0.0,
                conv_output_to_1d=ConvOutputTo1D.FLATTEN,
                activation=nn.ReLU,
            )

        # Compute output shape
        example_input = torch.randn(1, *input_shape)
        example_output = self.conv_1d(example_input)
        assert len(example_output.shape) == 2
        self.output_dim = example_output.shape[1]
        self.num_film_params = self.conv_1d.num_film_params if self.USE_RESNET else None

    def forward(
        self,
        x: torch.Tensor,
        beta: Optional[torch.Tensor] = None,
        gamma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (batch_size, n_channels, seq_len)
        assert len(x.shape) == 3
        batch_size, _, _ = x.shape

        if beta is not None or gamma is not None:
            if self.num_film_params is None:
                raise ValueError("FiLM not supported for non-ResNet architecture")
            assert beta is None or beta.shape == (batch_size, self.num_film_params)
            assert gamma is None or gamma.shape == (batch_size, self.num_film_params)

        x = self.conv_1d(x, beta=beta, gamma=gamma)
        assert x.shape == (batch_size, self.output_dim)

        return x


class Conv2Dto1D(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int, int]) -> None:
        super().__init__()
        self.input_shape = input_shape
        assert len(input_shape) == 4
        n_channels, depth, height, width = input_shape
        assert n_channels == 1

        seq_len = n_channels * depth

        self.n_channels_for_conv_2d = 1
        self.conv_encoder_2d = ConvEncoder2D(
            input_shape=(self.n_channels_for_conv_2d, height, width)
        )
        self.conv_encoder_1d = ConvEncoder1D(
            input_shape=(self.conv_encoder_2d.output_dim, seq_len)
        )
        self.output_dim = self.conv_encoder_1d.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 5
        batch_size, n_channels, depth, height, width = x.shape
        assert n_channels == 1

        seq_len = n_channels * depth

        # Conv 2d
        x = x.reshape(batch_size * seq_len, self.n_channels_for_conv_2d, height, width)
        x = self.conv_encoder_2d(x)
        assert x.shape == (batch_size * seq_len, self.conv_encoder_2d.output_dim)

        # Reshape to (batch_size, seq_len, output_dim)
        x = x.reshape(batch_size, seq_len, self.conv_encoder_2d.output_dim).permute(
            (0, 2, 1)
        )
        assert x.shape == (batch_size, self.conv_encoder_2d.output_dim, seq_len)

        # Conv 1d
        x = self.conv_encoder_1d(x)
        assert x.shape == (batch_size, self.output_dim)

        return x


class Conv2Dto1DFiLM(nn.Module):
    def __init__(
        self, input_shape: Tuple[int, int, int, int], film_input_dim: int
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.film_input_dim = film_input_dim
        assert len(input_shape) == 4
        n_channels, depth, height, width = input_shape
        assert n_channels == 1

        seq_len = n_channels * depth

        self.n_channels_for_conv_2d = 1
        self.conv_encoder_2d = ConvEncoder2D(
            input_shape=(self.n_channels_for_conv_2d, height, width)
        )
        self.conv_encoder_1d = ConvEncoder1D(
            input_shape=(self.conv_encoder_2d.output_dim, seq_len)
        )
        self.output_dim = self.conv_encoder_1d.output_dim

        self.film_generator_2d = FiLMGenerator(
            film_input_dim=film_input_dim,
            num_params_to_film=self.conv_encoder_2d.num_film_params,
            hidden_layers=[64, 64],
        )
        self.film_generator_1d = FiLMGenerator(
            film_input_dim=film_input_dim,
            num_params_to_film=self.conv_encoder_1d.num_film_params,
            hidden_layers=[64, 64],
        )

    def forward(self, x: torch.Tensor, film_input: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 5
        batch_size, n_channels, depth, height, width = x.shape
        assert n_channels == 1

        assert film_input.shape == (batch_size, self.film_generator_2d.film_input_dim)

        # Conv 2d
        beta_2d, gamma_2d = self.film_generator_2d(film_input)
        assert (
            beta_2d.shape
            == gamma_2d.shape
            == (batch_size, self.conv_encoder_2d.num_film_params)
        )

        seq_len = n_channels * depth
        x = x.reshape(batch_size * seq_len, self.n_channels_for_conv_2d, height, width)
        beta_2d = (
            beta_2d.reshape(batch_size, 1, self.conv_encoder_2d.num_film_params)
            .repeat(1, seq_len, 1)
            .reshape(batch_size * seq_len, self.conv_encoder_2d.num_film_params)
        )
        gamma_2d = (
            gamma_2d.reshape(batch_size, 1, self.conv_encoder_2d.num_film_params)
            .repeat(1, seq_len, 1)
            .reshape(batch_size * seq_len, self.conv_encoder_2d.num_film_params)
        )

        x = self.conv_encoder_2d(x, beta=beta_2d, gamma=gamma_2d)
        assert x.shape == (batch_size * seq_len, self.conv_encoder_2d.output_dim)

        # Reshape to (batch_size, seq_len, output_dim)
        x = x.reshape(batch_size, seq_len, self.conv_encoder_2d.output_dim).permute(
            (0, 2, 1)
        )
        assert x.shape == (batch_size, self.conv_encoder_2d.output_dim, seq_len)

        # Conv 1d
        beta_1d, gamma_1d = self.film_generator_1d(film_input)
        assert (
            beta_1d.shape
            == gamma_1d.shape
            == (batch_size, self.conv_encoder_1d.num_film_params)
        )
        x = self.conv_encoder_1d(x, beta=beta_1d, gamma=gamma_1d)
        assert x.shape == (batch_size, self.output_dim)

        return x


if __name__ == "__main__":
    from torchinfo import summary

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # batch_size, n_channels, depth, height, width = 5, 1, 81, 30, 40
    batch_size, n_channels, depth, height, width = 5, 1, 10, 30, 40
    conv_2d_to_1d = Conv2Dto1D(input_shape=(n_channels, depth, height, width)).to(
        device
    )

    example_input = torch.randn(
        batch_size, n_channels, depth, height, width, device=device
    )
    example_output = conv_2d_to_1d(example_input)
    print("Conv2Dto1D")
    print("=" * 80)
    print(f"example_input.shape = {example_input.shape}")
    print(f"example_output.shape = {example_output.shape}")
    print()

    # Create model 2
    film_input_dim = 10
    conv_2d_to_1d_film = Conv2Dto1DFiLM(
        input_shape=(n_channels, depth, height, width), film_input_dim=film_input_dim
    ).to(device)
    example_film_input = torch.randn(batch_size, film_input_dim, device=device)
    example_film_output = conv_2d_to_1d_film(example_input, example_film_input)
    print("Conv2Dto1DFiLM")
    print("=" * 80)
    print(f"example_input.shape = {example_input.shape}")
    print(f"example_film_input.shape = {example_film_input.shape}")
    print(f"example_film_output.shape = {example_film_output.shape}")
    print()
