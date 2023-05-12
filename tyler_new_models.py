import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from omegaconf import MISSING
from FiLM_resnet import resnet18, ResNet18_Weights
from FiLM_resnet_1d import ResNet1D
from torchvision.transforms import Lambda, Compose
from enum import Enum, auto
from functools import partial, cached_property
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from torchinfo import summary


def dataclass_to_kwargs(dataclass_instance: Any) -> Dict[str, Any]:
    return (
        {key: value for key, value in dataclass_instance.__dict__["_content"].items()}
        if dataclass_instance is not None
        else {}
    )


### ENUMS ###
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


CHANNEL_DIM = 1
CONV_2D_OUTPUT_TO_1D_MAP = {
    ConvOutputTo1D.FLATTEN: nn.Identity,
    ConvOutputTo1D.AVG_POOL_SPATIAL: partial(nn.AdaptiveAvgPool2d, output_size=(1, 1)),
    ConvOutputTo1D.AVG_POOL_CHANNEL: partial(Mean, dim=CHANNEL_DIM),
    ConvOutputTo1D.MAX_POOL_SPATIAL: partial(nn.AdaptiveMaxPool2d, output_size=(1, 1)),
    ConvOutputTo1D.MAX_POOL_CHANNEL: partial(Max, dim=CHANNEL_DIM),
    ConvOutputTo1D.SPATIAL_SOFTMAX: partial(
        SpatialSoftmax, temperature=1.0, output_variance=False
    ),
}
CONV_1D_OUTPUT_TO_1D_MAP = {
    ConvOutputTo1D.FLATTEN: nn.Identity,
    ConvOutputTo1D.AVG_POOL_SPATIAL: partial(nn.AdaptiveAvgPool1d, output_size=1),
    ConvOutputTo1D.AVG_POOL_CHANNEL: partial(Mean, dim=CHANNEL_DIM),
    ConvOutputTo1D.MAX_POOL_SPATIAL: partial(nn.AdaptiveMaxPool1d, output_size=1),
    ConvOutputTo1D.MAX_POOL_CHANNEL: partial(Max, dim=CHANNEL_DIM),
    ConvOutputTo1D.SPATIAL_SOFTMAX: partial(
        SpatialSoftmax, temperature=1.0, output_variance=False
    ),
}


### HELPER FUNCTIONS ###
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


### 2D ENCODERS ###


@dataclass
class ConvEncoder2DConfig:
    use_resnet: bool = MISSING
    use_pretrained: bool = MISSING
    pooling_method: ConvOutputTo1D = MISSING
    film_hidden_layers: List[int] = MISSING


class ConvEncoder2D(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        conditioning_dim: Optional[int] = None,
        use_resnet: bool = True,
        use_pretrained: bool = True,
        pooling_method: ConvOutputTo1D = ConvOutputTo1D.FLATTEN,
        film_hidden_layers: List[int] = [64, 64],
    ) -> None:
        super().__init__()

        # input_shape: (n_channels, height, width)
        self.input_shape = input_shape
        self.conditioning_dim = conditioning_dim
        self.use_resnet = use_resnet
        self.use_pretrained = use_pretrained
        self.pooling_method = pooling_method

        assert len(input_shape) == 3
        n_channels, height, width = input_shape
        assert n_channels == 1

        # Create conv architecture
        if self.use_resnet:
            weights = ResNet18_Weights.DEFAULT if self.use_pretrained else None
            weights_transforms = [weights.transforms()] if self.use_pretrained else []
            self.img_preprocess = Compose(
                [Lambda(lambda x: x.repeat(1, 3, 1, 1))] + weights_transforms
            )
            self.conv_2d = resnet18(weights=weights)
            self.conv_2d.avgpool = CONV_2D_OUTPUT_TO_1D_MAP[self.pooling_method]()
            self.conv_2d.fc = nn.Identity()
        else:
            raise NotImplementedError("TODO: Implement non-resnet conv encoder")
            # TODO: Properly config
            self.img_preprocess = nn.Identity()
            self.conv_2d = conv_encoder(
                input_shape=input_shape,
                conv_channels=[32, 64, 128, 256],
                pool_type=PoolType.MAX,
                dropout_prob=0.0,
                conv_output_to_1d=self.pooling_method,
                activation=nn.ReLU,
            )

        # Create FiLM generator
        if self.conditioning_dim is not None and self.num_film_params is not None:
            self.film_generator = FiLMGenerator(
                film_input_dim=self.conditioning_dim,
                num_params_to_film=self.num_film_params,
                hidden_layers=film_hidden_layers,
            )

    def forward(
        self,
        x: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (batch_size, n_channels, height, width)
        assert len(x.shape) == 4
        batch_size, _, _, _ = x.shape

        # Ensure valid use of conditioning
        assert (conditioning is None and self.conditioning_dim is None) or (
            conditioning is not None
            and self.conditioning_dim is not None
            and conditioning.shape == (batch_size, self.conditioning_dim)
        )

        # FiLM
        if conditioning is not None:
            beta, gamma = self.film_generator(conditioning)
            assert (
                beta.shape == gamma.shape == (batch_size, self.conv_2d.num_film_params)
            )
        else:
            beta, gamma = None, None

        # Conv
        x = self.img_preprocess(x)
        x = self.conv_2d(x, beta=beta, gamma=gamma)
        assert len(x.shape) == 2 and x.shape[0] == batch_size

        return x

    @property
    def num_film_params(self) -> Optional[int]:
        return (
            self.conv_2d.num_film_params
            if hasattr(self.conv_2d, "num_film_params")
            else None
        )

    @cached_property
    def output_dim(self) -> int:
        # Compute output shape
        example_batch_size = 2
        example_input = torch.randn(example_batch_size, *self.input_shape)
        example_conditioning = (
            torch.randn(example_batch_size, self.conditioning_dim)
            if self.conditioning_dim is not None
            else None
        )
        example_output = self(example_input, conditioning=example_conditioning)
        assert len(example_output.shape) == 2
        return example_output.shape[1]


### 1D ENCODERS ###


@dataclass
class ConvEncoder1DConfig:
    use_resnet: bool = MISSING
    pooling_method: ConvOutputTo1D = MISSING
    film_hidden_layers: List[int] = MISSING
    base_filters: int = MISSING
    kernel_size: int = MISSING
    stride: int = MISSING
    groups: int = MISSING
    n_block: int = MISSING
    downsample_gap: int = MISSING
    increasefilter_gap: int = MISSING
    use_do: bool = MISSING


class ConvEncoder1D(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        conditioning_dim: Optional[int] = None,
        use_resnet: bool = True,
        pooling_method: ConvOutputTo1D = ConvOutputTo1D.FLATTEN,
        film_hidden_layers: List[int] = [64, 64],
        base_filters: int = 64,
        kernel_size: int = 16,
        stride: int = 2,
        groups: int = 32,
        n_block: int = 8,
        downsample_gap: int = 6,
        increasefilter_gap: int = 12,
        use_do: bool = False,
    ) -> None:
        super().__init__()

        # input_shape: (n_channels, seq_len)
        self.input_shape = input_shape
        self.conditioning_dim = conditioning_dim
        self.use_resnet = use_resnet
        self.pooling_method = pooling_method

        assert len(input_shape) == 2
        n_channels, seq_len = input_shape

        # Create conv architecture
        if self.use_resnet:
            # TODO: Properly config
            self.conv_1d = ResNet1D(
                in_channels=n_channels,
                seq_len=seq_len,
                base_filters=base_filters,
                kernel_size=kernel_size,
                stride=stride,
                groups=groups,
                n_block=n_block,
                n_classes=2,  # Not used
                downsample_gap=downsample_gap,
                increasefilter_gap=increasefilter_gap,
                use_do=use_do,
                verbose=False,
            )
            # Set equivalent pooling setting
            self.conv_1d.avgpool = CONV_1D_OUTPUT_TO_1D_MAP[self.pooling_method]()
            self.conv_1d.fc = nn.Identity()
        else:
            raise NotImplementedError("TODO: Implement non-resnet conv encoder")
            # TODO: Properly config
            self.conv_1d = conv_encoder(
                input_shape=input_shape,
                conv_channels=[32, 64, 128, 256],
                pool_type=PoolType.MAX,
                dropout_prob=0.0,
                conv_output_to_1d=ConvOutputTo1D.FLATTEN,
                activation=nn.ReLU,
            )

        # Create FiLM generator
        if self.conditioning_dim is not None and self.num_film_params is not None:
            # TODO: Properly config
            self.film_generator = FiLMGenerator(
                film_input_dim=self.conditioning_dim,
                num_params_to_film=self.num_film_params,
                hidden_layers=film_hidden_layers,
            )

    def forward(
        self,
        x: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (batch_size, n_channels, seq_len)
        assert len(x.shape) == 3
        batch_size, _, _ = x.shape

        # Ensure valid use of conditioning
        assert (conditioning is None and self.conditioning_dim is None) or (
            conditioning is not None
            and self.conditioning_dim is not None
            and conditioning.shape == (batch_size, self.conditioning_dim)
        )

        # FiLM
        if conditioning is not None:
            beta, gamma = self.film_generator(conditioning)
            assert (
                beta.shape == gamma.shape == (batch_size, self.conv_1d.num_film_params)
            )
        else:
            beta, gamma = None, None

        # Conv
        x = self.conv_1d(x, beta=beta, gamma=gamma)
        assert len(x.shape) == 2 and x.shape[0] == batch_size

        return x

    @property
    def num_film_params(self) -> Optional[int]:
        return self.conv_1d.num_film_params if self.use_resnet else None

    @cached_property
    def output_dim(self) -> int:
        # Compute output shape
        example_batch_size = 2
        example_input = torch.randn(example_batch_size, *self.input_shape)
        example_conditioning = (
            torch.randn(example_batch_size, self.conditioning_dim)
            if self.conditioning_dim is not None
            else None
        )
        example_output = self(example_input, conditioning=example_conditioning)
        assert len(example_output.shape) == 2
        return example_output.shape[1]


@dataclass
class TransformerEncoder1DConfig:
    pooling_method: ConvOutputTo1D = MISSING
    n_heads: int = MISSING
    n_emb: int = MISSING
    p_drop_emb: float = MISSING
    p_drop_attn: float = MISSING
    n_layers: int = MISSING


class TransformerEncoder1D(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        conditioning_dim: Optional[int] = None,
        pooling_method: ConvOutputTo1D = ConvOutputTo1D.FLATTEN,
        n_heads: int = 8,
        n_emb: int = 128,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        n_layers: int = 4,
    ) -> None:
        super().__init__()

        # input_shape: (n_channels, seq_len)
        self.input_shape = input_shape
        self.conditioning_dim = conditioning_dim
        self.pooling_method = pooling_method
        self.n_emb = n_emb

        n_channels, seq_len = input_shape

        # Encoder
        self.encoder_input_emb = nn.Linear(self.encoder_input_dim, n_emb)
        self.encoder_pos_emb = Summer(PositionalEncoding1D(channels=n_emb))
        self.encoder_drop_emb = nn.Dropout(p=p_drop_emb)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb,
            nhead=n_heads,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=n_layers
        )

        if conditioning_dim is not None:
            # Decoder
            self.decoder_input_emb = nn.Linear(n_channels, n_emb)
            self.decoder_pos_emb = Summer(PositionalEncoding1D(channels=n_emb))
            self.decoder_drop_emb = nn.Dropout(p=p_drop_emb)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_heads,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer_decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer, num_layers=n_layers
            )

        self.ln_f = nn.LayerNorm(n_emb)
        self.pool = CONV_1D_OUTPUT_TO_1D_MAP[self.pooling_method]()

    def forward(
        self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        n_channels, seq_len = self.input_shape
        assert x.shape == (batch_size, n_channels, seq_len)

        assert (conditioning is not None and self.conditioning_dim is not None) or (
            conditioning is None and self.conditioning_dim is None
        )
        if conditioning is not None and self.conditioning_dim is not None:
            assert conditioning.shape == (batch_size, self.conditioning_dim)

            # Condition encoder
            # Need to repeat conditioning to match seq_len
            conditioning = conditioning.reshape(
                batch_size,
                1,
                self.conditioning_dim,
            ).repeat(1, seq_len, 1)
            assert conditioning.shape == (
                batch_size,
                seq_len,
                self.conditioning_dim,
            )
            conditioning = self._encoder(conditioning)
            assert conditioning.shape == (batch_size, seq_len, self.n_emb)

            # Decoder
            x = x.permute(0, 2, 1)
            assert x.shape == (batch_size, seq_len, n_channels)
            x = self._decoder(x, conditioning)
            assert x.shape == (batch_size, seq_len, self.n_emb)
        else:
            # Encoder
            x = x.permute(0, 2, 1)
            assert x.shape == (batch_size, seq_len, n_channels)
            x = self._encoder(x)
            assert x.shape == (batch_size, seq_len, self.n_emb)

        x = self.ln_f(x)
        assert x.shape == (batch_size, seq_len, self.n_emb)

        # Need to permute to (batch_size, n_channels, seq_len) for pooling
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.flatten(start_dim=1)

        assert len(x.shape) == 2 and x.shape[0] == batch_size

        return x

    def _encoder(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        _, seq_len = self.input_shape
        assert x.shape == (batch_size, seq_len, self.encoder_input_dim)

        x = self.encoder_input_emb(x)
        x = self.encoder_pos_emb(x)
        x = self.encoder_drop_emb(x)
        x = self.transformer_encoder(x)

        return x

    def _decoder(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        n_channels, seq_len = self.input_shape
        assert x.shape == (batch_size, seq_len, n_channels)

        x = self.decoder_input_emb(x)
        x = self.decoder_pos_emb(x)
        x = self.decoder_drop_emb(x)
        x = self.transformer_decoder(tgt=x, memory=conditioning)

        return x

    @cached_property
    def output_dim(self) -> int:
        # Compute output shape
        example_batch_size = 2
        example_input = torch.randn(example_batch_size, *self.input_shape)
        example_conditioning = (
            torch.randn(example_batch_size, self.conditioning_dim)
            if self.conditioning_dim is not None
            else None
        )
        example_output = self(example_input, conditioning=example_conditioning)
        assert len(example_output.shape) == 2
        return example_output.shape[1]

    @property
    def encoder_input_dim(self) -> int:
        n_channels, _ = self.input_shape
        return (
            self.conditioning_dim if self.conditioning_dim is not None else n_channels
        )


### Attention Encoder Decoder ###
class TransformerEncoderDecoder(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        pooling_method: ConvOutputTo1D = ConvOutputTo1D.FLATTEN,
        n_emb: int = 128,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        n_layers: int = 4,
    ) -> None:
        super().__init__()

        # input_shape: (n_channels, seq_len)
        self.input_shape = input_shape
        self.pooling_method = pooling_method
        self.n_emb = n_emb

        n_channels, seq_len = input_shape

        # Encoder
        self.encoder_input_emb = nn.Linear(n_channels, n_emb)
        self.encoder_pos_emb = Summer(PositionalEncoding1D(channels=n_emb))
        self.encoder_drop_emb = nn.Dropout(p=p_drop_emb)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb,
            nhead=4,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=n_layers
        )

        # Decoder
        self.decoder_input_emb = nn.Linear(n_channels, n_emb)
        self.decoder_pos_emb = Summer(PositionalEncoding1D(channels=n_emb))
        self.decoder_drop_emb = nn.Dropout(p=p_drop_emb)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=4,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=n_layers
        )

        self.ln_f = nn.LayerNorm(n_emb)
        self.pool = CONV_1D_OUTPUT_TO_1D_MAP[self.pooling_method]()

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        batch_size = conditioning.shape[0]
        n_channels, seq_len = self.input_shape
        assert conditioning.shape == x.shape == (batch_size, n_channels, seq_len)

        conditioning = conditioning.permute(0, 2, 1)
        assert conditioning.shape == (batch_size, seq_len, n_channels)
        conditioning = self._encoder(conditioning)
        assert conditioning.shape == (batch_size, seq_len, self.n_emb)

        # Decoder
        x = x.permute(0, 2, 1)
        assert x.shape == (batch_size, seq_len, n_channels)
        x = self._decoder(x, conditioning=conditioning)
        assert x.shape == (batch_size, seq_len, self.n_emb)

        x = self.ln_f(x)
        assert x.shape == (batch_size, seq_len, self.n_emb)

        # Need to permute to (batch_size, n_channels, seq_len) for pooling
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.flatten(start_dim=1)

        assert len(x.shape) == 2 and x.shape[0] == batch_size

        return x

    def _encoder(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        n_channels, seq_len = self.input_shape
        assert x.shape == (batch_size, seq_len, n_channels)

        x = self.encoder_input_emb(x)
        x = self.encoder_pos_emb(x)
        x = self.encoder_drop_emb(x)
        x = self.transformer_encoder(x)

        return x

    def _decoder(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        n_channels, seq_len = self.input_shape
        assert x.shape == (batch_size, seq_len, n_channels)

        x = self.decoder_input_emb(x)
        x = self.decoder_pos_emb(x)
        x = self.decoder_drop_emb(x)
        x = self.transformer_decoder(tgt=x, memory=conditioning)

        return x

    @cached_property
    def output_dim(self) -> int:
        # Compute output shape
        example_batch_size = 2
        example_input = torch.randn(example_batch_size, *self.input_shape)
        example_conditioning = torch.randn(example_batch_size, *self.input_shape)
        example_output = self(example_input, conditioning=example_conditioning)
        assert len(example_output.shape) == 2
        return example_output.shape[1]


### Classifiers ###


class Encoder1DType(Enum):
    CONV = auto()
    TRANSFORMER = auto()


@dataclass
class ClassifierConfig:
    conv_encoder_2d_config: ConvEncoder2DConfig = MISSING
    use_conditioning_2d: bool = MISSING
    conv_encoder_2d_embed_dim: int = MISSING
    conv_encoder_2d_mlp_hidden_layers: List[int] = MISSING

    conv_encoder_1d_config: ConvEncoder1DConfig = MISSING
    transformer_encoder_1d_config: TransformerEncoder1DConfig = MISSING
    encoder_1d_type: Encoder1DType = MISSING
    use_conditioning_1d: bool = MISSING
    head_mlp_hidden_layers: List[int] = MISSING


class Abstract2DTo1DClassifier(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        n_fingers: int,
        conditioning_dim: Optional[int] = None,
        conv_encoder_2d_config: Optional[ConvEncoder2DConfig] = None,
        use_conditioning_2d: bool = False,
        conv_encoder_2d_embed_dim: int = 32,
        conv_encoder_2d_mlp_hidden_layers: List[int] = [64, 64],
    ) -> None:
        # input_shape: (seq_len, height, width)
        super().__init__()
        assert len(input_shape) == 3

        # Store args
        self.input_shape = input_shape
        self.n_fingers = n_fingers
        self.conditioning_dim = conditioning_dim
        self.conv_encoder_2d_config = conv_encoder_2d_config
        self.use_conditioning_2d = use_conditioning_2d
        self.conv_encoder_2d_embed_dim = conv_encoder_2d_embed_dim
        self.conv_encoder_2d_mlp_hidden_layers = conv_encoder_2d_mlp_hidden_layers

        # Handle shapes
        self.n_channels_for_conv_2d = 1
        self.seq_len, self.height, self.width = input_shape

        # 2D
        conv_encoder_2d_input_shape = (
            self.n_channels_for_conv_2d,
            self.height,
            self.width,
        )
        self.conv_encoder_2d = ConvEncoder2D(
            input_shape=conv_encoder_2d_input_shape,
            conditioning_dim=conditioning_dim if use_conditioning_2d else None,
            # **dataclass_to_kwargs(conv_encoder_2d_config)
            use_resnet=conv_encoder_2d_config.use_resnet,
            use_pretrained=conv_encoder_2d_config.use_pretrained,
            pooling_method=conv_encoder_2d_config.pooling_method,
            film_hidden_layers=conv_encoder_2d_config.film_hidden_layers,
        )
        self.fc = mlp(
            num_inputs=self.conv_encoder_2d.output_dim,
            num_outputs=conv_encoder_2d_embed_dim,
            hidden_layers=conv_encoder_2d_mlp_hidden_layers,
        )

        self._prepare_1d_encoder()

    def forward(
        self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self._check_valid_input_shape(x, conditioning=conditioning)

        # Conv 2d (batch_size, n_fingers, seq_len, height, width) => (batch_size, n_fingers, seq_len, conv_encoder_2d_embed_dim)
        x = self._run_conv_encoder_2d(x, conditioning=conditioning)

        # Encoder 1d (batch_size, n_fingers, seq_len, conv_encoder_2d_embed_dim) => (batch_size, 2)
        x = self._run_1d_encoder(x, conditioning=conditioning)

        return x

    def _check_valid_input_shape(
        self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None
    ) -> None:
        assert len(x.shape) == 5
        batch_size = x.shape[0]
        assert x.shape == (
            batch_size,
            self.n_fingers,
            self.seq_len,
            self.height,
            self.width,
        )
        assert conditioning is None or conditioning.shape == (
            batch_size,
            self.n_fingers,
            self.conditioning_dim,
        )

    def _run_conv_encoder_2d(
        self,
        x: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        AVOID_OOM=True,
        RUN_SANITY_CHECK=False,
    ) -> torch.Tensor:
        self._check_valid_input_shape(x, conditioning=conditioning)
        batch_size = x.shape[0]

        if RUN_SANITY_CHECK:
            self.__run_sanity_check(x, conditioning=conditioning)

        # Conv 2d (batch_size, n_fingers, seq_len, height, width) => (batch_size, n_fingers, seq_len, conv_encoder_2d_embed_dim)
        # May OOM if do all at once
        if AVOID_OOM:
            x = self.__run_conv_encoder_2d_batch_fingers(x, conditioning=conditioning)
        else:
            x = self.__run_conv_encoder_2d_batch_fingers_seq(
                x, conditioning=conditioning
            )

        assert x.shape == (
            batch_size,
            self.n_fingers,
            self.seq_len,
            self.conv_encoder_2d.output_dim,
        )

        x = self.fc(x)

        assert x.shape == (
            batch_size,
            self.n_fingers,
            self.seq_len,
            self.conv_encoder_2d_embed_dim,
        )
        return x

    def __run_sanity_check(
        self,
        x: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> None:
        print("RUNNING SANITY CHECK, SETTING TO EVAL MODE")
        was_training = self.training
        self.eval()

        sanity_check_x = self.__run_conv_encoder_2d_nobatch(
            x, conditioning=conditioning
        )
        sanity_check_y = self.__run_conv_encoder_2d_batch_fingers(
            x, conditioning=conditioning
        )
        sanity_check_z = self.__run_conv_encoder_2d_batch_fingers_seq(
            x, conditioning=conditioning
        )
        assert torch.allclose(sanity_check_x, sanity_check_y, rtol=1e-3, atol=1e-3)
        assert torch.allclose(sanity_check_x, sanity_check_y, rtol=1e-3, atol=1e-3)

        print(f"Sanity check passed, setting training={was_training}")
        if was_training:
            self.train()

    def __run_conv_encoder_2d_batch_fingers(
        self,
        x: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]

        # Process each finger in each time-step separately
        x = x.reshape(
            batch_size * self.n_fingers, self.seq_len, 1, self.height, self.width
        )
        conditioning = (
            conditioning.reshape(batch_size * self.n_fingers, self.conditioning_dim)
            if conditioning is not None
            and self.use_conditioning_2d
            and self.conditioning_dim is not None
            else None
        )

        output_per_seq_list = []
        for seq_i in range(self.seq_len):
            temp = x[:, seq_i]
            assert temp.shape == (
                batch_size * self.n_fingers,
                1,
                self.height,
                self.width,
            )
            temp = self.conv_encoder_2d(temp, conditioning=conditioning)
            assert temp.shape == (
                batch_size * self.n_fingers,
                self.conv_encoder_2d.output_dim,
            )
            temp = temp.reshape(
                batch_size,
                self.n_fingers,
                self.conv_encoder_2d.output_dim,
            )
            output_per_seq_list.append(temp)
        x = torch.stack(output_per_seq_list, dim=2)
        return x

    def __run_conv_encoder_2d_batch_fingers_seq(
        self,
        x: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]

        # Do this by putting n_fingers and seq_len in the batch dimension
        x = x.reshape(
            batch_size * self.n_fingers * self.seq_len, 1, self.height, self.width
        )
        conditioning = (
            conditioning.reshape(batch_size, self.n_fingers, 1, self.conditioning_dim)
            .repeat(1, 1, self.seq_len, 1)
            .reshape(batch_size * self.n_fingers * self.seq_len, self.conditioning_dim)
            if conditioning is not None
            and self.use_conditioning_2d
            and self.conditioning_dim is not None
            else None
        )
        x = self.conv_encoder_2d(x, conditioning=conditioning)
        assert x.shape == (
            batch_size * self.n_fingers * self.seq_len,
            self.conv_encoder_2d.output_dim,
        )
        x = x.reshape(
            batch_size,
            self.n_fingers,
            self.seq_len,
            self.conv_encoder_2d.output_dim,
        )
        return x

    def __run_conv_encoder_2d_nobatch(
        self,
        x: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        outer_list = []
        # Iterate through each finger
        for finger_i in range(self.n_fingers):
            finger_x = x[:, finger_i]
            assert finger_x.shape == (batch_size, self.seq_len, self.height, self.width)
            finger_conditioning = (
                conditioning[:, finger_i] if conditioning is not None else None
            )
            assert finger_conditioning is None or finger_conditioning.shape == (
                batch_size,
                self.conditioning_dim,
            )

            # Iterate through each time-step
            inner_list = []
            for seq_i in range(self.seq_len):
                temp = finger_x[:, seq_i].reshape(
                    batch_size, 1, self.height, self.width
                )
                temp = self.conv_encoder_2d(temp, conditioning=finger_conditioning)
                inner_list.append(temp)
            inner_list = torch.stack(inner_list, dim=1)

            assert inner_list.shape == (
                batch_size,
                self.seq_len,
                self.conv_encoder_2d.output_dim,
            )
            outer_list.append(inner_list)
        outer_list = torch.stack(outer_list, dim=1)
        assert outer_list.shape == (
            batch_size,
            self.n_fingers,
            self.seq_len,
            self.conv_encoder_2d.output_dim,
        )

        return outer_list

    def get_success_logits(
        self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.forward(x, conditioning=conditioning)

    def get_success_probability(
        self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return nn.functional.softmax(
            self.get_success_logits(x, conditioning=conditioning), dim=-1
        )

    def _prepare_1d_encoder(self) -> None:
        raise NotImplementedError()

    def _run_1d_encoder(
        self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError()


class Condition2D1D_ConcatFingersAfter1D(Abstract2DTo1DClassifier):
    def __init__(
        self,
        conv_encoder_1d_config: Optional[ConvEncoder1DConfig] = None,
        transformer_encoder_1d_config: Optional[TransformerEncoder1DConfig] = None,
        encoder_1d_type: Encoder1DType = Encoder1DType.CONV,
        use_conditioning_1d: bool = False,
        head_mlp_hidden_layers: List[int] = [64, 64],
        **kwargs,
    ) -> None:
        if encoder_1d_type == Encoder1DType.CONV:
            self.encoder_1d_config = conv_encoder_1d_config
        elif encoder_1d_type == Encoder1DType.TRANSFORMER:
            self.encoder_1d_config = transformer_encoder_1d_config
        else:
            raise ValueError(f"Invalid encoder_1d_type = {encoder_1d_type}")
        self.encoder_1d_type = encoder_1d_type
        self.use_conditioning_1d = use_conditioning_1d
        self.head_mlp_hidden_layers = head_mlp_hidden_layers
        super().__init__(**kwargs)

        # Validate usage of conditioning
        if self.conditioning_dim is None:
            assert not self.use_conditioning_2d and not use_conditioning_1d

    def _prepare_1d_encoder(self) -> None:
        # 1D
        self.encoder_1d_input_dim = self.conv_encoder_2d_embed_dim
        self.encoder_1d_seq_len = self.seq_len
        self.encoder_1d_conditioning_dim = (
            self.conditioning_dim if self.use_conditioning_1d else None
        )

        input_shape = (self.encoder_1d_input_dim, self.encoder_1d_seq_len)
        if self.encoder_1d_type == Encoder1DType.CONV:
            self.encoder_1d = ConvEncoder1D(
                input_shape=input_shape,
                conditioning_dim=self.encoder_1d_conditioning_dim,
                # **dataclass_to_kwargs(self.encoder_1d_config)
                use_resnet=self.encoder_1d_config.use_resnet,
                pooling_method=self.encoder_1d_config.pooling_method,
                film_hidden_layers=self.encoder_1d_config.film_hidden_layers,
                base_filters=self.encoder_1d_config.base_filters,
                kernel_size=self.encoder_1d_config.kernel_size,
                stride=self.encoder_1d_config.stride,
                groups=self.encoder_1d_config.groups,
                n_block=self.encoder_1d_config.n_block,
                downsample_gap=self.encoder_1d_config.downsample_gap,
                increasefilter_gap=self.encoder_1d_config.increasefilter_gap,
                use_do=self.encoder_1d_config.use_do,
            )
        elif self.encoder_1d_type == Encoder1DType.TRANSFORMER:
            self.encoder_1d = TransformerEncoder1D(
                input_shape=input_shape,
                conditioning_dim=self.encoder_1d_conditioning_dim,
                # **dataclass_to_kwargs(self.encoder_1d_config)
                pooling_method=self.encoder_1d_config.pooling_method,
                n_heads=self.encoder_1d_config.n_heads,
                n_emb=self.encoder_1d_config.n_emb,
                p_drop_emb=self.encoder_1d_config.p_drop_emb,
                p_drop_attn=self.encoder_1d_config.p_drop_attn,
                n_layers=self.encoder_1d_config.n_layers,
            )
        else:
            raise ValueError(f"Invalid encoder_1d_type = {self.encoder_1d_type}")

        self.head_num_inputs = self.encoder_1d.output_dim * self.n_fingers

        # Head
        self.head = mlp(
            num_inputs=self.head_num_inputs,
            num_outputs=2,
            hidden_layers=self.head_mlp_hidden_layers,
        )

    def _run_1d_encoder(
        self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        assert x.shape == (
            batch_size,
            self.n_fingers,
            self.seq_len,
            self.conv_encoder_2d_embed_dim,
        )
        assert conditioning is None or conditioning.shape == (
            batch_size,
            self.n_fingers,
            self.conditioning_dim,
        )

        # Process all fingers at once
        x = x.reshape(
            batch_size * self.n_fingers, self.seq_len, self.conv_encoder_2d_embed_dim
        )
        conditioning = (
            conditioning.reshape(batch_size * self.n_fingers, self.conditioning_dim)
            if conditioning is not None
            and self.use_conditioning_1d
            and self.conditioning_dim is not None
            else None
        )

        # 1D encoder
        # Need to have (batch_size, n_channels, seq_len) for 1d encoder
        x = x.permute(0, 2, 1)
        assert x.shape == (
            batch_size * self.n_fingers,
            self.encoder_1d_input_dim,
            self.encoder_1d_seq_len,
        )
        assert conditioning is None or conditioning.shape == (
            batch_size * self.n_fingers,
            self.encoder_1d_conditioning_dim,
        )

        x = self.encoder_1d(x, conditioning=conditioning)
        assert x.shape == (
            batch_size * self.n_fingers,
            self.encoder_1d.output_dim,
        )

        # Concatenate fingers
        x = x.reshape(
            batch_size,
            self.n_fingers * self.encoder_1d.output_dim,
        )

        assert x.shape == (batch_size, self.head_num_inputs)

        x = self.head(x)
        assert x.shape == (batch_size, 2)

        return x


class CNN_3D_Classifier(nn.Module):
    def __init__(self, input_example_shape: Tuple[int, int, int]) -> None:
        # TODO: Make this not hardcoded
        super().__init__()
        self.input_example_shape = input_example_shape
        assert len(input_example_shape) == 3
        self.n_fingers = 2

        self.merged_input_shape = (
            self.n_fingers * input_example_shape[0],
            *input_example_shape[1:],
        )

        self.conv = conv_encoder(
            input_shape=self.merged_input_shape,
            conv_channels=[32, 64, 128, 256],
            pool_type=PoolType.MAX,
            dropout_prob=0.1,
            conv_output_to_1d=ConvOutputTo1D.AVG_POOL_SPATIAL,
        )

        # Get conv output shape
        example_batch_size = 2
        example_input = torch.zeros(example_batch_size, *self.merged_input_shape)
        conv_output = self.conv(example_input)
        assert (
            len(conv_output.shape) == 2 and conv_output.shape[0] == example_batch_size
        )
        _, conv_output_dim = conv_output.shape

        N_CLASSES = 2
        self.mlp = mlp(
            num_inputs=conv_output_dim,
            num_outputs=N_CLASSES,
            hidden_layers=[256, 256, 256],
        )

    def forward(
        self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        n_fingers = 2
        assert x.shape == (batch_size, n_fingers, *self.input_example_shape)

        # Merge fingers
        left_finger, right_finger = x[:, 0], x[:, 1]
        x_dim, y_dim = -3, -2
        right_finger = torch.flip(right_finger, dims=[x_dim, y_dim])
        x = torch.cat([left_finger, right_finger], dim=x_dim)
        assert x.shape == (batch_size, *self.merged_input_shape)

        x = self.conv(x)
        x = self.mlp(x)
        return x

    def get_success_logits(
        self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.forward(x)

    def get_success_probability(
        self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return nn.functional.softmax(self.get_success_logits(x), dim=-1)


def test_all_setups(setup_dict: Dict[str, Any]) -> None:
    import itertools
    from tqdm import tqdm

    keys = list(setup_dict.keys())
    options_list = [setup_dict[key] for key in keys]
    all_options_combinations = list(itertools.product(*options_list))
    for i, options in tqdm(
        enumerate(all_options_combinations), total=len(all_options_combinations)
    ):
        kwargs = dict(zip(keys, options))
        print("~" * 80)
        print(f"Testing setup {i + 1} / {len(all_options_combinations)}")
        print("\n".join([f"{key} = {value}" for key, value in kwargs.items()]))
        print("~" * 80)
        test_setup(**kwargs)
        print()


def test_setup(
    batch_size,
    n_fingers,
    seq_len,
    height,
    width,
    conditioning_dim,
    use_conditioning_2d,
    use_conditioning_1d,
    encoder_1d_type,
    conv_encoder_2d_embed_dim,
    conv_encoder_2d_mlp_hidden_layers,
    head_mlp_hidden_layers,
    device,
):
    print("Testing setup")
    print("---------------------")
    print(f"batch_size = {batch_size}")
    print(f"n_fingers = {n_fingers}")
    print(f"seq_len = {seq_len}")
    print(f"height = {height}")
    print(f"width = {width}")
    print(f"conditioning_dim = {conditioning_dim}")
    print()
    example_input = torch.randn(
        batch_size, n_fingers, seq_len, height, width, device=device
    )
    example_conditioning = torch.randn(
        batch_size, n_fingers, conditioning_dim, device=device
    )

    general_model = (
        Condition2D1D_ConcatFingersAfter1D(
            input_shape=(seq_len, height, width),
            n_fingers=n_fingers,
            conditioning_dim=conditioning_dim,
            use_conditioning_2d=use_conditioning_2d,
            conv_encoder_2d_embed_dim=conv_encoder_2d_embed_dim,
            conv_encoder_2d_mlp_hidden_layers=conv_encoder_2d_mlp_hidden_layers,
            use_conditioning_1d=use_conditioning_1d,
            encoder_1d_type=encoder_1d_type,
            head_mlp_hidden_layers=head_mlp_hidden_layers,
        )
        .to(device)
        .eval()
    )
    print(general_model)
    summary(
        general_model,
        input_size=[
            example_input.shape,
            example_conditioning.shape,
        ],
        device=device,
        depth=10,
    )

    example_output = general_model(example_input, example_conditioning)
    print("General2DTo1DClassifier")
    print("=" * 80)
    print(f"example_input.shape = {example_input.shape}")
    print(f"example_conditioning.shape = {example_conditioning.shape}")
    print(f"example_output.shape = {example_output.shape}")
    print()


def main() -> None:
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, n_fingers, seq_len, height, width = (
        2,
        2,
        10,
        30,
        40,
    )  # WARNING: Easy to OOM
    conditioning_dim = 11
    print(f"batch_size = {batch_size}")
    print(f"n_fingers = {n_fingers}")
    print(f"seq_len = {seq_len}")
    print(f"height = {height}")
    print(f"width = {width}")
    print(f"conditioning_dim = {conditioning_dim}")
    print()

    example_input = torch.randn(
        batch_size, n_fingers, seq_len, height, width, device=device
    )
    example_conditioning = torch.randn(
        batch_size, n_fingers, conditioning_dim, device=device
    )

    # Create model
    general_model = (
        Condition2D1D_ConcatFingersAfter1D(
            input_shape=(seq_len, height, width),
            n_fingers=n_fingers,
            conditioning_dim=conditioning_dim,
            use_conditioning_2d=True,
            conv_encoder_2d_embed_dim=32,
            conv_encoder_2d_mlp_hidden_layers=[64, 64],
            encoder_1d_type=Encoder1DType.CONV,
            use_conditioning_1d=True,
            head_mlp_hidden_layers=[64, 64],
        )
        .to(device)
        .eval()
    )

    example_output = general_model(example_input, example_conditioning)
    print("Condition2D1D_ConcatFingersAfter1D CNN")
    print("=" * 80)
    print(f"example_input.shape = {example_input.shape}")
    print(f"example_conditioning.shape = {example_conditioning.shape}")
    print(f"example_output.shape = {example_output.shape}")
    print()

    # Create model 2
    general_model_2 = Condition2D1D_ConcatFingersAfter1D(
        input_shape=(seq_len, height, width),
        n_fingers=n_fingers,
        conditioning_dim=conditioning_dim,
        use_conditioning_2d=True,
        conv_encoder_2d_embed_dim=32,
        conv_encoder_2d_mlp_hidden_layers=[64, 64],
        encoder_1d_type=Encoder1DType.TRANSFORMER,
        use_conditioning_1d=True,
        head_mlp_hidden_layers=[64, 64],
    ).to(device)

    example_output = general_model_2(example_input, example_conditioning)
    print("Condition2D1D_ConcatFingersAfter1D Transformer")
    print("=" * 80)
    print(f"example_input.shape = {example_input.shape}")
    print(f"example_conditioning.shape = {example_conditioning.shape}")
    print(f"example_output.shape = {example_output.shape}")
    print()

    # Spatial softmax
    x = torch.randn(batch_size, seq_len, width, device=device)
    xx = torch.randn(batch_size, seq_len, height, width, device=device)
    spatial_softmax = SpatialSoftmax(temperature=1.0, output_variance=False)
    spatial_softmax_with_variance = SpatialSoftmax(
        temperature=1.0, output_variance=True
    )
    print("SpatialSoftmax")
    print("=" * 80)
    print(f"x.shape = {x.shape}")
    print(f"xx.shape = {xx.shape}")
    print(f"spatial_softmax(x).shape = {spatial_softmax(x).shape}")
    print(f"spatial_softmax(xx).shape = {spatial_softmax(xx).shape}")
    print(
        f"spatial_softmax_with_variance(x).shape = {spatial_softmax_with_variance(x).shape}"
    )
    print(
        f"spatial_softmax_with_variance(xx).shape = {spatial_softmax_with_variance(xx).shape}"
    )
    print()

    # Test many model configs
    print("About to test many model configs")
    ARGUMENT_NAMES_TO_OPTIONS_DICT = {
        "batch_size": [3, 5],
        "n_fingers": [2],
        "seq_len": [10, 7],
        "height": [17],
        "width": [19],
        "conditioning_dim": [15],
        "use_conditioning_2d": [True, False],
        "use_conditioning_1d": [True, False],
        "encoder_1d_type": [e for e in Encoder1DType],
        "conv_encoder_2d_embed_dim": [32, 64],
        "conv_encoder_2d_mlp_hidden_layers": [[64, 64]],
        "head_mlp_hidden_layers": [[64, 64]],
        "device": [device],
    }
    test_all_setups(ARGUMENT_NAMES_TO_OPTIONS_DICT)


def set_seed(seed) -> None:
    import random
    import numpy as np
    import os

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)  # TODO: Is this slowing things down?

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Set random seed to {seed}")


if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception

    set_seed(42)

    LAUNCH_WITH_IPDB = False
    if LAUNCH_WITH_IPDB:
        with launch_ipdb_on_exception():
            main()
    else:
        main()
