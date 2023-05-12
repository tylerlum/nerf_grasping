import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from omegaconf import MISSING
from FiLM_resnet import resnet18, ResNet18_Weights
from FiLM_resnet_1d import ResNet1D
from torchvision.transforms import Lambda, Compose
from enum import Enum, auto
from functools import partial, cached_property
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from torchinfo import summary


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
            weights_transforms = weights.transforms() if self.use_pretrained else []
            self.img_preprocess = Compose(
                [
                    Lambda(lambda x: x.repeat(1, 3, 1, 1)),
                    weights_transforms,
                ]
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
        example_input = torch.randn(1, *self.input_shape)
        example_conditioning = (
            torch.randn(1, self.conditioning_dim)
            if self.conditioning_dim is not None
            else None
        )
        example_output = self(example_input, conditioning=example_conditioning)
        assert len(example_output.shape) == 2
        return example_output.shape[1]


### 1D ENCODERS ###
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
        example_input = torch.randn(1, *self.input_shape)
        example_conditioning = (
            torch.randn(1, self.conditioning_dim)
            if self.conditioning_dim is not None
            else None
        )
        example_output = self(example_input, conditioning=example_conditioning)
        assert len(example_output.shape) == 2
        return example_output.shape[1]


class TransformerEncoder1D(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        conditioning_dim: Optional[int] = None,
        pooling_method: ConvOutputTo1D = ConvOutputTo1D.FLATTEN,
        n_heads: int = 12,
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
        example_input = torch.randn(1, *self.input_shape)
        example_conditioning = (
            torch.randn(1, self.conditioning_dim)
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
        example_input = torch.randn(1, *self.input_shape)
        example_conditioning = torch.randn(1, *self.input_shape)
        example_output = self(example_input, conditioning=example_conditioning)
        assert len(example_output.shape) == 2
        return example_output.shape[1]


### Classifiers ###


class MergeFingersMethod(Enum):
    MULTIPLY_AFTER_1D = auto()
    ADD_AFTER_1D = auto()
    CONCAT_AFTER_1D = auto()

    MULTIPLY_BEFORE_1D = auto()
    ADD_BEFORE_1D = auto()
    CONCAT_BEFORE_1D_CHANNEL_WISE = auto()
    CONCAT_BEFORE_1D_TIME_WISE = auto()
    ATTENTION_BEFORE_1D = auto()


class Encoder1DType(Enum):
    CONV = auto()
    TRANSFORMER = auto()

class Abstract2DTo1DClassifier(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        n_fingers: int,
        conditioning_dim: Optional[int] = None,
        use_conditioning_2d: bool = False,
        use_conditioning_1d: bool = False,
        encoder_1d_type: Encoder1DType = Encoder1DType.CONV,
        conv_encoder_2d_embed_dim: int = 32,
        concat_conditioning_before_1d: bool = False,
        concat_conditioning_after_1d: bool = False,
        conv_encoder_2d_mlp_hidden_layers: List[int] = [64, 64],
        head_mlp_hidden_layers: List[int] = [64, 64],
    ) -> None:
        # input_shape: (seq_len, height, width)
        super().__init__()
        assert len(input_shape) == 3

        # Store args
        self.input_shape = input_shape
        self.n_fingers = n_fingers
        self.conditioning_dim = conditioning_dim
        self.use_conditioning_2d = use_conditioning_2d
        self.use_conditioning_1d = use_conditioning_1d
        self.encoder_1d_type = encoder_1d_type
        self.conv_encoder_2d_embed_dim = conv_encoder_2d_embed_dim
        self.concat_conditioning_before_1d = concat_conditioning_before_1d
        self.concat_conditioning_after_1d = concat_conditioning_after_1d

        # Handle shapes
        self.n_channels_for_conv_2d = 1
        self.seq_len, self.height, self.width = input_shape

        # Validate usage of conditioning
        if conditioning_dim is None:
            assert (
                not use_conditioning_2d
                and not use_conditioning_1d
                and not concat_conditioning_before_1d
                and not concat_conditioning_after_1d
            )

        # 2D
        conv_encoder_2d_input_shape = (
            self.n_channels_for_conv_2d,
            self.height,
            self.width,
        )
        self.conv_encoder_2d = ConvEncoder2D(
            input_shape=conv_encoder_2d_input_shape,
            conditioning_dim=conditioning_dim if use_conditioning_2d else None,
        )
        self.fc = mlp(
            num_inputs=self.conv_encoder_2d.output_dim,
            num_outputs=conv_encoder_2d_embed_dim,
            hidden_layers=conv_encoder_2d_mlp_hidden_layers,
        )

        # 1D
        if encoder_1d_type == Encoder1DType.CONV:
            self.encoder_1d = ConvEncoder1D(
                input_shape=(self.encoder_1d_input_dim, self.encoder_1d_seq_len),
                conditioning_dim=self.encoder_1d_conditioning_dim,
            )
        elif encoder_1d_type == Encoder1DType.TRANSFORMER:
            self.encoder_1d = TransformerEncoder1D(
                input_shape=(self.encoder_1d_input_dim, self.encoder_1d_seq_len),
                conditioning_dim=self.encoder_1d_conditioning_dim,
            )
        else:
            raise ValueError(f"Invalid encoder_1d_type = {encoder_1d_type}")

        # Head
        self.head = mlp(
            num_inputs=self.head_num_inputs,
            num_outputs=2,
            hidden_layers=head_mlp_hidden_layers,
        )

    def forward(
        self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Validate input
        assert len(x.shape) == 5
        batch_size = x.shape[0]
        assert x.shape[1:] == (self.n_fingers, self.seq_len, self.height, self.width)
        assert conditioning is None or conditioning.shape == (
            batch_size,
            self.n_fingers,
            self.conditioning_dim,
        )

        # Conv 2d (batch_size, n_fingers, seq_len, height, width) => (batch_size, n_fingers, seq_len, conv_encoder_2d_embed_dim)
        x = self._run_conv_encoder_2d(x, conditioning=conditioning)

        # 1d (batch_size, self.encoder_1d_seq_len, self.encoder_1d_input_dim) => (batch_size, self.encoder_1d.output_dim)
        x = self._run_conv_encoder_1d(x, conditioning=conditioning)

        # Head (batch_size, self.head_num_inputs) => (batch_size, 2)
        x = self.head(x)
        return x

    def _run_conv_encoder_2d(
        self,
        x: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        AVOID_OOM=True,
    ) -> torch.Tensor:
        # Validate input
        assert len(x.shape) == 5
        batch_size = x.shape[0]
        assert x.shape[1:] == (self.n_fingers, self.seq_len, self.height, self.width)
        assert conditioning is None or conditioning.shape == (
            batch_size,
            self.n_fingers,
            self.conditioning_dim,
        )

        # Conv 2d (batch_size, n_fingers, seq_len, height, width) => (batch_size, n_fingers, seq_len, conv_encoder_2d_embed_dim)
        if AVOID_OOM:
            # Process each finger in each time-step separately
            # OOM if do all at once
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
        else:
            # Do this by putting seq_len in the batch dimension
            x = x.reshape(
                batch_size * self.n_fingers * self.seq_len, 1, self.height, self.width
            )
            conditioning = (
                conditioning.reshape(
                    batch_size * self.n_fingers, self.conditioning_dim
                ).repeat(self.seq_len, 1)
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

    def _run_conv_encoder_1d(
        self, x: torch.Tensor, encoder_1d_conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        assert len(x.shape) == 3 and x.shape[1:] == (
            self.encoder_1d_seq_len,
            self.encoder_1d_input_dim,
        )
        assert encoder_1d_conditioning is None or encoder_1d_conditioning.shape == (
            batch_size,
            self.n_fingers,
            self.encoder_1d_conditioning_dim,
        )

        # 1d (batch_size, self.encoder_1d_seq_len, self.encoder_1d_input_dim) => (batch_size, self.encoder_1d.output_dim)
        # Need to have (batch_size, n_channels, seq_len) for 1d encoder
        x = x.permute(0, 2, 1)
        assert len(x.shape) == 3 and x.shape[1:] == (
            self.encoder_1d_input_dim,
            self.encoder_1d_seq_len,
        )

        if encoder_1d_conditioning is not None and self.use_conditioning_1d and self.conditioning_dim is not None:
            if x.shape[0] == batch_size * self.n_fingers:
                encoder_1d_conditioning = encoder_1d_conditioning.reshape(
                    batch_size * self.n_fingers, self.conditioning_dim
                )
            # TODO: Not figured out how to fix this

        x = self.encoder_1d(x, conditioning=encoder_1d_conditioning)
        assert len(x.shape) == 2 and x.shape[1] == self.encoder_1d.output_dim

        return x

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



class General2DTo1DClassifier(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        n_fingers: int,
        conditioning_dim: Optional[int] = None,
        use_conditioning_2d: bool = False,
        use_conditioning_1d: bool = False,
        merge_fingers_method: MergeFingersMethod = MergeFingersMethod.MULTIPLY_AFTER_1D,
        encoder_1d_type: Encoder1DType = Encoder1DType.CONV,
        conv_encoder_2d_embed_dim: int = 32,
        concat_conditioning_before_1d: bool = False,
        concat_conditioning_after_1d: bool = False,
        conv_encoder_2d_mlp_hidden_layers: List[int] = [64, 64],
        head_mlp_hidden_layers: List[int] = [64, 64],
    ) -> None:
        # input_shape: (seq_len, height, width)
        super().__init__()
        assert len(input_shape) == 3

        # Store args
        self.input_shape = input_shape
        self.n_fingers = n_fingers
        self.conditioning_dim = conditioning_dim
        self.use_conditioning_2d = use_conditioning_2d
        self.use_conditioning_1d = use_conditioning_1d
        self.merge_fingers_method = merge_fingers_method
        self.encoder_1d_type = encoder_1d_type
        self.conv_encoder_2d_embed_dim = conv_encoder_2d_embed_dim
        self.concat_conditioning_before_1d = concat_conditioning_before_1d
        self.concat_conditioning_after_1d = concat_conditioning_after_1d

        # Handle shapes
        self.n_channels_for_conv_2d = 1
        self.seq_len, self.height, self.width = input_shape

        # Validate usage of conditioning
        if conditioning_dim is None:
            assert (
                not use_conditioning_2d
                and not use_conditioning_1d
                and not concat_conditioning_before_1d
                and not concat_conditioning_after_1d
            )

        # 2D
        conv_encoder_2d_input_shape = (
            self.n_channels_for_conv_2d,
            self.height,
            self.width,
        )
        self.conv_encoder_2d = ConvEncoder2D(
            input_shape=conv_encoder_2d_input_shape,
            conditioning_dim=conditioning_dim if use_conditioning_2d else None,
        )
        self.fc = mlp(
            num_inputs=self.conv_encoder_2d.output_dim,
            num_outputs=conv_encoder_2d_embed_dim,
            hidden_layers=conv_encoder_2d_mlp_hidden_layers,
        )

        # Before 1D
        self.encoder_1d_input_dim = conv_encoder_2d_embed_dim
        self.encoder_1d_seq_len = self.seq_len
        if self.concat_conditioning_before_1d and conditioning_dim is not None:
            self.encoder_1d_input_dim += conditioning_dim

        if merge_fingers_method == MergeFingersMethod.ATTENTION_BEFORE_1D:
            self.attention_before_1d = TransformerEncoderDecoder(
                input_shape=(self.encoder_1d_input_dim, self.seq_len),
                n_emb=self.encoder_1d_input_dim,
                pooling_method=ConvOutputTo1D.FLATTEN,
            )

        if merge_fingers_method in [
            MergeFingersMethod.ADD_BEFORE_1D,
            MergeFingersMethod.MULTIPLY_BEFORE_1D,
            MergeFingersMethod.ADD_AFTER_1D,
            MergeFingersMethod.MULTIPLY_AFTER_1D,
            MergeFingersMethod.CONCAT_AFTER_1D,
            MergeFingersMethod.ATTENTION_BEFORE_1D,
        ]:
            pass
        elif merge_fingers_method == MergeFingersMethod.CONCAT_BEFORE_1D_CHANNEL_WISE:
            self.encoder_1d_input_dim *= n_fingers
        elif merge_fingers_method == MergeFingersMethod.CONCAT_BEFORE_1D_TIME_WISE:
            self.encoder_1d_seq_len *= n_fingers
        else:
            raise ValueError(f"Invalid merge_fingers_method = {merge_fingers_method}")

        # 1D
        encoder_1d_conditioning_dim = (
            None if not use_conditioning_1d else conditioning_dim if  TODO
        )
        if encoder_1d_type == Encoder1DType.CONV:
            self.encoder_1d = ConvEncoder1D(
                input_shape=(self.encoder_1d_input_dim, self.encoder_1d_seq_len),
                conditioning_dim=conditioning_dim if use_conditioning_1d else None,
            )
        elif encoder_1d_type == Encoder1DType.TRANSFORMER:
            self.encoder_1d = TransformerEncoder1D(
                input_shape=(self.encoder_1d_input_dim, self.encoder_1d_seq_len),
                conditioning_dim=conditioning_dim if use_conditioning_1d else None,
            )
        else:
            raise ValueError(f"Invalid encoder_1d_type = {encoder_1d_type}")

        # After 1D
        self.head_num_inputs = self.encoder_1d.output_dim
        if self.concat_conditioning_after_1d and conditioning_dim is not None:
            self.head_num_inputs += conditioning_dim

        if merge_fingers_method == MergeFingersMethod.CONCAT_AFTER_1D:
            self.head_num_inputs *= n_fingers
        elif merge_fingers_method in [
            MergeFingersMethod.MULTIPLY_AFTER_1D,
            MergeFingersMethod.ADD_AFTER_1D,
            MergeFingersMethod.MULTIPLY_BEFORE_1D,
            MergeFingersMethod.ADD_BEFORE_1D,
            MergeFingersMethod.CONCAT_BEFORE_1D_CHANNEL_WISE,
            MergeFingersMethod.CONCAT_BEFORE_1D_TIME_WISE,
            MergeFingersMethod.ATTENTION_BEFORE_1D,
        ]:
            pass
        else:
            raise ValueError(f"Invalid merge_fingers_method = {merge_fingers_method}")

        # Head
        self.head = mlp(
            num_inputs=self.head_num_inputs,
            num_outputs=2,
            hidden_layers=head_mlp_hidden_layers,
        )

    def forward(
        self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Validate input
        assert len(x.shape) == 5
        batch_size = x.shape[0]
        assert x.shape[1:] == (self.n_fingers, self.seq_len, self.height, self.width)
        assert conditioning is None or conditioning.shape == (
            batch_size,
            self.n_fingers,
            self.conditioning_dim,
        )

        # Conv 2d (batch_size, n_fingers, seq_len, height, width) => (batch_size, n_fingers, seq_len, conv_encoder_2d_embed_dim)
        x = self._run_conv_encoder_2d(x, conditioning=conditioning)

        # Before 1d operations (batch_size, n_fingers, seq_len, conv_encoder_2d_embed_dim) => (batch_size, self.encoder_1d_seq_len, self.encoder_1d_input_dim)
        x = self._run_merge_before_1d(x, conditioning=conditioning)

        # 1d (batch_size, self.encoder_1d_seq_len, self.encoder_1d_input_dim) => (batch_size, self.encoder_1d.output_dim)
        x = self._run_conv_encoder_1d(x, conditioning=conditioning)

        # After 1d operations (batch_size, self.encoder_1d.output_dim) => (batch_size, self.head_num_inputs)
        x = self._run_merge_after_1d(x, conditioning=conditioning)

        # Head (batch_size, self.head_num_inputs) => (batch_size, 2)
        x = self.head(x)
        return x

    def _run_conv_encoder_2d(
        self,
        x: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        AVOID_OOM=True,
    ) -> torch.Tensor:
        # Validate input
        assert len(x.shape) == 5
        batch_size = x.shape[0]
        assert x.shape[1:] == (self.n_fingers, self.seq_len, self.height, self.width)
        assert conditioning is None or conditioning.shape == (
            batch_size,
            self.n_fingers,
            self.conditioning_dim,
        )

        # Conv 2d (batch_size, n_fingers, seq_len, height, width) => (batch_size, n_fingers, seq_len, conv_encoder_2d_embed_dim)
        if AVOID_OOM:
            # Process each finger in each time-step separately
            # OOM if do all at once
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
        else:
            # Do this by putting seq_len in the batch dimension
            x = x.reshape(
                batch_size * self.n_fingers * self.seq_len, 1, self.height, self.width
            )
            conditioning = (
                conditioning.reshape(
                    batch_size * self.n_fingers, self.conditioning_dim
                ).repeat(self.seq_len, 1)
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

    def _run_merge_before_1d(
        self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Validate
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

        # Concat conditioning before 1d
        if (
            self.concat_conditioning_before_1d
            and conditioning is not None
            and self.conditioning_dim is not None
        ):
            conditioning = conditioning.reshape(
                batch_size,
                self.n_fingers,
                1,
                self.conditioning_dim,
            ).repeat(1, 1, self.seq_len, 1)
            assert conditioning.shape == (
                batch_size,
                self.n_fingers,
                self.seq_len,
                self.conditioning_dim,
            )

            x = torch.cat([x, conditioning], dim=-1)

        embed_dim = (
            self.conv_encoder_2d_embed_dim + self.conditioning_dim
            if self.conditioning_dim is not None
            else self.conv_encoder_2d_embed_dim
        )

        assert x.shape == (
            batch_size,
            self.n_fingers,
            self.seq_len,
            embed_dim,
        )

        # Before 1d operations (batch_size, n_fingers, seq_len, conv_encoder_2d_embed_dim) => (-1, self.encoder_1d_seq_len, self.encoder_1d_input_dim)
        if self.merge_fingers_method == MergeFingersMethod.MULTIPLY_BEFORE_1D:
            x = x.prod(dim=1)
        elif self.merge_fingers_method == MergeFingersMethod.ADD_BEFORE_1D:
            x = x.sum(dim=1)
        elif (
            self.merge_fingers_method
            == MergeFingersMethod.CONCAT_BEFORE_1D_CHANNEL_WISE
        ):
            x = x.permute(0, 2, 1, 3)
            assert x.shape == (
                batch_size,
                self.seq_len,
                self.n_fingers,
                embed_dim,
            )
            x = x.reshape(
                batch_size,
                self.seq_len,
                self.n_fingers * embed_dim,
            )
        elif self.merge_fingers_method == MergeFingersMethod.CONCAT_BEFORE_1D_TIME_WISE:
            x = x.reshape(
                batch_size,
                self.n_fingers * self.seq_len,
                embed_dim,
            )
        elif self.merge_fingers_method == MergeFingersMethod.ATTENTION_BEFORE_1D:
            # HACK: Assumes n_fingers == 2 so can do attention between the two
            # Could also make n_fingers go along seq_len so that can attend across that too?
            assert self.n_fingers == 2
            left_finger, right_finger = x[:, 0], x[:, 1]
            assert (
                left_finger.shape
                == right_finger.shape
                == (batch_size, self.seq_len, embed_dim)
            )
            left_finger, right_finger = left_finger.permute(
                0, 2, 1
            ), right_finger.permute(0, 2, 1)
            assert (
                left_finger.shape
                == right_finger.shape
                == (batch_size, embed_dim, self.seq_len)
            )
            x = self.attention_before_1d(left_finger, conditioning=right_finger)
            assert x.shape == (
                batch_size,
                embed_dim * self.seq_len,
            )
            x = x.reshape(batch_size, embed_dim, self.seq_len).permute(0, 2, 1)
        elif self.merge_fingers_method in [
            MergeFingersMethod.ADD_AFTER_1D,
            MergeFingersMethod.MULTIPLY_AFTER_1D,
            MergeFingersMethod.CONCAT_AFTER_1D,
        ]:
            # Process fingers individually
            x = x.reshape(
                batch_size * self.n_fingers,
                self.seq_len,
                embed_dim,
            )
        else:
            raise ValueError(
                f"Invalid merge_fingers_method = {self.merge_fingers_method}"
            )

        assert len(x.shape) == 3 and x.shape[1:] == (
            self.encoder_1d_seq_len,
            self.encoder_1d_input_dim,
        )

        return x

    def _run_conv_encoder_1d(
        self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        assert len(x.shape) == 3 and x.shape[1:] == (
            self.encoder_1d_seq_len,
            self.encoder_1d_input_dim,
        )
        assert conditioning is None or conditioning.shape == (
            batch_size,
            self.n_fingers,
            self.conditioning_dim,
        )

        # 1d (batch_size, self.encoder_1d_seq_len, self.encoder_1d_input_dim) => (batch_size, self.encoder_1d.output_dim)
        # Need to have (batch_size, n_channels, seq_len) for 1d encoder
        x = x.permute(0, 2, 1)
        assert len(x.shape) == 3 and x.shape[1:] == (
            self.encoder_1d_input_dim,
            self.encoder_1d_seq_len,
        )

        if conditioning is not None and self.use_conditioning_1d and self.conditioning_dim is not None:
            if x.shape[0] == batch_size * self.n_fingers:
                conditioning = conditioning.reshape(
                    batch_size * self.n_fingers, self.conditioning_dim
                )
            # TODO: Not figured out how to fix this

        x = self.encoder_1d(x, conditioning=conditioning)
        assert len(x.shape) == 2 and x.shape[1] == self.encoder_1d.output_dim

        return x

    def _run_merge_after_1d(
        self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert len(x.shape) == 2 and x.shape[1] == self.encoder_1d.output_dim
        # assert x.shape[0] == batch_size or x.shape[0] == batch_size * self.n_fingers

        # After 1d operations (batch_size, self.encoder_1d.output_dim) => (batch_size, self.head_num_inputs)
        if self.merge_fingers_method in [
            MergeFingersMethod.ADD_AFTER_1D,
            MergeFingersMethod.MULTIPLY_AFTER_1D,
            MergeFingersMethod.CONCAT_AFTER_1D,
        ]:
            x = x.reshape(-1, self.n_fingers, self.encoder_1d.output_dim)

            if self.merge_fingers_method == MergeFingersMethod.ADD_AFTER_1D:
                x = x.sum(dim=1)
            elif self.merge_fingers_method == MergeFingersMethod.MULTIPLY_AFTER_1D:
                x = x.prod(dim=1)
            elif self.merge_fingers_method == MergeFingersMethod.CONCAT_AFTER_1D:
                x = x.reshape(-1, self.n_fingers * self.encoder_1d.output_dim)
            else:
                raise ValueError(
                    f"Invalid merge_fingers_method = {self.merge_fingers_method}"
                )

        assert len(x.shape) == 2

        # Concat conditioning after 1d
        if (
            self.concat_conditioning_after_1d
            and conditioning is not None
            and self.conditioning_dim is not None
        ):
            x = torch.cat([x, conditioning], dim=-1)

        assert len(x.shape) == 2 and x.shape[1] == self.head_num_inputs

        return x

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
    merge_fingers_method,
    encoder_1d_type,
    conv_encoder_2d_embed_dim,
    concat_conditioning_before_1d,
    concat_conditioning_after_1d,
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

    general_model = General2DTo1DClassifier(
        input_shape=(seq_len, height, width),
        n_fingers=n_fingers,
        conditioning_dim=conditioning_dim,
        use_conditioning_2d=use_conditioning_2d,
        use_conditioning_1d=use_conditioning_1d,
        merge_fingers_method=merge_fingers_method,
        encoder_1d_type=encoder_1d_type,
        conv_encoder_2d_embed_dim=conv_encoder_2d_embed_dim,
        concat_conditioning_before_1d=concat_conditioning_before_1d,
        concat_conditioning_after_1d=concat_conditioning_after_1d,
        conv_encoder_2d_mlp_hidden_layers=conv_encoder_2d_mlp_hidden_layers,
        head_mlp_hidden_layers=head_mlp_hidden_layers,
    ).to(device)
    print(general_model)
    summary(
        general_model,
        input_size=[
            (batch_size, n_fingers, seq_len, height, width),
            (
                batch_size,
                conditioning_dim,
            ),
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
        4,
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
    general_model = General2DTo1DClassifier(
        input_shape=(seq_len, height, width),
        n_fingers=n_fingers,
        conditioning_dim=conditioning_dim,
        use_conditioning_2d=True,
        use_conditioning_1d=True,
        merge_fingers_method=MergeFingersMethod.CONCAT_AFTER_1D,
        encoder_1d_type=Encoder1DType.CONV,
        conv_encoder_2d_embed_dim=32,
        concat_conditioning_before_1d=True,
        concat_conditioning_after_1d=True,
        conv_encoder_2d_mlp_hidden_layers=[64, 64],
        head_mlp_hidden_layers=[64, 64],
    ).to(device)

    example_output = general_model(example_input, example_conditioning)
    print("General2DTo1DClassifier CNN")
    print("=" * 80)
    print(f"example_input.shape = {example_input.shape}")
    print(f"example_conditioning.shape = {example_conditioning.shape}")
    print(f"example_output.shape = {example_output.shape}")
    print()

    # Create model 2
    general_model_2 = General2DTo1DClassifier(
        input_shape=(seq_len, height, width),
        n_fingers=n_fingers,
        conditioning_dim=conditioning_dim,
        use_conditioning_2d=True,
        use_conditioning_1d=False,
        merge_fingers_method=MergeFingersMethod.ATTENTION_BEFORE_1D,
        encoder_1d_type=Encoder1DType.TRANSFORMER,
        conv_encoder_2d_embed_dim=32,
        concat_conditioning_before_1d=False,
        concat_conditioning_after_1d=True,
        conv_encoder_2d_mlp_hidden_layers=[64, 64],
        head_mlp_hidden_layers=[64, 64],
    ).to(device)

    example_output = general_model_2(example_input, example_conditioning)
    print("General2DTo1DClassifier Transformer")
    print("=" * 80)
    print(f"example_input.shape = {example_input.shape}")
    print(f"example_conditioning.shape = {example_conditioning.shape}")
    print(f"example_output.shape = {example_output.shape}")
    print()

    # Create model 3
    general_model_3 = General2DTo1DClassifier(
        input_shape=(seq_len, height, width),
        n_fingers=n_fingers,
        conditioning_dim=conditioning_dim,
        use_conditioning_2d=False,
        use_conditioning_1d=False,
        merge_fingers_method=MergeFingersMethod.ADD_AFTER_1D,
        encoder_1d_type=Encoder1DType.TRANSFORMER,
        conv_encoder_2d_embed_dim=32,
        concat_conditioning_before_1d=False,
        concat_conditioning_after_1d=False,
        conv_encoder_2d_mlp_hidden_layers=[64, 64],
        head_mlp_hidden_layers=[64, 64],
    ).to(device)

    example_output = general_model_3(example_input, example_conditioning)

    print("General2DTo1DClassifier Transformer 2")
    print("=" * 80)
    print(f"example_input.shape = {example_input.shape}")
    print(f"example_conditioning.shape = {example_conditioning.shape}")
    print(f"example_output.shape = {example_output.shape}")
    print()

    # Test many model configs
    print("About to test many model configs")
    ARGUMENT_NAMES_TO_OPTIONS_DICT = {
        "batch_size": [3, 5],
        "n_fingers": [2],
        "seq_len": [4, 7],
        "height": [17],
        "width": [19],
        "conditioning_dim": [15],
        "use_conditioning_2d": [True, False],
        "use_conditioning_1d": [True, False],
        "merge_fingers_method": [m for m in MergeFingersMethod],
        "encoder_1d_type": [e for e in Encoder1DType],
        "conv_encoder_2d_embed_dim": [32, 64],
        "concat_conditioning_before_1d": [True, False],
        "concat_conditioning_after_1d": [True, False],
        "conv_encoder_2d_mlp_hidden_layers": [[64, 64]],
        "head_mlp_hidden_layers": [[64, 64]],
        "device": [device],
    }
    test_all_setups(ARGUMENT_NAMES_TO_OPTIONS_DICT)

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


if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception

    LAUNCH_WITH_IPDB = False
    if LAUNCH_WITH_IPDB:
        with launch_ipdb_on_exception():
            main()
    else:
        main()
