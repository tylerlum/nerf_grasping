import torch
import torch.nn as nn
from typing import Tuple, Optional
from functools import lru_cache
from nerf_grasping.models.tyler_new_models import (
    conv_encoder,
    PoolType,
    ConvOutputTo1D,
    mlp,
    ConvEncoder2D,
    ConvEncoder1D,
)


def assert_equals(a, b):
    assert a == b, f"{a} != {b}"


class CNN_3D_Classifier(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int, int], n_fingers) -> None:
        # TODO: Make this not hardcoded
        super().__init__()
        self.input_shape = input_shape
        self.n_fingers = n_fingers

        self.conv = conv_encoder(
            input_shape=self.input_shape,
            conv_channels=[32, 64, 128],
            pool_type=PoolType.MAX,
            dropout_prob=0.1,
            conv_output_to_1d=ConvOutputTo1D.AVG_POOL_SPATIAL,
        )

        # Get conv output shape
        example_batch_size = 2
        example_input = torch.zeros(
            example_batch_size, self.n_fingers, *self.input_shape
        )
        example_input = example_input.reshape(
            example_batch_size * self.n_fingers, *self.input_shape
        )
        conv_output = self.conv(example_input)
        self.conv_output_dim = conv_output.shape[-1]
        assert conv_output.shape == (
            example_batch_size * self.n_fingers,
            self.conv_output_dim,
        )

        self.mlp = mlp(
            num_inputs=self.conv_output_dim * self.n_fingers,
            num_outputs=self.n_classes,
            hidden_layers=[256, 256],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        assert x.shape == (
            batch_size,
            self.n_fingers,
            *self.input_shape,
        ), f"{x.shape}"

        # Put n_fingers into batch dim
        x = x.reshape(batch_size * self.n_fingers, *self.input_shape)

        x = self.conv(x)
        assert x.shape == (
            batch_size * self.n_fingers,
            self.conv_output_dim,
        ), f"{x.shape}"
        x = x.reshape(batch_size, self.n_fingers, self.conv_output_dim)
        x = x.reshape(batch_size, self.n_fingers * self.conv_output_dim)

        x = self.mlp(x)
        assert x.shape == (batch_size, self.n_classes), f"{x.shape}"
        return x

    def get_success_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def get_success_probability(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.softmax(self.get_success_logits(x), dim=-1)

    @property
    @lru_cache()
    def n_classes(self) -> int:
        return 2


class CNN_2D_1D_Classifier(nn.Module):
    def __init__(self, grid_shape: Tuple[int, int, int]) -> None:
        # TODO: Make this not hardcoded
        super().__init__()
        self.grid_shape = grid_shape

        n_fingers = 4
        n_pts_x, n_pts_y, n_pts_z = self.grid_shape
        seq_len = n_pts_z
        conditioning_dim = 7

        self.conv_2d = ConvEncoder2D(
            input_shape=(1, n_pts_x, n_pts_y),
            conditioning_dim=conditioning_dim,
            use_resnet=True,
            use_pretrained=True,
            pooling_method=ConvOutputTo1D.AVG_POOL_SPATIAL,
            film_hidden_layers=[256, 256],
        )
        self.conv_1d = ConvEncoder1D(
            input_shape=(self.conv_2d.output_dim, seq_len),
            conditioning_dim=conditioning_dim,
            use_resnet=True,
            pooling_method=ConvOutputTo1D.AVG_POOL_SPATIAL,
            base_filters=64,
            kernel_size=16,
            stride=2,
            groups=32,
            n_block=8,
            downsample_gap=2,
            increasefilter_gap=4,
            use_do=False,
        )
        self.mlp = mlp(
            num_inputs=n_fingers * self.conv_1d.output_dim
            + n_fingers * conditioning_dim,
            num_outputs=self.n_classes,
            hidden_layers=[256, 256],
        )

    def forward(
        self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        n_fingers = 4
        n_pts_x, n_pts_y, n_pts_z = self.grid_shape
        seq_len = n_pts_z
        conditioning_dim = 7
        assert_equals(x.shape, (batch_size, n_fingers, n_pts_x, n_pts_y, seq_len))

        if conditioning is not None:
            assert_equals(conditioning.shape, (batch_size, n_fingers, conditioning_dim))

        x = x.reshape(batch_size * n_fingers, n_pts_x, n_pts_y, seq_len)
        if conditioning is not None:
            conditioning = conditioning.reshape(
                batch_size * n_fingers, conditioning_dim
            )

        x = x.permute(0, 3, 1, 2)
        assert_equals(x.shape, (batch_size * n_fingers, seq_len, n_pts_x, n_pts_y))

        output_list = []
        for i in range(seq_len):
            input = x[:, i : i + 1, :, :]
            assert_equals(input.shape, (batch_size * n_fingers, 1, n_pts_x, n_pts_y))
            output = self.conv_2d(input, conditioning=conditioning)
            assert_equals(
                output.shape, (batch_size * n_fingers, self.conv_2d.output_dim)
            )
            output_list.append(output)

        x = torch.stack(output_list, dim=1)
        assert_equals(
            x.shape, (batch_size * n_fingers, seq_len, self.conv_2d.output_dim)
        )

        x = x.permute(0, 2, 1)
        assert_equals(
            x.shape, (batch_size * n_fingers, self.conv_2d.output_dim, seq_len)
        )
        x = self.conv_1d(x, conditioning=conditioning)
        assert_equals(x.shape, (batch_size * n_fingers, self.conv_1d.output_dim))

        x = x.reshape(batch_size, n_fingers, self.conv_1d.output_dim)
        x = x.reshape(batch_size, n_fingers * self.conv_1d.output_dim)

        if conditioning is not None:
            conditioning = conditioning.reshape(batch_size, n_fingers, conditioning_dim)
            conditioning = conditioning.reshape(
                batch_size, n_fingers * conditioning_dim
            )
        x = torch.cat([x, conditioning], dim=1)
        assert_equals(
            x.shape,
            (
                batch_size,
                n_fingers * self.conv_1d.output_dim + n_fingers * conditioning_dim,
            ),
        )

        x = self.mlp(x)
        assert_equals(x.shape, (batch_size, self.n_classes))
        return x

    def get_success_logits(self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(x, conditioning=conditioning)

    @property
    @lru_cache()
    def n_classes(self) -> int:
        return 2