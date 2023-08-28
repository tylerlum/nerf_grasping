import torch
import torch.nn as nn
from typing import Tuple, Iterable
from functools import lru_cache
from nerf_grasping.models.tyler_new_models import (
    conv_encoder,
    PoolType,
    ConvOutputTo1D,
    mlp,
)


class CNN_3D_Classifier(nn.Module):
    def __init__(
        self,
        input_shape: Iterable[int],
        conv_channels: Iterable[int],
        mlp_hidden_layers: Iterable[int],
        n_fingers,
    ) -> None:
        # TODO: Make this not hardcoded
        super().__init__()
        self.input_shape = input_shape
        self.n_fingers = n_fingers

        self.conv = conv_encoder(
            input_shape=self.input_shape,
            conv_channels=conv_channels,
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
            hidden_layers=mlp_hidden_layers,
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
