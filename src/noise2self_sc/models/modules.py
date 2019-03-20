#!/usr/bin/env python

import collections

from typing import Sequence

import torch
import torch.nn as nn


def make_fc_layers(
    layers: Sequence[int], dropout_rate: float = 0.1, use_bias: bool = True
):
    """A helper function to build fully-connected layers for a neural network.

    :param layers: Size of the intermediate layers
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    :param use_bias: Include a bias parameter in linear layers
    """

    return nn.Sequential(
        *(
            nn.Sequential(
                nn.BatchNorm1d(n_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
                nn.Linear(n_in, n_out, bias=use_bias),
                nn.Dropout(p=dropout_rate),
            )
            for i, (n_in, n_out) in enumerate(zip(layers[:-1], layers[1:]))
        )
    )


def make_conv_layers(
    layers: Sequence[int], dropout_rate: float = 0.1, use_bias: bool = True
):
    """A helper function for creating a small convolutional network to merge the
    parallel inputs of a UNet architecture.

    :param layers: Size of the layers. The first value should be the number of channels
                   being merged, which the last should be the desired result
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    :param use_bias: Include a bias parameter in convolutional layers
    """
    return nn.Sequential(
        *(
            nn.Sequential(
                nn.BatchNorm1d(n_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
                nn.Conv1d(n_in, n_out, 1, bias=use_bias),
                nn.Dropout(p=dropout_rate),
            )
            for i, (n_in, n_out) in enumerate(zip(layers[:-1], layers[1:]))
        )
    )


class ResNetBlock(nn.Module):
    """A residual network block. Passes the input through a series of fully-connected
    ReLU layers before adding it to the original input and applying a final ReLU

    :param n_input: The dimensionality of the input
    :param layers: Size of the intermediate layers (not including final n_input)
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    :param use_bias: Include a bias parameter in linear layers
    """

    def __init__(
        self,
        *,
        n_input: int,
        layers: Sequence[int],
        dropout_rate: float = 0.1,
        use_bias: bool = True,
    ):
        super(ResNetBlock, self).__init__()

        self.layers = make_fc_layers(
            layers=[n_input] + layers + [n_input],
            dropout_rate=dropout_rate,
            use_bias=use_bias,
        )

        # shrink the initial weights to keep the starting point close to the identity
        for layer in self.layers:
            for module in layer:
                if isinstance(module, nn.Linear):
                    module.weight.data.div_(10.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)
