#!/usr/bin/env python

import collections

from typing import Sequence

import torch
import torch.nn as nn


class FCLayers(nn.Module):
    """A helper class to build fully-connected layers for a neural network.

    :param n_input: The dimensionality of the input
    :param layers: Size of the intermediate layers
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(self, n_input: int, layers: Sequence[int], dropout_rate: float = 0.1):
        super(FCLayers, self).__init__()

        layers_dim = [n_input] + layers

        self.layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer_{}".format(i),
                        nn.Sequential(
                            nn.Linear(n_in, n_out),
                            nn.BatchNorm1d(n_out, eps=1e-3, momentum=0.01),
                            nn.ReLU(),
                            nn.Dropout(p=dropout_rate),
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layers in self.layers:
            for layer in layers:
                if isinstance(layer, nn.BatchNorm1d) and x.dim() == 3:
                    x = torch.cat(
                        [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                    )
                else:
                    x = layer(x)
        return x


class ResNetBlock(nn.Module):
    """A residual network block. Passes the input through a series of fully-connected
    ReLU layers before adding it to the original input and applying a final ReLU

    :param n_input: The dimensionality of the input
    :param layers: Size of the intermediate layers (not including final n_input)
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self, *, n_input: int, layers: Sequence[int], dropout_rate: float = 0.1
    ):
        super(ResNetBlock, self).__init__()

        self.fc_layers = FCLayers(n_input, layers + [n_input], dropout_rate)

        # shrink the initial weights to keep the starting point close to the identity
        for layers in self.fc_layers.layers:
            for layer in layers:
                if isinstance(layer, nn.Linear):
                    layer.weight.data.div_(10.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.relu(self.fc_layers(x) + x)
