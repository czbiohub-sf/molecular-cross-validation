#!/usr/bin/env python

import collections

from typing import Sequence

import torch
import torch.nn as nn


class FCLayers(nn.Module):
    def __init__(
        self, *, n_input: int, layers: Sequence[int], dropout_rate: float = 0.1
    ):
        super(FCLayers, self).__init__()
        layers_dim = [n_input] + layers

        self.fc_layers = nn.Sequential(
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

    def forward(self, x: torch.Tensor):
        for layers in self.fc_layers:
            for layer in layers:
                if isinstance(layer, nn.BatchNorm1d) and x.dim() == 3:
                    x = torch.cat(
                        [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                    )
                else:
                    x = layer(x)

        return x


class ResNetBlock(nn.Module):
    def __init__(
        self, *, n_input: int, layers: Sequence[int], dropout_rate: float = 0.1
    ):
        super(ResNetBlock, self).__init__()

        layers_dim = [n_input] + layers + [n_input]

        self.fc_layers = nn.Sequential(
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

        for layers in self.fc_layers:
            for layer in layers:
                if isinstance(layer, nn.Linear):
                    layer.weight.data.div_(10.0)

    def forward(self, x: torch.Tensor):
        identity = x

        for layers in self.fc_layers:
            for layer in layers:
                if isinstance(layer, nn.BatchNorm1d) and x.dim() == 3:
                    x = torch.cat(
                        [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                    )
                else:
                    x = layer(x)

        return nn.functional.relu(x + identity)
