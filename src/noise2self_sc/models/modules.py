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
