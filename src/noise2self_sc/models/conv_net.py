#!/usr/bin/env python

import collections

from typing import Sequence

import torch
import torch.nn as nn

from noise2self_sc.models.modules import ConvNetBlock


class ConvNetAutoencoder(nn.Module):
    """A convnet autoencoder. This model passes the data through a bottlenecked
    autoencoder, but also passes the raw data through to the last layer and combines
    the two using a convolution. Not expected to work well unless using noise-to-self
    training.

    :param n_input: dimensionality of the input data (number of features).
    :param n_block: number of ConvNet blocks to use.
    :param layers: sequence of widths for each ConvNet block.
    :param conv_layers: sequence of widths for intermediate layers of convnets
    :param dropout_rate: used between fully-connected layers in ConvNet blocks.
    :param use_cuda: whether to put parameters into GPU memory.
    """

    def __init__(
        self,
        *,
        n_input: int,
        n_blocks: int,
        layers: Sequence[int],
        conv_layers: Sequence[int],
        dropout_rate: float = 0.1,
        use_cuda: bool = False,
    ):
        super(ConvNetAutoencoder, self).__init__()

        self.conv_net_blocks = nn.Sequential(
            *(
                ConvNetBlock(
                    n_input=n_input,
                    layers=layers[:],
                    conv_layers=conv_layers[:],
                    dropout_rate=dropout_rate,
                )
                for i in range(n_blocks)
            )
        )

        if use_cuda:
            self.cuda()

        self.use_cuda = use_cuda

    def forward(self, x: torch.Tensor):
        return self.conv_net_blocks(x)
