#!/usr/bin/env python

from typing import Sequence

import torch
import torch.nn as nn

from noise2self_sc.models.modules import make_fc_layers, make_conv_layers


class UnetAutoencoder(nn.Module):
    """A denoising autoencoder for single-cell count data that uses skip connections.

    :param n_input: dimensionality of the input data (number of features).
    :param n_latent: width the bottleneck layer.
    :param layers: sequence of widths for intermediate layers. Order provided is used
                   for the encoder, and is reversed for the decoder.
    :param conv_layers: sequence of widths for intermediate layers of convnets
    :param dropout_rate: used between fully-connected layers in encoder/decoder.
    :param use_cuda: whether to put parameters into GPU memory.
    """

    def __init__(
        self,
        *,
        n_input: int,
        n_latent: int,
        layers: Sequence[int],
        conv_layers: Sequence[int],
        dropout_rate: float = 0.1,
        use_cuda: bool = False,
    ):
        super(UnetAutoencoder, self).__init__()

        layers = [n_input] + layers + [n_latent]

        self.encoder = make_fc_layers(layers=layers, dropout_rate=dropout_rate)

        conv_layers = [2] + conv_layers + [1]

        self.conv_layers = nn.ModuleList(
            [
                make_conv_layers(layers=conv_layers, dropout_rate=dropout_rate)
                for i in range(len(layers) - 1)
            ]
        )

        self.decoder = make_fc_layers(layers=layers[::-1], dropout_rate=dropout_rate)

        if use_cuda:
            self.cuda()

        self.use_cuda = use_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ys = []

        for layers in self.encoder:
            ys.append(x)
            x = layers(x)

        ys = ys[::-1]

        for i, layers in enumerate(self.decoder):
            x = torch.stack([layers(x), ys[i]], dim=1)
            x = self.conv_layers[i](x).squeeze()

        return x
