#!/usr/bin/env python

from typing import Sequence

import torch
import torch.nn as nn

from noise2self_sc.models.modules import make_fc_layers


class CountAutoencoder(nn.Module):
    """A denoising autoencoder for single-cell count data.

    :param n_input: dimensionality of the input data (number of features).
    :param n_latent: width the bottleneck layer.
    :param layers: sequence of widths for intermediate layers. Order provided is used
                   for the encoder, and is reversed for the decoder.
    :param dropout_rate: used between fully-connected layers in encoder/decoder.
    :param use_cuda: whether to put parameters into GPU memory.
    """

    def __init__(
        self,
        *,
        n_input: int,
        n_latent: int,
        layers: Sequence[int],
        dropout_rate: float = 0.1,
        use_cuda: bool = False,
    ):
        super(CountAutoencoder, self).__init__()

        self.encoder = make_fc_layers(
            layers=[n_input] + layers + [n_latent], dropout_rate=dropout_rate
        )

        self.decoder = make_fc_layers(
            layers=[n_latent] + layers[::-1] + [n_input], dropout_rate=dropout_rate
        )

        if use_cuda:
            self.cuda()

        self.use_cuda = use_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x)),



class NBCountAutoencoder(nn.Module):
    """A denoising autoencoder for single-cell count data. Outputs parameters for a
    negative binomial distribution.

    :param n_input: dimensionality of the input data (number of features).
    :param n_latent: width the bottleneck layer.
    :param layers: sequence of widths for intermediate layers. Order provided is used
                   for the encoder, and is reversed for the decoder.
    :param dropout_rate: used between fully-connected layers in encoder/decoder.
    :param use_cuda: whether to put parameters into GPU memory.
    """

    def __init__(
        self,
        *,
        n_input: int,
        n_latent: int,
        layers: Sequence[int],
        dropout_rate: float = 0.1,
        use_cuda: bool = False,
    ):
        super(NBCountAutoencoder, self).__init__()

        self.encoder = make_fc_layers(
            layers=[n_input] + layers + [n_latent], dropout_rate=dropout_rate
        )

        self.decoder = make_fc_layers(
            layers=[n_latent] + layers[::-1], dropout_rate=dropout_rate
        )

        # log-rate
        self.r_decoder = make_fc_layers(
            layers=[layers[0], n_input], dropout_rate=dropout_rate
        )

        # normalized scale
        self.scale_decoder = nn.Sequential(
            make_fc_layers(layers=[layers[0], n_input], dropout_rate=dropout_rate),
            nn.LogSoftmax(dim=-1)
        )

        if use_cuda:
            self.cuda()

        self.use_cuda = use_cuda

    def forward(self, x: torch.Tensor, loglib: torch.Tensor) -> torch.Tensor:
        y = self.decoder(self.encoder(x))

        log_r = self.r_decoder(y)
        scale = self.scale_decoder(y)

        return log_r, loglib + scale - log_r
