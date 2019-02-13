#!/usr/bin/env python

from typing import Sequence

import torch
import torch.nn as nn

from noise2self_sc.modules import FCLayers


class NBDecoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        layers: Sequence[int],
        dropout_rate: float = 0.1,
    ):
        super(NBDecoder, self).__init__()
        self.decoder = FCLayers(
            n_input=n_input, layers=layers, dropout_rate=dropout_rate
        )

        # log-rate
        self.r_decoder = nn.Linear(layers[-1], n_output)

        # normalized scale
        self.scale_decoder = nn.Sequential(
            nn.Linear(layers[-1], n_output), nn.LogSoftmax(dim=-1)
        )

    def forward(self, z: torch.Tensor):
        px = self.decoder(z)

        log_r = self.r_decoder(px)
        scale = self.scale_decoder(px)

        return log_r, scale


class CountAutoencoder(nn.Module):
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
        super(CountAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            FCLayers(n_input=n_input, layers=layers, dropout_rate=dropout_rate),
            nn.Linear(layers[-1], n_latent),
        )

        self.decoder = NBDecoder(
            n_input=n_latent,
            n_output=n_input,
            layers=layers[::-1],
            dropout_rate=dropout_rate,
        )

        if use_cuda:
            self.cuda()

        self.use_cuda = use_cuda

    def forward(self, x: torch.Tensor, loglib: torch.Tensor):
        z = self.encoder(x)
        log_r, scale = self.decoder(z)

        return log_r, loglib + scale - log_r
