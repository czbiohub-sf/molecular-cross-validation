#!/usr/bin/env python

import collections

from typing import Sequence

import torch
import torch.nn as nn

from src.noise2scelf import ResNetBlock


class ResidualEncoder(nn.Module):
    def __init__(
        self,
        *,
        n_input: int,
        n_blocks: int,
        layers: Sequence[int],
        dropout_rate: float = 0.1,
        use_cuda: bool = False,
    ):
        super(ResidualEncoder, self).__init__()

        self.resnet_blocks = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Block_{}".format(i),
                        ResNetBlock(
                            n_input=n_input, layers=layers[:], dropout_rate=dropout_rate
                        ),
                    )
                    for i in range(n_blocks)
                ]
            )
        )

        # log-rate
        self.r_decoder = nn.Linear(n_input, n_input)

        # logit
        self.logit_decoder = nn.Linear(n_input, n_input)

        if use_cuda:
            self.cuda()

        self.use_cuda = use_cuda

    def forward(self, x: torch.Tensor):
        res = self.resnet_blocks(x)

        log_r = self.r_decoder(res)
        logit = self.logit_decoder(res)

        return log_r, logit
