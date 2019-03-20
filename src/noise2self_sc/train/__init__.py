#!/usr/bin/env python

import itertools
import warnings

from typing import Tuple

import numpy as np

import torch
import torch.nn as nn

from torch.optim import Optimizer
from torch.utils.data import DataLoader

from noise2self_sc.train.cosine_scheduler import CosineWithRestarts


class NegativeBinomialNLLLoss(nn.Module):
    """Negative log likelihood loss with Negative Binomial distribution of target.

    :param eps: value to use for numerical stability of total_count
    """

    def __init__(self, eps: float = 1e-8):
        super(NegativeBinomialNLLLoss, self).__init__()
        self.eps = eps

    def forward(self, params: Tuple[torch.Tensor, torch.Tensor], target: torch.Tensor):
        """Calculate the NB loss

        :param params: log of the rate parameters and logit values for success rate
        :param target: target values for computing log_prob
        :return: mean of negative log likelihood
        """
        log_r, logits = nb_params

        d = torch.distributions.NegativeBinomial(
            torch.exp(log_r) + self.eps, logits=logits, validate_args=True
        )

        return -torch.mean(d.log_prob(target))


def train_loop(
    model: nn.Module,
    criterion: nn.Module,
    optim: Optimizer,
    training_data: DataLoader,
    use_cuda: bool,
):
    """Iterate through training data, compute losses and take gradient steps

    :param model: a torch Module that can take input data and return the prediction
    :param criterion: a loss function
    :param optim: a torch Optimizer
    :param training_data: training dataset. Should produce tuples of Tensors, all but
                          the last are considered to be input and the last is the target
    :param use_cuda: whether to use the GPU
    :return: total loss for the epoch
    """
    total_epoch_loss = 0.0

    for data in training_data:
        if use_cuda:
            data = tuple(x.cuda() for x in data)

        ys = model(*data[:-1])
        loss = criterion(ys, data[-1])

        total_epoch_loss += loss.data.item()

        optim.zero_grad()
        loss.backward()
        optim.step()

    return total_epoch_loss / len(training_data)


def validate_loop(
    model: nn.Module,
    criterion: nn.Module,
    optim: Optimizer,
    validation_data: DataLoader,
    use_cuda: bool,
):
    """Iterate through test data and compute losses

    :param model: a torch Module that can take input data and return the prediction
    :param criterion: a loss function
    :param optim: a torch Optimizer (will zero the gradient after testing)
    :param validation_data: testing dataset. Should produce tuples of Tensors, all but
                            the last are considered to be input; the last is the target
    :param use_cuda: whether to use the GPU
    :return: total loss for the epoch
    """
    total_epoch_loss = 0.0

    for data in validation_data:
        if use_cuda:
            data = tuple(x.cuda() for x in data)

        ys = model(*data[:-1])
        loss = criterion(ys, data[-1])

        total_epoch_loss += loss.data.item()

    optim.zero_grad()

    return total_epoch_loss / len(validation_data)


def train_until_plateau(
    model: nn.Module,
    criterion: nn.Module,
    optim: Optimizer,
    training_data: DataLoader,
    validation_data: DataLoader,
    t_max: int = 128,
    factor: float = 1.0,
    min_cycles: int = 3,
    threshold: float = 0.01,
    eta_min: float = 1e-4,
    use_cuda: bool = False,
    verbose: bool = False,
) -> Tuple[list, list]:
    """Train a model with cosine scheduling until validation loss stabilizes.

    This function implements uses CosineWithRestarts to train until the learning rate
    falls below a given threshold or it reachs ``max_epochs``, whichever comes first.

    :param model: torch Module that can take input data and return the prediction
    :param criterion: A loss function
    :param optim: torch Optimizer (will zero the gradient after testing)
    :param training_data: Training dataset. Should produce tuples of Tensors, all but
                          the last are considered to be input and the last is the target
    :param validation_data: Validation dataset in the same format
    :param t_max: The maximum number of iterations within the first cycle.
    :param factor: The factor by which the cycle length (``T_max``) increases after
                   each restart
    :param min_cycles: Minimum number of cycles to run before checking for convergence
    :param threshold: Tolerance threshold for calling convergence
    :param eta_min: Minimum learning rate
    :param use_cuda: Whether to use the GPU
    :param verbose: Output training progress to stdout
    :return: Lists of training and validation loss values
    """

    assert 0.0 <= threshold < 1.0

    train_loss = []
    val_loss = []

    best = np.inf
    rel_epsilon = 1.0 - threshold
    cycle = 0
    scheduler = CosineWithRestarts(optim, t_max=t_max, eta_min=eta_min, factor=factor)

    for epoch in itertools.count():
        train_loss.append(train_loop(model, criterion, optim, training_data, use_cuda))
        val_loss.append(
            validate_loop(model, criterion, optim, validation_data, use_cuda)
        )

        scheduler.step()
        if scheduler.starting_cycle:
            if verbose:
                print(
                    f"[epoch {epoch:03d}]  average training loss: {train_loss[-1]:.5f}"
                )
            cycle += 1

            if val_loss[-1] < best * rel_epsilon:
                best = val_loss[-1]
            elif cycle >= min_cycles:
                break

    return train_loss, val_loss
