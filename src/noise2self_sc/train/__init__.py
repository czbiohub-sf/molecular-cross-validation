#!/usr/bin/env python

from typing import Tuple
import warnings

import torch
import torch.nn as nn

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


class NegativeBinomialNLLoss(nn.Module):
    """Negative log likelihood loss with Negative Binomial distribution of target.

    :param eps: value to use for numerical stability of total_count
    """

    def __init__(self, eps: float = 1e-8):
        super(NegativeBinomialNLLoss, self).__init__()
        self.eps = eps

    def forward(self, log_r: torch.Tensor, logits: torch.Tensor, target: torch.Tensor):
        """Calculate the NB loss

        :param log_r: log of the rate parameters
        :param logits: logit values for success rate
        :param target: target values for computing log_prob
        :return: mean of negative log likelihood
        """
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
    total_epoch_loss = 0.

    for data in training_data:
        if use_cuda:
            data = tuple(x.cuda() for x in data)

        ys = model(*data[:-1])
        loss = criterion(*ys, data[-1])

        total_epoch_loss += loss.data.item()

        optim.zero_grad()
        loss.backward()
        optim.step()

    return total_epoch_loss


def test_loop(
    model: nn.Module,
    criterion: nn.Module,
    optim: Optimizer,
    testing_data: DataLoader,
    use_cuda: bool,
):
    """Iterate through test data and compute losses

    :param model: a torch Module that can take input data and return the prediction
    :param criterion: a loss function
    :param optim: a torch Optimizer (will zero the gradient after testing)
    :param testing_data: testing dataset. Should produce tuples of Tensors, all but the
                         last are considered to be input and the last is the target
    :param use_cuda: whether to use the GPU
    :return: total loss for the epoch
    """
    total_epoch_loss = 0.

    for data in testing_data:
        if use_cuda:
            data = tuple(x.cuda() for x in data)

        ys = model(*data[:-1])
        loss = criterion(*ys, data[-1])

        total_epoch_loss += loss.data.item()

    optim.zero_grad()

    return total_epoch_loss


def train_until_plateau(
    model: nn.Module,
    criterion: nn.Module,
    optim: Optimizer,
    training_data: DataLoader,
    testing_data: DataLoader,
    max_epochs: int = 1000,
    min_lr: float = 1e-6,
    use_cuda: bool = False,
) -> Tuple[list, list]:
    """Train a model with LR reductions until validation loss stabilizes.

    This function implements uses ReduceLROnPlateau to train until the learning rate
    falls below a given threshold or it reachs ``max_epochs``, whichever comes first.

    :param model: a torch Module that can take input data and return the prediction
    :param criterion: a loss function
    :param optim: a torch Optimizer (will zero the gradient after testing)
    :param training_data: training dataset. Should produce tuples of Tensors, all but
                          the last are considered to be input and the last is the target
    :param testing_data: testing dataset in the same format
    :param max_epochs: maximum number of epochs to run before returning
    :param min_lr: learning rate threshold to stop training
    :param use_cuda: whether to use the GPU
    :return: lists of training and validation loss values
    """

    train_loss = []
    test_loss = []

    scheduler = ReduceLROnPlateau(optim, "min", factor=0.5)

    for epoch in range(max_epochs):
        train_loss.append(train_loop(model, criterion, optim, training_data, use_cuda))
        test_loss.append(test_loop(model, criterion, optim, testing_data, use_cuda))
        scheduler.step(test_loss[-1])

        if any(pg['lr'] <= min_lr for pg in optim.param_groups):
            break
    else:
        warnings.warn("Reached max_epochs without hitting min_lr")

    return train_loss, test_loss
