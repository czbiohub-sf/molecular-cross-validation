#!/usr/bin/env python

import torch
import torch.nn as nn

from torch.optim import Optimizer
from torch.utils.data import DataLoader


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
    data_loader: DataLoader,
    use_cuda: bool,
):
    """Function to iterate through training data, compute losses and take gradient steps

    :param model: a torch Module that can take input data and return the prediction
    :param criterion: a loss function
    :param optim: a torch Optimizer
    :param data_loader: training dataset. Should produce tuples of Tensors, all but the
                        last are considered to be input and the last is the target
    :param use_cuda: whether to use the GPU
    :return: total loss for the epoch
    """
    total_epoch_loss = 0.

    for data in data_loader:
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
    data_loader: DataLoader,
    use_cuda: bool,
):
    """Function to iterate through test data and compute losses

    :param model: a torch Module that can take input data and return the prediction
    :param criterion: a loss function
    :param optim: a torch Optimizer (will zero the gradient after testing)
    :param data_loader: testing dataset. Should produce tuples of Tensors, all but the
                        last are considered to be input and the last is the target
    :param use_cuda: whether to use the GPU
    :return: total loss for the epoch
    """
    total_epoch_loss = 0.

    for data in data_loader:
        if use_cuda:
            data = tuple(x.cuda() for x in data)

        ys = model(*data[:-1])
        loss = criterion(*ys, data[-1])

        total_epoch_loss += loss.data.item()

    optim.zero_grad()

    return total_epoch_loss


def train_until_plateau(model, training_data, testing_data):
    pass
