#!/usr/bin/env python

import itertools
import warnings

from typing import Callable, Tuple, Union

import numpy as np

import scipy.spatial.distance as sdist

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Optimizer
from torch.utils.data import DataLoader

from noise2self_sc.train.cosine_scheduler import CosineWithRestarts


Transform = Callable[[torch.Tensor], torch.Tensor]


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


class Noise2SelfDataLoader(DataLoader):
    """A version of the standard DataLoader that generates a new noise2self split every
    time it is iterated on. The dataset passed in should be an instance of TensorDataset
    and contain one tensor of UMI counts. Any transformation(s) of the data must be done
    downstream of this class.
    """

    def __iter__(self):
        x_data = torch.distributions.Binomial(
            self.dataset.tensors[0], probs=0.5
        ).sample()
        y_data = self.dataset.tensors[0] - x_data

        if self.pin_memory:
            x_data = x_data.cuda()
            y_data = y_data.cuda()

        for indices in iter(self.batch_sampler):
            yield x_data[indices], y_data[indices]


def train_loop(
    model: nn.Module,
    criterion: nn.Module,
    optim: Optimizer,
    training_data: DataLoader,
    training_t: Transform,
    criterion_t: Transform,
    use_cuda: bool,
):
    """Iterate through training data, compute losses and take gradient steps

    :param model: a torch Module that can take input data and return the prediction
    :param criterion: a loss function
    :param optim: a torch Optimizer
    :param training_data: training dataset. Should produce a tuple of tensors: the
                            first is used as input and the last is the target. If the
                            tuple has only one element then it's used for both
    :param training_t: Transformation to the data when training the model
    :param criterion_t: Transformation to the data when scoring the output
    :param use_cuda: whether to use the GPU
    :return: total loss for the epoch, averaged over the number of batches
    """
    total_epoch_loss = 0.0

    for data in training_data:
        if use_cuda:
            data = tuple(x.cuda() for x in data)

        y = model(training_t(data[0]))
        loss = criterion(y, criterion_t(data[-1]))

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
    training_t: Transform,
    criterion_t: Transform,
    use_cuda: bool,
):
    """Iterate through test data and compute losses

    :param model: a torch Module that can take input data and return the prediction
    :param criterion: a loss function
    :param optim: a torch Optimizer (will zero the gradient after testing)
    :param validation_data: validation dataset. Should produce a tuple of tensors: the
                            first is used as input and the last is the target. If the
                            tuple has only one element then it's used for both
    :param training_t: Transformation to the data when training the model
    :param criterion_t: Transformation to the data when scoring the output
    :param use_cuda: whether to use the GPU
    :return: total loss for the epoch, averaged over the number of batches
    """
    total_epoch_loss = 0.0

    for data in validation_data:
        if use_cuda:
            data = tuple(x.cuda() for x in data)

        y = model(training_t(data[0]))
        loss = criterion(y, criterion_t(data[-1]))

        total_epoch_loss += loss.data.item()

    optim.zero_grad()

    return total_epoch_loss / len(validation_data)


def mse_loop(
    model: nn.Module,
    ground_truth: torch.Tensor,
    data_loader: DataLoader,
    training_t: Transform,
    eval_t: Transform,
    use_cuda: bool,
):
    """Iterate through a data loader and compute the mean-squared-error to a given
    "ground truth" tensor

    :param model: a torch Module that can take input data and return the prediction
    :param ground_truth: the presumed ground truth for these data
    :param data_loader: dataset to iterate through and produce predictions. The first
                        element will be passed to the model
    :param training_t: Transformation to the data when training the model
    :param eval_t: Transformation when comparing to ground truth
    :param use_cuda: Whether to use the GPU
    :return: the mean squared error averaged over the number of batches
    """
    data_index = data_loader.sampler.indices
    if use_cuda:
        ground_truth = ground_truth[data_index, :].cuda().flatten()
    else:
        ground_truth = ground_truth[data_index, :].flatten()

    total_epoch_mse = 0.0

    for data in data_loader:
        if use_cuda:
            data = tuple(x.cuda() for x in data)

        y = eval_t(model(training_t(data[0])))
        mse = F.mse_loss(ground_truth, y.detach().flatten())

        total_epoch_mse += mse.data.item()

    return total_epoch_mse / len(data_loader)


def cosine_loop(
    model: nn.Module,
    ground_truth: torch.Tensor,
    data_loader: DataLoader,
    training_t: Transform,
    eval_t: Transform,
    use_cuda: bool,
):
    """Iterate through a data loader and compute the cosine similarity to a given
    "ground truth" tensor

    :param model: a torch Module that can take input data and return the prediction
    :param ground_truth: the presumed ground truth for these data
    :param data_loader: dataset to iterate through and produce predictions. The first
                        element will be passed to the model
    :param training_t: Transformation to the data when training the model
    :param eval_t: Transformation when comparing to ground truth
    :param use_cuda: Whether to use the GPU
    :return: the cosine similarity averaged over the number of batches
    """
    data_index = data_loader.sampler.indices
    if use_cuda:
        ground_truth = ground_truth[data_index, :].cuda().flatten()
    else:
        ground_truth = ground_truth[data_index, :].flatten()

    total_epoch_sim = 0.0

    for data in data_loader:
        if use_cuda:
            data = tuple(x.cuda() for x in data)

        y = eval_t(model(training_t(data[0])))
        sim = F.cosine_similarity(ground_truth, y.detach().flatten(), dim=0)

        total_epoch_sim += sim.data.item()

    return total_epoch_sim / len(data_loader)


def correlation_loop(
    model: nn.Module,
    ground_truth: Union[torch.Tensor, np.ndarray],
    data_loader: DataLoader,
    training_t: Transform,
    eval_t: Transform,
    use_cuda: bool,
):
    """Iterate through a data loader and compute the correlation distance to a given
    "ground truth" array

    :param model: a torch Module that can take input data and return the prediction1
    :param ground_truth: the presumed ground truth for these data
    :param data_loader: dataset to iterate through and produce predictions. The first
                        element will be passed to the model
    :param training_t: Transformation to the data when training the model
    :param eval_t: Transformation when comparing to ground truth
    :param use_cuda: Whether to use the GPU
    :return: the correlation distance averaged over the number of batches
    """

    data_index = data_loader.sampler.indices
    ground_truth = np.asarray(ground_truth[data_index, :]).flatten()

    total_epoch_dist = 0.0

    for data in data_loader:
        if use_cuda:
            data = tuple(x.cuda() for x in data)

        y = eval_t(model(training_t(data[0])))
        dist = sdist.correlation(ground_truth, y.cpu().detach().flatten().numpy())

        total_epoch_dist += dist

    return total_epoch_dist / len(data_loader)


def train_until_plateau(
    model: nn.Module,
    criterion: nn.Module,
    optim: Optimizer,
    training_data: DataLoader,
    validation_data: DataLoader,
    training_t: Transform = None,
    crit_t: Transform = None,
    eval_t: Transform = None,
    ground_truth: Union[torch.Tensor, np.ndarray] = None,
    min_cycles: int = 3,
    threshold: float = 0.01,
    scheduler_kw: dict = None,
    use_cuda: bool = False,
    verbose: bool = False,
) -> Tuple[list, list]:
    """Train a model with cosine scheduling until validation loss stabilizes. This
    function uses CosineWithRestarts to train until the learning rate stops improving.

    :param model: torch Module that can take input data and return the prediction
    :param criterion: A loss function
    :param optim: torch Optimizer (will zero the gradient after testing)
    :param training_data: Training dataset. Should produce tuples of Tensors, all but
                          the last are considered to be input and the last is the target
    :param validation_data: Validation dataset in the same format
    :param training_t: Transformation to the data when training the model
    :param crit_t: Transformation to the data when scoring the output
    :param eval_t: Transformation when comparing to ground truth
    :param ground_truth: The presumed ground truth for these data
    :param min_cycles: Minimum number of cycles to run before checking for convergence
    :param threshold: Tolerance threshold for calling convergence
    :param scheduler_kw: dictionary of keyword arguments for CosineWithRestarts
    :param use_cuda: Whether to use the GPU
    :param verbose: Print training progress to stdout
    :return: Lists of training and validation loss and correlation values
    """

    assert 0.0 <= threshold < 1.0

    if training_t is None:
        training_t = lambda x: x
    if crit_t is None:
        crit_t = lambda x: x
    if eval_t is None:
        eval_t = lambda x: x

    if scheduler_kw is None:
        scheduler_kw = dict()

    train_loss = []
    val_loss = []
    train_mse = []
    val_mse = []

    scheduler = CosineWithRestarts(optim, **scheduler_kw)
    best = np.inf
    rel_epsilon = 1.0 - threshold
    cycle = 0

    for epoch in itertools.count():
        train_loss.append(
            train_loop(
                model, criterion, optim, training_data, training_t, crit_t, use_cuda
            )
        )
        val_loss.append(
            validate_loop(
                model, criterion, optim, validation_data, training_t, crit_t, use_cuda
            )
        )

        if ground_truth is not None:
            train_mse.append(
                mse_loop(
                    model, ground_truth, training_data, training_t, eval_t, use_cuda
                )
            )
            val_mse.append(
                mse_loop(
                    model, ground_truth, validation_data, training_t, eval_t, use_cuda
                )
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

    return train_loss, val_loss, train_mse, val_mse
