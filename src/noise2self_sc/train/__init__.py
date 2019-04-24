#!/usr/bin/env python

import itertools
import warnings

from collections import defaultdict
from typing import Callable, Sequence, Tuple, Union

import numpy as np

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

    def __init__(self, dataset, **kwargs):
        super(Noise2SelfDataLoader, self).__init__(dataset=dataset, **kwargs)

        if kwargs.get("pin_memory", False):
            self.b_dist = torch.distributions.Binomial(
                self.dataset.tensors[0].cuda(), probs=0.5
            )
        else:
            self.b_dist = torch.distributions.Binomial(
                self.dataset.tensors[0], probs=0.5
            )

    def __iter__(self):
        x_data = self.b_dist.sample()
        y_data = self.b_dist.total_count - x_data

        for indices in iter(self.batch_sampler):
            yield (x_data[indices], y_data[indices]) + tuple(
                d[indices] for d in self.dataset.tensors[1:]
            )


def train_loop(
    model: nn.Module,
    criterion: nn.Module,
    optim: Optimizer,
    data_loader: DataLoader,
    training_t: Transform,
    training_i: int,
    criterion_t: Transform,
    criterion_i: int,
    use_cuda: bool,
):
    """Iterate through training data, compute losses and take gradient steps

    :param model: a torch Module that can take input data and return the prediction
    :param criterion: a loss function
    :param optim: a torch Optimizer
    :param data_loader: training dataset. Should produce a tuple of tensors: the first
                        is used as input and the last is the target. If the tuple has
                        only one element then it's used for both
    :param training_t: Transformation to the data when training the model
    :param training_i: index of the data (from DataLoader tuple) when training the model
    :param criterion_t: Transformation to the data when scoring the output
    :param criterion_i: index of the data (from DataLoader tuple) to score against
    :param use_cuda: whether to use the GPU
    :return: total loss for the epoch, averaged over the number of batches
    """
    total_epoch_loss = 0.0

    for data in data_loader:
        if use_cuda:
            data = tuple(x.cuda() for x in data)

        y = model(training_t(data[training_i]))
        loss = criterion(y, criterion_t(data[criterion_i]))

        total_epoch_loss += loss.data.item()

        optim.zero_grad()
        loss.backward()
        optim.step()

    return total_epoch_loss / len(data_loader)


def validate_loop(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: DataLoader,
    training_t: Transform,
    training_i: int,
    criterion_t: Transform,
    criterion_i: int,
    use_cuda: bool,
):
    """Iterate through test data and compute losses

    :param model: a torch Module that can take input data and return the prediction
    :param criterion: a loss function
    :param data_loader: validation dataset. Should produce a tuple of tensors: the first
                        is used as input and the last is the target. If the tuple has
                        only one element then it's used for both
    :param training_t: Transformation to the data when training the model
    :param training_i: index of the data (from DataLoader tuple) when training the model
    :param criterion_t: Transformation to the data when scoring the output
    :param criterion_i: index of the data (from DataLoader tuple) to score against
    :param use_cuda: whether to use the GPU
    :return: total loss for the epoch, averaged over the number of batches
    """
    total_epoch_loss = 0.0

    for data in data_loader:
        if use_cuda:
            data = tuple(x.cuda() for x in data)

        y = model(training_t(data[training_i]))
        loss = criterion(y, criterion_t(data[criterion_i]))

        total_epoch_loss += loss.data.item()

    return total_epoch_loss / len(data_loader)


def train_until_plateau(
    model: nn.Module,
    criterion: nn.Module,
    optim: Optimizer,
    training_data: DataLoader,
    validation_data: DataLoader,
    training_t: Transform = None,
    training_i: int = 0,
    criterion_t: Transform = None,
    criterion_i: int = 1,
    evaluation_t: Transform = None,
    evaluation_i: Sequence[int] = (1, 2),
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
    :param training_i: Which of the tensors to use as the training target
    :param criterion_t: Transformation to the data when scoring the output
    :param criterion_i: Which of the tensors to use for scoring
    :param evaluation_t: Transformation to the data when evaluating ground truth
    :param evaluation_i: Which of the tensors to use for evaluation (can be multiple)
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
    if criterion_t is None:
        criterion_t = lambda x: x
    if evaluation_t is None:
        evaluation_t = lambda x: x

    if scheduler_kw is None:
        scheduler_kw = dict()

    train_loss = []
    val_loss = []
    train_eval = defaultdict(list)
    val_eval = defaultdict(list)

    scheduler = CosineWithRestarts(optim, **scheduler_kw)
    best = np.inf
    rel_epsilon = 1.0 - threshold
    cycle = 0

    for epoch in itertools.count():
        optim.zero_grad()  # just make sure things are zeroed before train loop
        model.train()

        train_loss.append(
            train_loop(
                model=model,
                criterion=criterion,
                optim=optim,
                data_loader=training_data,
                training_t=training_t,
                training_i=training_i,
                criterion_t=criterion_t,
                criterion_i=criterion_i,
                use_cuda=use_cuda,
            )
        )

        model.eval()
        val_loss.append(
            validate_loop(
                model=model,
                criterion=criterion,
                data_loader=validation_data,
                training_t=training_t,
                training_i=training_i,
                criterion_t=criterion_t,
                criterion_i=criterion_i,
                use_cuda=use_cuda,
            )
        )

        for eval_i in evaluation_i:
            train_eval[eval_i].append(
                validate_loop(
                    model=model,
                    criterion=criterion,
                    data_loader=training_data,
                    training_t=training_t,
                    training_i=training_i,
                    criterion_t=evaluation_t,
                    criterion_i=eval_i,
                    use_cuda=use_cuda,
                )
            )

            val_eval[eval_i].append(
                validate_loop(
                    model=model,
                    criterion=criterion,
                    data_loader=validation_data,
                    training_t=training_t,
                    training_i=training_i,
                    criterion_t=evaluation_t,
                    criterion_i=eval_i,
                    use_cuda=use_cuda,
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

    return train_loss, val_loss, train_eval, val_eval
