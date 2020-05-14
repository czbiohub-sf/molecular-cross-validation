#!/usr/bin/env python

import itertools

from typing import Callable, Sequence, Tuple, Type

import numpy as np

import torch
import torch.nn as nn

from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler

Transform = Callable[[torch.Tensor], torch.Tensor]


def split_dataset(
    *xs: torch.Tensor,
    batch_size: int,
    indices: np.ndarray = None,
    n_train: int = None,
    dataloader_cls: Type[DataLoader] = DataLoader,
):
    if indices is None:
        indices = np.random.permutation(xs[0].shape[0])

    if n_train is None:
        n_train = int(0.875 * xs[0].shape[0])

    ds = TensorDataset(*xs)

    training_dl = dataloader_cls(
        dataset=ds,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(indices[:n_train]),
    )

    validation_dl = dataloader_cls(
        dataset=ds,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(indices[n_train:]),
    )

    return training_dl, validation_dl


def evaluate_epoch(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: DataLoader,
    input_t: Transform,
    eval_i: Sequence[int] = (0,),
    optim: Optimizer = None,
    clip_norm: float = None,
):
    """Iterate through training data, compute losses and take gradient steps

    :param model: a torch Module that can take input data and return the prediction
    :param criterion: a loss function
    :param data_loader: training dataset. Should produce a tuple of at least one tensor,
                        which will be the input to the model
    :param input_t: Transformation to apply to the input
    :param eval_i: Indices into the data tuple for evaluation, defaults to the input
    :param optim: If a torch Optimizer, this will compute gradients and adjust weights
    :param clip_norm: clip gradient norm to a given absolute value
    :return: total loss for the epoch, averaged over the number of batches
    """
    total_epoch_loss = 0.0

    for data in data_loader:
        y = model(input_t(data[0]))
        loss = criterion(y, *(data[i] for i in eval_i))

        total_epoch_loss += loss.data.item()

        if optim is not None:
            optim.zero_grad()
            loss.backward()
            if clip_norm is not None:
                clip_grad_norm_(model.parameters(), clip_norm)
            optim.step()

    return total_epoch_loss / len(data_loader)


def train_until_plateau(
    model: nn.Module,
    training_loss: nn.Module,
    optim: Optimizer,
    training_data: DataLoader,
    validation_data: DataLoader,
    input_t: Transform,
    eval_i: Sequence[int] = (0,),
    min_cycles: int = 3,
    threshold: float = 0.01,
    scheduler_kw: dict = None,
    verbose: bool = False,
) -> Tuple[list, list]:
    """Train a model with cosine scheduling until validation loss stabilizes. This
    function uses CosineAnnealingWarmRestarts to train until the learning rate stops
    improving.

    :param model: torch Module that can take input data and return the prediction
    :param training_loss: The loss function used for training the model
    :param optim: torch Optimizer (will zero the gradient after testing)
    :param training_data: Training dataset. Should produce tuples of Tensors, all but
                          the last are considered to be input and the last is the target
    :param validation_data: Validation dataset in the same format
    :param input_t: Function to apply to the input
    :param eval_i: Indices into the data for evaluation, defaults to the input
    :param min_cycles: Minimum number of cycles to run before checking for convergence
    :param threshold: Tolerance threshold for calling convergence
    :param scheduler_kw: dictionary of keyword arguments for LR scheduler
    :param verbose: Print training progress to stdout
    :return: Lists of training and validation loss and correlation values
    """

    assert 0.0 <= threshold < 1.0

    if scheduler_kw is None:
        scheduler_kw = dict()

    train_loss = []
    val_loss = []

    scheduler = CosineAnnealingWarmRestarts(optim, **scheduler_kw)
    best = np.inf
    rel_epsilon = 1.0 - threshold
    neg_epsilon = 1.0 + threshold
    cycle = 0

    for epoch in itertools.count():
        optim.zero_grad()  # just make sure things are zeroed before train loop
        model.train()

        train_loss.append(
            evaluate_epoch(
                model=model,
                criterion=training_loss,
                data_loader=training_data,
                input_t=input_t,
                eval_i=eval_i,
                optim=optim,
                clip_norm=100.0,
            )
        )

        model.eval()
        val_loss.append(
            evaluate_epoch(
                model=model,
                criterion=training_loss,
                data_loader=validation_data,
                input_t=input_t,
                eval_i=eval_i,
            )
        )

        scheduler.step()
        if scheduler.T_cur == 0:
            if verbose:
                print(
                    f"[epoch {epoch:03d}]  average training loss: {train_loss[-1]:.5f}"
                )
            cycle += 1

            if 0 <= val_loss[-1] < best * rel_epsilon:
                best = val_loss[-1]
            elif 0 > val_loss[-1] and val_loss[-1] < best * neg_epsilon:
                best = val_loss[-1]
            elif cycle >= min_cycles:
                break

    return train_loss, val_loss
