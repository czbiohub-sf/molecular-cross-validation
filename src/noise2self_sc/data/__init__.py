#!/usr/bin/env python

import numpy as np

import torch

from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler

from noise2self_sc.data.simulate import simulate_classes


def split_dataset(
    *xs: torch.Tensor,
    batch_size: int,
    indices: np.ndarray = None,
    n_train: int = None,
    noise2self: bool = False,
):
    if indices is None:
        indices = np.random.permutation(xs[0].shape[0])

    if n_train is None:
        n_train = int(0.875 * xs[0].shape[0])

    if noise2self:
        ds = TensorDataset(xs[0] + xs[1], *xs[2:])
        dataloader_cls = Noise2SelfDataLoader
    else:
        ds = TensorDataset(*xs)
        dataloader_cls = DataLoader

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
