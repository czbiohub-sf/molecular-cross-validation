#!/usr/bin/env python

import numpy as np

import torch

from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler

from noise2self_sc.data.simulate import simulate_classes


def split_dataset(
    *xs: torch.Tensor, batch_size: int, train_p: float, use_cuda: bool = False
):
    n_cells = xs[0].shape[0]

    example_indices = np.random.permutation(n_cells)
    n_train = int(train_p * n_cells)

    dataset = TensorDataset(*xs)

    data_loader_train = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=use_cuda,
        sampler=SubsetRandomSampler(example_indices[:n_train]),
    )

    data_loader_test = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=use_cuda,
        sampler=SubsetRandomSampler(example_indices[n_train:]),
    )

    return data_loader_train, data_loader_test
