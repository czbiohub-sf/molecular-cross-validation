#!/usr/bin/env python

import argparse
import math
import pathlib
import pickle

import numpy as np

from simscity import latent, drug, sequencing


def simulate_classes(
    n_classes: int,
    n_latent: int,
    n_cells: int,
    n_features: int,
    prog_kw: dict = None,
    class_kw: dict = None,
    library_kw: dict = None,
):
    prog_kw = (
        dict(scale=1.0 / np.sqrt(n_features), sparsity=1.0)
        if prog_kw is None
        else prog_kw.copy()
    )
    class_kw = (
        dict(scale=1.0 / np.sqrt(n_latent), sparsity=1.0)
        if class_kw is None
        else class_kw.copy()
    )
    library_kw = (
        dict(loc=np.log(0.1 * n_features), scale=0.5)
        if library_kw is None
        else library_kw.copy()
    )

    assert n_cells // n_classes == n_cells / n_classes

    n_cells_per_class = n_cells // n_classes

    programs = latent.gen_programs(n_latent, n_features, **prog_kw)

    classes = latent.gen_classes(n_latent, n_classes, **class_kw)

    class_labels = np.tile(np.arange(n_classes), n_cells_per_class)

    latent_exp = np.empty((n_cells_per_class, n_classes, n_latent))
    for i in np.arange(n_classes):
        latent_exp[:, i, :] = latent.gen_class_samples(n_cells_per_class, classes[i, :])

    exp = np.dot(latent_exp, programs)

    lib_size = sequencing.library_size((n_cells_per_class, n_classes), **library_kw)

    umis = sequencing.umi_counts(np.exp(exp), lib_size=lib_size)

    return (
        latent_exp.reshape(n_cells, n_latent),
        class_labels,
        programs,
        lib_size.reshape(n_cells),
        umis.reshape(n_cells, n_features),
    )


def expected_sqrt(mean):
    """Return expected square root of a poisson distribution. Expects ndarray input.
    Uses Taylor series centered at 0 or mean, as appropriate."""

    truncated_taylor_around_0 = np.zeros(mean.shape)
    nonzeros = mean != 0
    mean = mean + 1e-8
    for k in range(15):
        truncated_taylor_around_0 += mean ** k / math.factorial(k) * np.sqrt(k)

    truncated_taylor_around_0 *= np.exp(-mean)
    truncated_taylor_around_mean = (
        np.sqrt(mean) - np.sqrt(mean) ** (-0.5) / 8 + np.sqrt(mean) ** (-1.5) / 16
    )

    return nonzeros * (
        truncated_taylor_around_0 * (mean < 4)
        + truncated_taylor_around_mean * (mean >= 4)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", type=pathlib.Path, required=True)

    data_group = parser.add_argument_group("Parameters for dataset")

    data_group.add_argument("--n_classes", type=int, default=8)
    data_group.add_argument("--n_latent", type=int, default=8)
    data_group.add_argument("--n_cells_per_class", type=int, default=512)
    data_group.add_argument("--n_genes", type=int, default=512)

    args = parser.parse_args()

    seed = sum(map(ord, f"biohub_{args.seed}"))

    dataset_file = args.output_dir / f"dataset_{args.seed}.pickle"
    training_index_file = args.output_dir / f"training_index_{args.seed}.pickle"

    np.random.seed(seed)

    n_cells = args.n_cells_per_class * args.n_classes

    example_indices = np.random.permutation(n_cells)
    n_train = int(0.875 * n_cells)

    with open(training_index_file, "wb") as out:
        pickle.dump((example_indices, n_train), out)

    exp, class_labels, programs, lib_size, umis = simulate_classes(
        args.n_classes,
        args.n_latent,
        args.n_cells,
        args.n_genes,
        prog_kw=dict(scale=3.0 / np.sqrt(args.n_features), sparsity=1.0),
        class_kw=dict(scale=3.0 / np.sqrt(args.n_latent), sparsity=1.0),
        library_kw=dict(loc=np.log(n_features * 0.5), scale=0.2),
    )

    umis_X = np.random.binomial(umis, 0.5)  # molecular cross-validation
    umis_Y = umis - umis_X

    true_exp = np.dot(exp, programs)  # true expression in log-normal space
    p = np.exp(true_exp) / np.exp(true_exp).sum(1, keepdims=True)
    exp_means = 0.5 * p * lib_size[:, None]  # expected mean of umis_X/umis_Y

    expected_sqrt_umis = expected_sqrt(exp_means)

    with open(dataset_file, "wb") as out:
        pickle.dump((true_exp, expected_sqrt_umis, umis_X, umis_Y), out)
