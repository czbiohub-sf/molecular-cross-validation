#!/usr/bin/env python

import argparse
import pathlib
import pickle

import numpy as np

from simscity import latent, drug, sequencing


def simulate_classes(
    n_classes: int,
    n_latent: int,
    n_cells_per_class: int,
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

    n_cells = n_classes * n_cells_per_class

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


def main():
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

    np.random.seed(seed)

    exp, class_labels, programs, lib_size, umis = simulate_classes(
        args.n_classes,
        args.n_latent,
        args.n_cells_per_class,
        args.n_genes,
        prog_kw=dict(scale=3.0 / np.sqrt(args.n_genes), sparsity=1.0),
        class_kw=dict(scale=3.0 / np.sqrt(args.n_latent), sparsity=1.0),
        library_kw=dict(loc=np.log(args.n_genes * 0.5), scale=0.2),
    )

    true_exp = np.dot(exp, programs)  # true expression in log-normal space
    true_means = np.exp(true_exp) / np.exp(true_exp).sum(1, keepdims=True)

    with open(dataset_file, "wb") as out:
        pickle.dump((true_means, np.inf, umis), out)


if __name__ == "__main__":
    main()
