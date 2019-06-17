#!/usr/bin/env python

import argparse
import pathlib
import pickle

import numpy as np
import scipy.sparse
import scipy.stats

import noise2self_sc as n2s

import scanpy as sc

import math


def poisson_fit(gene_reads):
    n_cells = gene_reads.shape[0]
    pct = (gene_reads > 0).sum(0) / n_cells
    exp = gene_reads.sum(0) / gene_reads.sum()
    numis = gene_reads.sum(1)

    prob_zero = np.exp(-np.dot(exp[:, None], numis[None, :]))
    exp_pct_nz = (1 - prob_zero).mean(1)

    var_pct_nz = (prob_zero * (1 - prob_zero)).mean(1) / n_cells
    std_pct_nz = np.sqrt(var_pct_nz)

    exp_p = np.ones_like(pct)
    ix = std_pct_nz != 0
    exp_p[ix] = scipy.stats.norm.cdf(pct[ix], loc=exp_pct_nz[ix], scale=std_pct_nz[ix])

    return exp_p


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
    parser.add_argument("--input_h5ad", type=pathlib.Path, default=None)
    parser.add_argument("--output_dir", type=pathlib.Path, required=True)

    data_group = parser.add_argument_group("Parameters for dataset")

    data_group.add_argument("--n_cells", type=int, help="Number of cells to select")
    data_group.add_argument("--n_genes", type=int, help="Number of genes to select")
    data_group.add_argument("--min_counts", type=int, help="Minimum counts per cell")
    data_group.add_argument("--min_genes", type=int, help="Minimum genes per cell")
    data_group.add_argument("--subsample", type=int, help="Number of UMIs to subsample")

    args = parser.parse_args()

    seed = sum(map(ord, f"biohub_{args.seed}"))

    dataset_file = args.output_dir / f"dataset_{args.seed}.pickle"
    training_index_file = args.output_dir / f"training_index_{args.seed}.pickle"

    np.random.seed(seed)

    print("loading h5ad")
    data = sc.read(args.input_h5ad)

    if args.min_counts:
        sc.pp.filter_cells(data, min_counts=args.min_counts)

    if args.min_genes:
        sc.pp.filter_cells(data, min_genes=args.min_genes)

    if scipy.sparse.issparse(data.X):
        umis = np.asarray(data.X.astype(int).todense())
    else:
        umis = np.asarray(data.X.astype(int))

    # take top cells by umi count
    if args.n_cells < umis.shape[0]:
        # count umis per cell
        cell_count = umis.sum(1)

        top_cells = cell_count >= sorted(cell_count)[-args.n_cells]

        print(f"filtered to {args.n_cells} deepest cells")
        umis = umis[top_cells, :]

    # take most variable genes by poisson fit
    if args.n_genes < umis.shape[1]:
        # compute deviation from poisson model
        exp_p = poisson_fit(umis)

        top_genes = exp_p < sorted(exp_p)[args.n_genes]

        print(f"filtering to {args.n_genes} genes")
        umis = umis[:, top_genes]

    # calculating expected means from deep data
    true_means = umis / umis.sum(1, keepdims=True)

    if args.subsample:
        print(f"downsampling to {args.subsample} counts per cell")
        umis = sc.pp.downsample_counts(
            sc.AnnData(umis), args.subsample, replace=False, copy=True
        ).X.astype(int)

    umi_means = 0.5 * true_means * umis.sum(1, keepdims=True)
    expected_sqrt_umis = expected_sqrt(umi_means)

    print(f"final umi matrix: {umis.shape}")

    print("making n2s split")
    umis_X = np.random.binomial(umis, 0.5)
    umis_Y = umis - umis_X

    with open(dataset_file, "wb") as out:
        pickle.dump((true_means, expected_sqrt_umis, umis_X, umis_Y), out)

    example_indices = np.random.permutation(umis.shape[0])
    n_train = int(0.875 * umis.shape[0])

    with open(training_index_file, "wb") as out:
        pickle.dump((example_indices, n_train), out)
