#!/usr/bin/env python

import argparse
import logging
import pathlib
import pickle

import numpy as np
import scipy.sparse

import scanpy as sc

from noise2self_sc.util import poisson_fit


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--input_h5ad", type=pathlib.Path, required=True)
    parser.add_argument("--output_dir", type=pathlib.Path, required=True)

    data_group = parser.add_argument_group("Parameters for dataset")

    data_group.add_argument("--n_cells", type=int, help="Number of cells to select")
    data_group.add_argument("--n_genes", type=int, help="Number of genes to select")
    data_group.add_argument("--min_counts", type=int, help="Minimum counts per cell")
    data_group.add_argument("--min_genes", type=int, help="Minimum genes per cell")
    data_group.add_argument("--min_cells", type=int, help="Minimum cells per gene")
    data_group.add_argument("--subsample", type=int, help="Number of UMIs to subsample")

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    seed = sum(map(ord, f"biohub_{args.seed}"))

    dataset_file = args.output_dir / f"dataset_{args.seed}.pickle"

    logger.info("loading h5ad")
    data = sc.read(args.input_h5ad)

    if args.min_counts:
        sc.pp.filter_cells(data, min_counts=args.min_counts)

    if args.min_genes:
        sc.pp.filter_cells(data, min_genes=args.min_genes)

    if args.min_cells:
        sc.pp.filter_genes(data, min_cells=args.min_cells)

    if scipy.sparse.issparse(data.X):
        umis = np.asarray(data.X.astype(int).todense())
    else:
        umis = np.asarray(data.X.astype(int))

    # take top cells by umi count
    if args.n_cells and args.n_cells < umis.shape[0]:
        # count umis per cell
        cell_count = umis.sum(1)

        top_cells = cell_count >= sorted(cell_count)[-args.n_cells]

        logger.info(f"filtered to {args.n_cells} deepest cells")
        umis = umis[top_cells, :]

    # take most variable genes by poisson fit
    if args.n_genes and args.n_genes < umis.shape[1]:
        # compute deviation from poisson model
        exp_p = poisson_fit(umis)

        top_genes = exp_p < sorted(exp_p)[args.n_genes]

        logger.info(f"filtering to {args.n_genes} genes")
        umis = umis[:, top_genes]

    # calculating expected means from deep data
    true_means = umis / umis.sum(1, keepdims=True)

    if args.subsample:
        logger.info(f"downsampling to {args.subsample} counts per cell")
        umis = sc.pp.downsample_counts(
            sc.AnnData(umis),
            args.subsample,
            replace=False,
            copy=True,
            random_state=seed,
        ).X.astype(int)

    logger.info(f"final umi matrix: {umis.shape}")

    with open(dataset_file, "wb") as out:
        pickle.dump((true_means, umis), out)


if __name__ == "__main__":
    main()
