#!/usr/bin/env python

import argparse
import logging
import pathlib
import pickle

import numpy as np

from sklearn.metrics import mean_squared_error

import noise2self_sc.util as ut

import magic


def main():
    parser = argparse.ArgumentParser()

    run_group = parser.add_argument_group("run", description="Per-run parameters")
    run_group.add_argument("--seed", type=int, required=True)
    run_group.add_argument(
        "--data_split", type=float, default=0.9, help="Split for self-supervision"
    )
    run_group.add_argument(
        "--n_trials", type=int, default=10, help="Number of times to resample"
    )
    run_group.add_argument("--median_scale", action="store_true")

    data_group = parser.add_argument_group(
        "data", description="Input and output parameters"
    )
    data_group.add_argument("--dataset", type=pathlib.Path, required=True)
    data_group.add_argument("--output_dir", type=pathlib.Path, required=True)
    data_group.add_argument(
        "--genes", type=int, nargs="+", required=True, help="Genes to smooth (indices)"
    )

    model_group = parser.add_argument_group(
        "model",
        description="Model parameters. [max] or [min, max] or [min, max, interval]",
    )

    model_group.add_argument(
        "--neighbors",
        type=int,
        nargs="+",
        default=(1, 11),
        metavar="K",
        help="Number of neighbors in kNN graph",
    )
    model_group.add_argument(
        "--components",
        type=int,
        nargs="+",
        default=(5, 51, 5),
        metavar="PC",
        help="Maximum number of components to compute",
    )
    model_group.add_argument(
        "--time",
        type=int,
        nargs="+",
        default=(1, 6),
        metavar="T",
        help="Number of time steps for diffusion",
    )

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    dataset_name = args.dataset.parent.name
    output_file = args.output_dir / f"{dataset_name}_magic_mse_{args.seed}.pickle"

    logger.info(f"writing output to {output_file}")

    seed = sum(map(ord, f"biohub_{args.seed}"))
    random_state = np.random.RandomState(seed)

    with open(args.dataset, "rb") as f:
        true_means, true_counts, umis = pickle.load(f)

    k_range = np.arange(*args.neighbors)
    pc_range = np.arange(*args.components)
    t_range = np.arange(*args.time)

    rec_loss = dict()
    mcv_loss = dict()

    # run n_trials for self-supervised sweep
    for i in range(args.n_trials):
        umis_X, umis_Y = ut.split_molecules(umis, args.data_split, 0.0, random_state)

        if args.median_scale:
            median_count = np.median(umis.sum(axis=1))

            umis_X = umis_X / umis_X.sum(axis=1, keepdims=True) * median_count
            umis_Y = umis_Y / umis_Y.sum(axis=1, keepdims=True) * median_count
        else:
            umis_Y = umis_Y * args.data_split / (1 - args.data_split)

        for n_pcs in pc_range:
            for k in k_range:
                for t in t_range:
                    magic_op = magic.MAGIC(n_pca=n_pcs, verbose=0)
                    magic_op.set_params(knn=k, t=t)
                    denoised = magic_op.fit_transform(umis_X, genes=args.genes)
                    denoised = np.maximum(denoised, 0)

                    rec_loss[i, n_pcs, k, t] = mean_squared_error(
                        denoised, umis_X[:, args.genes]
                    )
                    mcv_loss[i, n_pcs, k, t] = mean_squared_error(
                        denoised, umis_Y[:, args.genes]
                    )

    results = {
        "dataset": dataset_name,
        "method": "magic",
        "loss": "mse",
        "normalization": "sqrt",
        "param_range": [pc_range, k_range, t_range],
        "rec_loss": rec_loss,
        "mcv_loss": mcv_loss,
    }

    with open(output_file, "wb") as out:
        pickle.dump(results, out)


if __name__ == "__main__":
    main()
