#!/usr/bin/env python

import argparse
import logging
import pathlib
import pickle

import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.utils.extmath import randomized_svd

from noise2self_sc.util import expected_sqrt, convert_expectations


def main():
    parser = argparse.ArgumentParser()

    run_group = parser.add_argument_group("run", description="Per-run parameters")
    run_group.add_argument("--data_seed", type=int, required=True)
    run_group.add_argument("--run_seed", type=int, required=True)
    run_group.add_argument(
        "--data_split", type=float, default=0.9, help="Split for self-supervision"
    )
    run_group.add_argument(
        "--n_trials", type=int, default=10, help="Number of times to resample"
    )

    data_group = parser.add_argument_group(
        "data", description="Input and output parameters"
    )
    data_group.add_argument("--dataset", type=pathlib.Path, required=True)
    data_group.add_argument("--output_dir", type=pathlib.Path, required=True)

    model_group = parser.add_argument_group("model", description="Model parameters")
    model_group.add_argument(
        "--max_components",
        type=int,
        default=50,
        metavar="K",
        help="Number of components to compute",
    )

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    dataset_name = args.dataset.name.split("_")[0]
    output_file = args.output_dir / f"mse_pca_{args.data_seed}_{args.run_seed}.pickle"

    logger.info(f"writing output to {output_file}")

    seed = sum(map(ord, f"biohub_{args.run_seed}"))

    np.random.seed(seed)
    data_rng = np.random.RandomState(args.data_seed)

    with open(args.dataset, "rb") as f:
        true_means, umis = pickle.load(f)

    exp_means = expected_sqrt(true_means * umis.sum(1, keepdims=True))
    exp_split_means = expected_sqrt(
        true_means * (1 - args.data_split) * umis.sum(1, keepdims=True)
    )

    k_range = np.arange(1, args.max_components + 1)

    re_losses = np.empty((args.n_trials, k_range.shape[0]), dtype=float)
    ss_losses = np.empty_like(re_losses)
    gt0_losses = np.empty(k_range.shape[0], dtype=float)
    gt1_losses = np.empty_like(re_losses)

    # calculate gt loss for sweep using full data
    U, S, V = randomized_svd(np.sqrt(umis), n_components=args.max_components)

    for j, k in enumerate(k_range):
        pca_X = U[:, :k].dot(np.diag(S[:k])).dot(V[:k, :])
        gt0_losses[j] = mean_squared_error(exp_means, pca_X)

    # run n_trials for self-supervised sweep
    for i in range(args.n_trials):
        umis_X = data_rng.binomial(umis, args.data_split)
        umis_Y = umis - umis_X

        umis_X = np.sqrt(umis_X)
        umis_Y = np.sqrt(umis_Y)

        U, S, V = randomized_svd(umis_X, n_components=args.max_components)

        for j, k in enumerate(k_range):
            pca_X = U[:, :k].dot(np.diag(S[:k])).dot(V[:k, :])

            re_losses[i, j] = mean_squared_error(umis_X, pca_X)
            ss_losses[i, j] = mean_squared_error(
                umis_Y, convert_expectations(pca_X, args.data_split)
            )
            gt1_losses[i, j] = mean_squared_error(
                exp_split_means, convert_expectations(pca_X, args.data_split)
            )

    results = {
        "dataset": dataset_name,
        "method": "pca",
        "loss": "mse",
        "normalization": "sqrt",
        "param_range": k_range,
        "re_loss": re_losses,
        "ss_loss": ss_losses,
        "gt0_loss": gt0_losses,
        "gt1_loss": gt1_losses,
    }

    with open(output_file, "wb") as out:
        pickle.dump(results, out)


if __name__ == "__main__":
    main()
