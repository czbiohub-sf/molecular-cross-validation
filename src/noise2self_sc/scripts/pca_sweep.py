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
        "--n_trials", type=int, default=10, help="Number of times to resample PCA"
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

    expected_sqrt_full_mean = expected_sqrt(true_means * umis.sum(1, keepdims=True))

    re_losses = np.empty((args.n_trials, args.max_components), dtype=float)
    ss_losses = np.empty((args.n_trials, args.max_components), dtype=float)
    gt_losses = np.empty((args.n_trials, args.max_components), dtype=float)

    k_range = np.arange(1, args.max_components + 1)

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
            gt_losses[i, j] = mean_squared_error(
                expected_sqrt_full_mean,
                convert_expectations(pca_X, args.data_split, 1.0),
            )

    k_opt = k_range[np.argmin(ss_losses.mean(0))]
    logger.info(f"Optimal number of PCs: {k_opt}")

    results = {
        "dataset": dataset_name,
        "method": "pca",
        "loss": "mse",
        "normalization": "sqrt",
        "param_range": k_range,
        "re_loss": re_losses,
        "ss_loss": ss_losses,
        "gt_loss": gt_losses,
    }

    with open(output_file, "wb") as out:
        pickle.dump(results, out)


if __name__ == "__main__":
    main()
