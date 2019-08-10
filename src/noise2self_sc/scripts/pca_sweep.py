#!/usr/bin/env python

import argparse
import logging
import pathlib
import pickle

import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.utils.extmath import randomized_svd


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
        true_means, expected_sqrt_half_umis, umis = pickle.load(f)

    re_losses = []
    ss_losses = []
    gt_losses = []

    k_range = np.arange(1, args.max_components + 1)
    if true_means is not None:
        true_means /= true_means.sum(1, keepdims=True)

    for i in range(args.n_trials):
        umis_X = data_rng.binomial(umis, args.data_split)
        umis_Y = umis - umis_X

        umis_X /= umis_X.sum(1, keepdims=True)
        umis_Y /= umis_Y.sum(1, keepdims=True)

        U, S, V = randomized_svd(umis_X, n_components=args.max_components)

        re_loss = []
        ss_loss = []
        gt_loss = []

        for k in k_range:
            pca_X = U[:, :k].dot(np.diag(S[:k])).dot(V[:k, :])
            re_loss.append(mean_squared_error(umis_X, pca_X))
            ss_loss.append(mean_squared_error(umis_Y, pca_X))
            if true_means is not None:
                gt_loss.append(mean_squared_error(true_means, pca_X))

        re_losses.append(re_loss)
        ss_losses.append(ss_loss)
        if true_means is not None:
            gt_losses.append(gt_loss)

    re_loss = np.vstack(re_losses).mean(0)
    ss_loss = np.vstack(ss_losses).mean(0)
    if true_means is not None:
        gt_loss = np.vstack(gt_losses).mean(0)
    else:
        gt_loss = None

    k_opt = k_range[np.argmin(ss_loss)]
    logger.info(f"Optimal number of PCs: {k_opt}")

    results = {
        "dataset": dataset_name,
        "method": "pca",
        "loss": "mse",
        "normalization": "sqrt",
        "param_range": k_range,
        "re_loss": re_loss,
        "ss_loss": ss_loss,
        "gt_loss": gt_loss,
    }

    with open(output_file, "wb") as out:
        pickle.dump(results, out)


if __name__ == "__main__":
    main()
