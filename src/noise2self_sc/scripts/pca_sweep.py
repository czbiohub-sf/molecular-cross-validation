#!/usr/bin/env python

import argparse
import logging
import pathlib
import pickle

import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.utils.extmath import randomized_svd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, required=True)

    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--output_dir", type=pathlib.Path, required=True)

    parser.add_argument(
        "--max_components", type=int, default=50, help="Number of components to compute"
    )

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    dataset_name = args.dataset.name.split("_")[0]
    output_file = args.output_dir / f"mse_pca_{args.seed}.pickle"

    logger.info(f"writing output to {output_file}")

    seed = sum(map(ord, f"biohub_{args.seed}"))

    np.random.seed(seed)

    with open(args.dataset, "rb") as f:
        true_means, expected_sqrt_umis, umis_X, umis_Y = pickle.load(f)

    x1_norm = np.sqrt(umis_X)
    x2_norm = np.sqrt(umis_Y)

    U, S, V = randomized_svd(x1_norm, n_components=args.max_components)

    k_range = np.arange(1, args.max_components + 1)

    re_loss = []
    ss_loss = []
    gt_loss = []

    for k in k_range:
        x_pred = U[:, :k].dot(np.diag(S[:k])).dot(V[:k, :])
        re_loss.append(mean_squared_error(x1_norm, x_pred))
        ss_loss.append(mean_squared_error(x2_norm, x_pred))
        if expected_sqrt_umis is not None:
            gt_loss.append(mean_squared_error(expected_sqrt_umis, x_pred))

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
        "gt_loss": gt_loss or None,
    }

    with open(output_file, "wb") as out:
        pickle.dump(results, out)
