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

    with open(args.dataset, "rb") as f:
        true_means, expected_sqrt_half_umis, umis = pickle.load(f)

    umis_X = np.random.RandomState(args.data_seed).binomial(umis, 0.5)
    umis_Y = umis - umis_X

    umis_X = np.sqrt(umis_X)
    umis_Y = np.sqrt(umis_Y)

    U, S, V = randomized_svd(umis_X, n_components=args.max_components)

    k_range = np.arange(1, args.max_components + 1)

    re_loss = []
    ss_loss = []
    gt_loss = []

    for k in k_range:
        pca_X = U[:, :k].dot(np.diag(S[:k])).dot(V[:k, :])
        re_loss.append(mean_squared_error(umis_X, pca_X))
        ss_loss.append(mean_squared_error(umis_Y, pca_X))
        if expected_sqrt_half_umis is not None:
            gt_loss.append(mean_squared_error(expected_sqrt_half_umis, pca_X))

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


if __name__ == "__main__":
    main()
