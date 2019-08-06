#!/usr/bin/env python

import argparse
import logging
import pathlib
import pickle

import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, required=True)

    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--output_dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--max_time", type=int, default=10, help="Maximum diffusion time"
    )

    diff_op_group = parser.add_argument_group(
        "knn", description="Parameters for computing the diffusion operator"
    )
    diff_op_group.add_argument(
        "--n_components", type=int, default=30, help="Number of components to compute"
    )
    diff_op_group.add_argument(
        "--n_neighbors", type=int, default=15, help="Neighbors for kNN graph"
    )
    diff_op_group.add_argument(
        "--lazy_p", type=float, default=0.25, help="Lazy p argument?"
    )

    loss_group = parser.add_mutually_exclusive_group(required=True)
    loss_group.add_argument(
        "--mse",
        action="store_const",
        const="mse",
        dest="loss",
        help="mean-squared error",
    )
    loss_group.add_argument(
        "--pois",
        action="store_const",
        const="pois",
        dest="loss",
        help="poisson likelihood",
    )

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    dataset_name = args.dataset.name.split("_")[0]
    output_file = args.output_dir / f"{args.loss}_diffusion_{args.seed}.pickle"

    logger.info(f"writing output to {output_file}")

    seed = sum(map(ord, f"biohub_{args.seed}"))

    np.random.seed(seed)

    with open(args.dataset, "rb") as f:
        true_means, expected_sqrt_umis, umis_X, umis_Y = pickle.load(f)

    # calculate diffusion operator
    n_counts = np.median((umis_X + umis_Y).sum(axis=1)) / 2

    x1_norm = umis_X / umis_X.sum(axis=1, keepdims=True) * n_counts

    U, S, V = randomized_svd(np.sqrt(x1_norm), args.n_components)

    Xp = U.dot(np.diag(S))

    nbrs = NearestNeighbors(n_neighbors=args.n_neighbors).fit(Xp)

    diff_op = np.array(nbrs.kneighbors_graph(Xp, mode="connectivity").todense())
    diff_op += diff_op.T
    diff_op /= diff_op.sum(axis=1, keepdims=True)

    diff_op = (1 - args.lazy_p) * diff_op + args.lazy_p * np.eye(diff_op.shape[0])


    if args.loss == "mse":
        loss = mean_squared_error
        umis_X = np.sqrt(umis_X)
        umis_Y = np.sqrt(umis_Y)
    else:
        assert args.loss == "pois"
        loss = lambda y_true, y_pred: (y_true - y_pred * np.log(y_pred + 1e-6)).mean()
        umis_X = np.sqrt(umis_X)

    diff_X = umis_X.copy().astype(np.float)

    # perform diffusion over the knn graph
    t_range = np.arange(args.max_time)

    re_loss = []
    ss_loss = []
    gt_loss = []

    for t in t_range:
        re_loss.append(loss(umis_X, diff_X))
        ss_loss.append(loss(umis_Y, diff_X))
        if expected_sqrt_umis is not None:
            gt_loss.append(loss(expected_sqrt_umis, diff_X))
        diff_X = diff_op.dot(diff_X)

    t_opt = t_range[np.argmin(ss_loss)]
    logger.info(f"Optimal diffusion time: {t_opt}")

    results = {
        "dataset": dataset_name,
        "method": "diffusion",
        "loss": args.loss,
        "normalization": "sqrt",
        "param_range": t_range,
        "re_loss": re_loss,
        "ss_loss": ss_loss,
        "gt_loss": gt_loss or None,
    }

    with open(output_file, "wb") as out:
        pickle.dump(results, out)
