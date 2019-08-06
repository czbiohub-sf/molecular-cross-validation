#!/usr/bin/env python

import argparse
import logging
import pathlib
import pickle

import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
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
        "--max_time", type=int, default=10, help="Maximum diffusion time"
    )

    loss_group = model_group.add_mutually_exclusive_group(required=True)
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

    diff_op_group = parser.add_argument_group(
        "diff_op", description="Parameters for computing the diffusion operator"
    )
    diff_op_group.add_argument(
        "--n_components",
        type=int,
        default=30,
        metavar="N",
        help="Number of components to compute",
    )
    diff_op_group.add_argument(
        "--n_neighbors",
        type=int,
        default=15,
        metavar="N",
        help="Neighbors for kNN graph",
    )
    diff_op_group.add_argument(
        "--tr_prob",
        type=float,
        default=0.5,
        help="Transition probability in lazy random walk",
    )

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    dataset_name = args.dataset.name.split("_")[0]
    output_file = args.output_dir / f"{args.loss}_diffusion_{args.run_seed}.pickle"

    logger.info(f"writing output to {output_file}")

    seed = sum(map(ord, f"biohub_{args.run_seed}"))

    np.random.seed(seed)

    with open(args.dataset, "rb") as f:
        true_means, expected_sqrt_half_umis, umis = pickle.load(f)

    umis_X = np.random.RandomState(args.data_seed).binomial(umis, 0.5)
    umis_Y = umis - umis_X

    # calculate diffusion operator
    n_counts = np.median(umis.sum(axis=1)) / 2

    x1_norm = np.sqrt(umis_X / umis_X.sum(axis=1, keepdims=True) * n_counts)

    U, S, V = randomized_svd(x1_norm, args.n_components)

    Xp = U.dot(np.diag(S))

    nbrs = NearestNeighbors(n_neighbors=args.n_neighbors).fit(Xp)

    diff_op = np.array(nbrs.kneighbors_graph(Xp, mode="connectivity").todense())
    diff_op += diff_op.T
    diff_op /= diff_op.sum(axis=1, keepdims=True)

    diff_op = args.tr_prob * diff_op + (1 - args.tr_prob) * np.eye(diff_op.shape[0])

    if args.loss == "mse":
        loss = mean_squared_error
        normalization = "sqrt"
        exp_means = expected_sqrt_half_umis
        umis_X = np.sqrt(umis_X)
        umis_Y = np.sqrt(umis_Y)
    else:
        assert args.loss == "pois"
        loss = lambda y_true, y_pred: (y_pred - y_true * np.log(y_pred + 1e-6)).mean()
        normalization = "none"
        if true_means is not None:
            exp_means = true_means * umis_X.sum(1, keepdims=True)
        else:
            exp_means = None
        # umis_X and umis_Y are kept as counts

    diff_X = umis_X.copy().astype(np.float)

    # perform diffusion over the knn graph
    t_range = np.arange(args.max_time)

    re_loss = []
    ss_loss = []
    gt_loss = []

    for t in t_range:
        re_loss.append(loss(umis_X, diff_X))
        ss_loss.append(loss(umis_Y, diff_X))
        if exp_means is not None:
            gt_loss.append(loss(exp_means, diff_X))
        diff_X = diff_op.dot(diff_X)

    t_opt = t_range[np.argmin(ss_loss)]
    logger.info(f"Optimal diffusion time: {t_opt}")

    results = {
        "dataset": dataset_name,
        "method": "diffusion",
        "loss": args.loss,
        "normalization": normalization,
        "param_range": t_range,
        "re_loss": re_loss,
        "ss_loss": ss_loss,
        "gt_loss": gt_loss or None,
    }

    with open(output_file, "wb") as out:
        pickle.dump(results, out)


if __name__ == "__main__":
    main()
