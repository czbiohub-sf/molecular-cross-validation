#!/usr/bin/env python

import argparse
import logging
import pathlib
import pickle

import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd

import noise2self_sc.util as ut


def compute_diff_op(
    umis: np.ndarray,
    n_components: int,
    n_neighbors: int,
    tr_prob: float,
    random_state: np.random.RandomState,
):
    # calculate diffusion operator
    n_counts = np.median(umis.sum(axis=1))

    x1_norm = np.sqrt(umis / umis.sum(axis=1, keepdims=True) * n_counts)

    U, S, V = randomized_svd(x1_norm, n_components, random_state=random_state)

    p = U.dot(np.diag(S))

    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(p)

    diff_op = np.array(nbrs.kneighbors_graph(p, mode="connectivity").todense())
    diff_op += diff_op.T
    diff_op = diff_op / diff_op.sum(axis=1, keepdims=True)
    diff_op = tr_prob * diff_op + (1 - tr_prob) * np.eye(diff_op.shape[0])

    return diff_op


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

    diff_op_group = model_group.add_argument_group(
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

    dataset_name = args.dataset.parent.name
    output_file = args.output_dir / f"{args.loss}_diffusion_{args.seed}.pickle"
    logger.info(f"writing output to {output_file}")

    seed = sum(map(ord, f"biohub_{args.seed}"))
    random_state = np.random.RandomState(seed)

    with open(args.dataset, "rb") as f:
        true_means, true_counts, umis = pickle.load(f)

    t_range = np.arange(args.max_time + 1)

    rec_loss = np.empty((args.n_trials, t_range.shape[0]), dtype=float)
    mcv_loss = np.empty_like(rec_loss)
    gt0_loss = np.empty(t_range.shape[0], dtype=float)
    gt1_loss = np.empty_like(rec_loss)

    data_split, data_split_complement, overlap = ut.overlap_correction(
        args.data_split, umis.sum(1, keepdims=True) / true_counts
    )

    if args.loss == "mse":
        exp_means = ut.expected_sqrt(true_means * umis.sum(1, keepdims=True))
        exp_split_means = ut.expected_sqrt(
            true_means * data_split_complement * umis.sum(1, keepdims=True)
        )

        loss = mean_squared_error
        normalization = "sqrt"
    else:
        assert args.loss == "pois"
        exp_means = true_means * umis.sum(1, keepdims=True)
        exp_split_means = data_split_complement * exp_means

        loss = lambda y_true, y_pred: (y_pred - y_true * np.log(y_pred + 1e-6)).mean()
        normalization = "none"

    # calculate gt loss for sweep using full data
    diff_op = compute_diff_op(
        umis, args.n_components, args.n_neighbors, args.tr_prob, random_state
    )

    if args.loss == "mse":
        diff = np.sqrt(umis)
    else:
        diff = umis.copy().astype(np.float)

    for t in t_range:
        gt0_loss[t] = loss(exp_means, diff)
        diff = diff_op.dot(diff)

    # run n_trials for self-supervised sweep
    for i in range(args.n_trials):
        umis_X, umis_Y = ut.split_molecules(umis, data_split, overlap, random_state)

        diff_op = compute_diff_op(
            umis_X, args.n_components, args.n_neighbors, args.tr_prob, random_state
        )

        if args.loss == "mse":
            umis_X = np.sqrt(umis_X)
            umis_Y = np.sqrt(umis_Y)

        diff_X = umis_X.copy().astype(np.float)

        # perform diffusion over the knn graph
        for t in t_range:
            if args.loss == "mse":
                conv_exp = ut.convert_expectations(
                    diff_X, data_split, data_split_complement
                )
            else:
                conv_exp = diff_X / data_split * data_split_complement

            rec_loss[i, t] = loss(umis_X, diff_X)
            mcv_loss[i, t] = loss(umis_Y, conv_exp)
            gt1_loss[i, t] = loss(exp_split_means, conv_exp)

            diff_X = diff_op.dot(diff_X)

    results = {
        "dataset": dataset_name,
        "method": "diffusion",
        "loss": args.loss,
        "normalization": normalization,
        "param_range": t_range,
        "rec_loss": rec_loss,
        "mcv_loss": mcv_loss,
        "gt0_loss": gt0_loss,
        "gt1_loss": gt1_loss,
    }

    with open(output_file, "wb") as out:
        pickle.dump(results, out)


if __name__ == "__main__":
    main()
