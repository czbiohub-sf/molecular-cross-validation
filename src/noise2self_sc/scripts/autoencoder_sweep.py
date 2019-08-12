#!/usr/bin/env python

import argparse
import collections
import logging
import pathlib
import pickle
import time

import numpy as np

import torch
import torch.nn as nn

import noise2self_sc as n2s
import noise2self_sc.train

from noise2self_sc.models.autoencoder import CountAutoencoder
from noise2self_sc.train.aggmo import AggMo
from noise2self_sc.util import expected_sqrt, convert_expectations


def main():
    parser = argparse.ArgumentParser()

    run_group = parser.add_argument_group("run", description="Per-run parameters")
    run_group.add_argument("--data_seed", type=int, required=True)
    run_group.add_argument("--run_seed", type=int, required=True)
    run_group.add_argument(
        "--data_split", type=float, default=0.9, help="Split for self-supervision"
    )
    run_group.add_argument("--gpu", type=int, required=True)

    data_group = parser.add_argument_group(
        "data", description="Input and output parameters"
    )
    data_group.add_argument("--dataset", type=pathlib.Path, required=True)
    data_group.add_argument("--output_dir", type=pathlib.Path, required=True)

    model_group = parser.add_argument_group("model", description="Model parameters")

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

    model_group.add_argument(
        "--layers",
        nargs="+",
        type=int,
        metavar="L",
        default=[128],
        help="Layers in the input/output networks",
    )
    model_group.add_argument(
        "--max_bottleneck",
        type=int,
        default=7,
        metavar="B",
        help="max bottleneck (log2)",
    )
    model_group.add_argument(
        "--learning_rate", type=float, default=0.1, metavar="LR", help="learning rate"
    )
    model_group.add_argument(
        "--dropout", type=float, default=0.0, metavar="P", help="dropout probability"
    )

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    logger.info(f"torch version {torch.__version__}")

    output_file = args.output_dir / (
        f"{args.loss}_autoencoder_{args.data_seed}_{args.run_seed}.pickle"
    )

    logger.info(f"writing output to {output_file}")

    seed = sum(map(ord, f"biohub_{args.run_seed}"))

    np.random.seed(seed)
    data_rng = np.random.RandomState(args.data_seed)

    device = torch.device(f"cuda:{args.gpu}")

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)

    with open(args.dataset, "rb") as f:
        true_means, umis = pickle.load(f)

    n_features = umis.shape[-1]
    bottlenecks = [2 ** i for i in range(args.max_bottleneck + 1)]
    bottlenecks.extend(3 * b // 2 for b in bottlenecks[1:-1])
    bottlenecks.sort()

    logger.info(f"testing bottlenecks {bottlenecks}")

    if max(bottlenecks) > max(args.layers):
        raise ValueError("Max bottleneck width is larger than your network layers")

    if args.loss == "mse":
        exp_means = expected_sqrt(
            true_means * args.data_split * umis.sum(1, keepdims=True)
        )
        exp_means = torch.from_numpy(exp_means).to(torch.float).to(device)

        training_t = None
        criterion_t = None
        loss_fn = nn.MSELoss()
    else:
        assert args.loss == "pois"
        exp_means = (
            torch.from_numpy(true_means * umis.sum(1, keepdim=True)).to(torch.float)
        ).to(device)

        training_t = torch.log1p
        criterion_t = None
        loss_fn = nn.PoissonNLLLoss()

    model_factory = lambda bottleneck: CountAutoencoder(
        n_input=n_features,
        n_latent=bottleneck,
        layers=args.layers,
        use_cuda=True,
        dropout_rate=args.dropout,
    )

    optimizer_factory = lambda m: AggMo(
        m.parameters(),
        lr=args.learning_rate,
        betas=[0.0, 0.9, 0.99],
        weight_decay=0.0001,
    )

    results = dict()
    scheduler_kw = {"t_max": 128, "eta_min": args.learning_rate / 100.0, "factor": 1.0}

    with torch.cuda.device(device):
        umis_X = data_rng.binomial(umis, args.data_split)
        umis_Y = umis - umis_X

        if args.loss == "mse":
            umis_X = np.sqrt(umis_X)
            umis_Y = convert_expectations(np.sqrt(umis_Y), 1 - args.data_split)

        umis_X = torch.from_numpy(umis_X).to(torch.float)
        umis_Y = torch.from_numpy(umis_Y).to(torch.float)

        sample_indices = data_rng.permutation(umis.shape[0])
        n_train = int(0.875 * umis.shape[0])

        train_dl, val_dl = noise2self_sc.train.split_dataset(
            umis_X,
            umis_Y,
            exp_means,
            batch_size=len(sample_indices),
            indices=sample_indices,
            n_train=n_train,
        )

        t0 = time.time()

        for b in bottlenecks:
            logger.info(f"testing bottleneck width {b}")
            model = model_factory(b)
            optimizer = optimizer_factory(model)

            results[b] = n2s.train.train_until_plateau(
                model,
                loss_fn,
                optimizer,
                train_dl,
                val_dl,
                training_t=training_t,
                training_i=0,
                criterion_t=criterion_t,
                criterion_i=0,
                evaluation_i=(1, 2),
                min_cycles=3,
                threshold=0.01,
                scheduler_kw=scheduler_kw,
                use_cuda=True,
            )

            logger.debug(f"finished {b} after {time.time() - t0} seconds")

    with open(output_file, "wb") as out:
        pickle.dump(results, out)


if __name__ == "__main__":
    main()
