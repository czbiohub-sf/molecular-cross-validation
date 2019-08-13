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
import torch.nn.functional as func
import torch.utils.data

import noise2self_sc as n2s
import noise2self_sc.train

from noise2self_sc.models.autoencoder import CountAutoencoder
from noise2self_sc.train.aggmo import AggMo
from noise2self_sc.util import expected_sqrt, convert_expectations


class AdjustedMSELoss(object):
    def __init__(self, a: float):
        assert 0.0 < a <= 1.0
        self.a = a

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.detach().cpu()

        if self.a < 1.0:
            y_pred = torch.from_numpy(
                convert_expectations(y_pred.numpy(), self.a, 1 - self.a)
            ).to(torch.float)

        return func.mse_loss(y_pred, y_true)


def poisson_nll_loss_cpu(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    y_pred = y_pred.detach().cpu()

    return func.poisson_nll_loss(y_pred, y_true)


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

    dataset_name = args.dataset.name.split("_")[0]
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

    re_losses = np.empty(len(bottlenecks), dtype=float)
    ss_losses = np.empty_like(re_losses)
    gt0_losses = np.empty_like(re_losses)
    gt1_losses = np.empty_like(re_losses)

    if args.loss == "mse":
        exp_means = expected_sqrt(true_means * umis.sum(1, keepdims=True))
        exp_means = torch.from_numpy(exp_means).to(torch.float)

        exp_split_means = expected_sqrt(
            true_means * (1 - args.data_split) * umis.sum(1, keepdims=True)
        )
        exp_split_means = torch.from_numpy(exp_split_means).to(torch.float)

        normalization = "sqrt"
        loss_fn = nn.MSELoss()
        input_t = torch.nn.Identity()
        eval0_fn = AdjustedMSELoss(1.0)
        eval1_fn = AdjustedMSELoss(args.data_split)
    else:
        assert args.loss == "pois"
        exp_means = true_means * umis.sum(1, keepsdims=True)
        exp_means = torch.from_numpy(exp_means).to(torch.float)
        exp_split_means = exp_means

        normalization = "log1p"
        loss_fn = nn.PoissonNLLLoss()
        input_t = torch.log1p
        eval0_fn = poisson_nll_loss_cpu
        eval1_fn = poisson_nll_loss_cpu

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

    scheduler_kw = {"t_max": 128, "eta_min": args.learning_rate / 100.0, "factor": 1.0}

    train_losses = []
    val_losses = []

    with torch.cuda.device(device):
        umis_X = data_rng.binomial(umis, args.data_split)
        umis_Y = umis - umis_X

        if args.loss == "mse":
            umis_X = np.sqrt(umis_X)
            umis_Y = np.sqrt(umis_Y)

        umis_X = torch.from_numpy(umis_X).to(torch.float).to(device)
        umis_Y = torch.from_numpy(umis_Y).to(torch.float)

        sample_indices = data_rng.permutation(umis.shape[0])
        n_train = int(0.875 * umis.shape[0])

        train_dl, val_dl = noise2self_sc.train.split_dataset(
            umis_X,
            umis_Y,
            exp_split_means,
            batch_size=len(sample_indices),
            indices=sample_indices,
            n_train=n_train,
        )

        gt_dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(umis_X, exp_means),
            batch_size=exp_means.size(0),
        )

        t0 = time.time()

        for j, b in enumerate(bottlenecks):
            logger.info(f"testing bottleneck width {b}")
            model = model_factory(b)
            optimizer = optimizer_factory(model)

            train_loss, val_loss = n2s.train.train_until_plateau(
                model,
                loss_fn,
                optimizer,
                train_dl,
                val_dl,
                input_t=input_t,
                min_cycles=3,
                threshold=0.01,
                scheduler_kw=scheduler_kw,
            )
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            logger.debug(f"finished {b} after {time.time() - t0} seconds")

            re_losses[j] = train_loss[-1]
            ss_losses[j] = n2s.train.evaluate_epoch(
                model, eval1_fn, train_dl, input_t, eval_i=1
            )

            gt0_losses[j] = n2s.train.evaluate_epoch(
                model, eval0_fn, gt_dl, input_t, eval_i=1
            )
            gt1_losses[j] = n2s.train.evaluate_epoch(
                model, eval1_fn, train_dl, input_t, eval_i=2
            )

    results = {
        "dataset": dataset_name,
        "method": "autoencoder",
        "loss": args.loss,
        "normalization": normalization,
        "param_range": bottlenecks,
        "re_loss": re_losses,
        "ss_loss": ss_losses,
        "gt0_loss": gt0_losses,
        "gt1_loss": gt1_losses,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    with open(output_file, "wb") as out:
        pickle.dump(results, out)


if __name__ == "__main__":
    main()
