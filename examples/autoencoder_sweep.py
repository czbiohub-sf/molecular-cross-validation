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

from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler

import noise2self_sc as n2s

from noise2self_sc.train import Noise2SelfDataLoader
from noise2self_sc.train.aggmo import AggMo


parser = argparse.ArgumentParser()

run_group = parser.add_argument_group("run", description="Per-run parameters")
run_group.add_argument("--seed", type=int, required=True)
run_group.add_argument("--gpu", type=int, required=True)

data_group = parser.add_argument_group(
    "data", description="Input and output parameters"
)
data_group.add_argument("--workset", type=pathlib.Path, required=True)
data_group.add_argument("--training", type=pathlib.Path, required=True)
data_group.add_argument("--output_dir", type=pathlib.Path, required=True)

model_group = parser.add_argument_group("model", description="Model parameters")
model_group.add_argument("--layers", nargs="+", type=int, default=[128])
model_group.add_argument(
    "--max_bottleneck", type=int, default=7, help="max bottleneck (log2)"
)
model_group.add_argument("--n2s", action="store_true", help="self-supervised training")
model_group.add_argument("--pois", action="store_true", help="poisson loss")
model_group.add_argument(
    "--learning_rate", type=float, default=0.1, help="learning rate"
)
model_group.add_argument(
    "--dropout", type=float, default=0.0, help="dropout probability"
)


args = parser.parse_args()


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

logger.info(f"torch version {torch.__version__}")

output_file = args.output_dir / (
    f"{'n2s_' if args.n2s else ''}{'pois' if args.pois else 'mse'}_{args.seed}.pickle"
)

logger.info(f"writing output to {output_file}")

seed = sum(map(ord, f"biohub_{args.seed}"))

device = torch.device(f"cuda:{args.gpu}")
np.random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)

with open(args.workset, "rb") as f:
    true_means, expected_sqrt_umis, umis_X, umis_Y = pickle.load(f)

umis_X = torch.from_numpy(umis_X).to(torch.float)
umis_Y = torch.from_numpy(umis_Y).to(torch.float)

n_features = umis_X.shape[-1]
bottlenecks = [2 ** i for i in range(args.max_bottleneck + 1)]
bottlenecks.extend(3 * b // 2 for b in bottlenecks[1:-1])
bottlenecks.sort()

logger.info(f"testing bottlenecks {bottlenecks}")

if max(bottlenecks) > max(args.layers):
    raise ValueError("Max bottleneck width is larger than your network layers")

if args.pois:
    exp_means = (
        torch.from_numpy(true_means).to(torch.float) * umis_X.sum(1, keepdim=True)
    ).to(device)

    training_t = torch.log1p
    criterion_t = lambda x: x
else:
    exp_means = torch.from_numpy(expected_sqrt_umis ** 2).to(torch.float).to(device)

    training_t = torch.sqrt
    criterion_t = torch.sqrt


with open(args.training, "rb") as f:
    sample_indices, n_train = pickle.load(f)

batch_size = len(sample_indices)


def test_bottlenecks(bs, m_factory, opt_factory, criterion, train_data, val_data):
    b_results = dict()
    scheduler_kw = {"t_max": 128, "eta_min": args.learning_rate / 100.0, "factor": 1.0}

    t0 = time.time()

    for b in bs:
        logger.info(f"testing bottleneck width {b}")
        model = m_factory(b)
        optimizer = opt_factory(model)

        b_results[b] = n2s.train.train_until_plateau(
            model,
            criterion,
            optimizer,
            train_data,
            val_data,
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

    return b_results


with torch.cuda.device(device):
    model_factory = lambda bottleneck: n2s.models.CountAutoencoder(
        n_input=n_features,
        n_latent=bottleneck,
        layers=args.layers,
        use_cuda=True,
        dropout_rate=args.dropout,
    )

    optimizer_factory = lambda model: AggMo(
        model.parameters(),
        lr=args.learning_rate,
        betas=[0.0, 0.9, 0.99],
        weight_decay=0.0001,
    )

    if args.pois:
        loss_fn = nn.PoissonNLLLoss()
    else:
        loss_fn = nn.MSELoss()

    train_dl, val_dl = n2s.data.split_dataset(
        umis_X,
        umis_Y,
        exp_means,
        batch_size=batch_size,
        indices=sample_indices,
        n_train=n_train,
        noise2self=args.n2s,
    )

    results = test_bottlenecks(
        bottlenecks, model_factory, optimizer_factory, loss_fn, train_dl, val_dl
    )

with open(output_file, "wb") as out:
    pickle.dump(results, out)
