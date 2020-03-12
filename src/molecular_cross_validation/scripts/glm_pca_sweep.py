#!/usr/bin/env python

import argparse
import logging
import pathlib
import pickle

import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd

import molecular_cross_validation.util as ut

from glmpca import glmpca

def glmpca_denoise(umis, n_components, penalty=10):
    Y = umis.T
    success = False
    for i in np.arange(10):
        try:
            output = glmpca.glmpca(Y, n_components, penalty=penalty, fam='poi', verbose=True)
            offsets = np.log(Y.mean(0))
            M = np.exp(output['loadings'].dot(output['factors'].T) + output['coefX'] + offsets)
            success = True
        except:
            penalty = 2*penalty
        if success:
            break
        else:
            M = Y*0
    return M.T

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
        "--max_pcs", type=int, default=10, help="Maximum number of PCs"
    )
    model_group.add_argument(
        "--spacing_pcs", type=int, default=5, help="Spacing between PCs"
    )
    model_group.add_argument(
        "--penalty", type=int, default=1, help="Penalty"
    )

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    dataset_name = args.dataset.parent.name
    output_file = (
        args.output_dir / f"{dataset_name}_glmpca_{args.seed}.pickle"
    )

    logger.info(f"writing output to {output_file}")

    seed = sum(map(ord, f"biohub_{args.seed}"))
    random_state = np.random.RandomState(seed)

    with open(args.dataset, "rb") as f:
        true_means, true_counts, umis = pickle.load(f)

    k_range = np.arange(args.spacing_pcs, args.spacing_pcs + args.max_pcs, args.spacing_pcs)

    rec_loss = np.empty((args.n_trials, k_range.shape[0]), dtype=float)
    mcv_loss = np.empty_like(rec_loss)
    gt0_loss = np.empty(k_range.shape[0], dtype=float)
    gt1_loss = np.empty_like(rec_loss)

    data_split, data_split_complement, overlap = ut.overlap_correction(
        args.data_split, umis.sum(1, keepdims=True) / true_counts
    )
   
    exp_means = true_means * umis.sum(1, keepdims=True)
    exp_split_means = data_split_complement * exp_means

    loss = lambda y_true, y_pred: (y_pred - y_true * np.log(y_pred + 1e-6)).mean()
    normalization = "none"
    
    for j, k in enumerate(k_range):
        print("Beginning k = ", k)
        pca_X = glmpca_denoise(umis, k)
        gt0_loss[j] = loss(exp_means, pca_X)

    # run n_trials for self-supervised sweep
    for i in range(args.n_trials):
        print(k, " trial ", i)
        umis_X, umis_Y = ut.split_molecules(umis, data_split, overlap, random_state)

        for j, k in enumerate(k_range):
            pca_X = glmpca_denoise(umis_X, k, args.penalty)
            conv_exp = pca_X * data_split_complement / data_split 
            
            rec_loss[i, j] = loss(umis_X, pca_X)
            mcv_loss[i, j] = loss(umis_Y, conv_exp)
            gt1_loss[i, j] = loss(exp_split_means, conv_exp)

    results = {
        "dataset": dataset_name,
        "method": "glmpca",
        "loss": "pois",
        "normalization": "none",
        "param_range": k_range,
        "rec_loss": rec_loss,
        "mcv_loss": mcv_loss,
        "gt0_loss": gt0_loss,
        "gt1_loss": gt1_loss,
    }

    with open(output_file, "wb") as out:
        pickle.dump(results, out)

if __name__ == "__main__":
    main()
