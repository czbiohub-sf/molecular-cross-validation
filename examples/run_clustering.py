#!/usr/bin/env python

import argparse
import os

from downstream import cluster_denoised
from sweep import recipe_seurat, mcv_sweep_recipe_pca, sweep_pca_sqrt
from data import DATASETS, load_data
import pickle

# outdir = '../runs/'
# data_dir = '/Users/josh/src/molecular-cross-validation/data'

def main():

    parser = argparse.ArgumentParser()

    run_group = parser.add_argument_group("run", description="Per-run parameters")
    run_group.add_argument(
        "dataset", type=str, help="Dataset"
    )
    run_group.add_argument(
        "max_pcs", type=int, help="Max PCs"
    )
    run_group.add_argument(
        "normalization", type=str, help="Normalization. 'seurat' or 'sqrt'"
    )
    run_group.add_argument(
        "datadir", type=str, help="Data directory"
    )
    run_group.add_argument(
        "outdir", type=str, help="Output directory"
    )

    args = parser.parse_args()

    adata = load_data(args.dataset, args.datadir)

    print("Denoising...")
    if args.normalization == 'seurat':
        denoised = mcv_sweep_recipe_pca(adata,
                                       recipe_seurat,
                                       max_pcs=args.max_pcs,
                                       save_reconstruction=False)
    elif args.normalization == 'sqrt':
        denoised = sweep_pca_sqrt(adata,
                                 max_pcs=args.max_pcs,
                                 p=0.9,
                                 save_reconstruction=False)

    print("Clustering...")
    cluster_denoised(denoised,
                     'cell_type',
                     adaptive=True,
                     kmeans=True)

    with open(os.path.join(args.outdir,
                           args.dataset + '_' +
                           args.normalization + '_' +
                           str(args.max_pcs) +
                           '.pickle'), "wb") as out:
        pickle.dump(denoised, out)

if __name__ == "__main__":
    main()