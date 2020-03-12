#!/usr/bin/env python

import argparse
import os

from downstream import cluster_denoised
from sweep import recipe_seurat, mcv_sweep_recipe_pca, sweep_pca_sqrt, sweep_diffusion_sqrt
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
        "max_param", type=int, help="Max PCs/Time"
    )
    run_group.add_argument(
        "method", type=str, help="Method. 'seurat' or 'sqrt' for pca, or 'diffusion'"
    )
    run_group.add_argument(
        "datadir", type=str, help="Data directory"
    )
    run_group.add_argument(
        "outdir", type=str, help="Output directory"
    )

    args = parser.parse_args()

    adata = load_data(args.dataset, args.datadir)

    clustering_kwargs = {}
    print("Denoising...")
    if args.method == 'seurat':
        denoised = mcv_sweep_recipe_pca(adata,
                                       recipe_seurat,
                                       max_pcs=args.max_param,
                                       save_reconstruction=False)
    elif args.method == 'sqrt':
        denoised = sweep_pca_sqrt(adata,
                                 max_pcs=args.max_param,
                                 p=0.9,
                                 save_reconstruction=False)
        
    elif args.method == 'diffusion':
        denoised = sweep_diffusion_sqrt(adata,
                                 max_t=args.max_param,
                                 p=0.9,
                                 save_reconstruction=True)
        clustering_kwargs['n_comps'] = 300
        
    elif args.method == 'raw':
        denoised = sweep_raw_sqrt(adata,
                                 max_t=args.max_param,
                                 p=0.9,
                                 save_reconstruction=True)
        clustering_kwargs['n_comps'] = TK

    print("Clustering...")
    cluster_denoised(denoised,
                     'cell_type',
                     adaptive=True,
                     kmeans=True, **clustering_kwargs)

    with open(os.path.join(args.outdir,
                           args.dataset + '_' +
                           args.method + '_' +
                           str(args.max_param) +
                           '.pickle'), "wb") as out:
        pickle.dump(denoised, out)

if __name__ == "__main__":
    main()