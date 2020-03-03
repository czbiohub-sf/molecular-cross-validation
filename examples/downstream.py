import pandas as pd
import scanpy as sc
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

import numpy as np

from sweep import poisson_nll_loss

from tqdm import tqdm

import warnings
from numba.errors import NumbaPerformanceWarning

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

def compute_gt_mse(adata_denoised, adata):
    X_true = adata.obsm['X_true']
    X = adata_denoised.X

    X_true = X_true * (X.sum(1, keepdims=True) / X_true.sum(1, keepdims=True))
    adata_denoised.uns['gt_mse'] = mean_squared_error(X, X_true)


def compute_gt_poisson(adata_denoised, adata, p=0.9):
    X_true = adata.obsm['X_true']
    X = adata_denoised.X

    X_pred = X * (X_true.sum(1, keepdims=True) / X.sum(1, keepdims=True))
    adata_denoised.uns['gt_lik'] = poisson_nll_loss(X_pred, X_true)

    X_true = X_true * (
                X.sum(1, keepdims=True) / X_true.sum(1, keepdims=True)) * (
                         1 - p)
    adata_denoised.uns['gt_poisson'] = poisson_nll_loss(X * (1 - p), X_true)


def gene_gene_corr(adata, mat_key=None, sqrt=False):
    X = adata.X
    if sqrt:
        X = np.sqrt(X)
    if mat_key:
        X = adata.obsm[mat_key]
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr)
    adata.varm['ggcorr'] = corr


def corr_mat_dist(cor1, cor2):
    return 1 - np.trace(cor1.dot(cor2)) / np.sqrt(
        np.trace(cor1.dot(cor1)) * np.trace(cor2.dot(cor2)))


def gene_gene_corr_eb(adata, mat_key=None):
    """For count data, the covariances are overdispersed."""
    X = adata.X
    if mat_key:
        X = adata.obsm[mat_key]

    cov = np.cov(X, rowvar=False)
    means = np.mean(X, axis=0)

    var_adjusted = np.diag(cov) - means
    cov_adjusted = cov - np.diag(means)

    # Don't adjust if it's too small
    var_adjusted = np.maximum(var_adjusted, cov_adjusted.max(axis=0))

    cov_adjusted = cov - np.diag(np.diag(cov)) + np.diag(var_adjusted) + 1e-6
    assert np.all(np.diag(cov_adjusted) > 0)

    corr = cov_adjusted / (
        np.sqrt(var_adjusted[:, np.newaxis] * var_adjusted[np.newaxis, :]))

    adata.varm['ggcorr_eb'] = corr

from sklearn.cluster import KMeans

def compute_cluster(adata,
                    sqrt=True,
                    n_comps=100,
                    n_neighbors=10,
                    target_n_clusters=None,
                    kmeans_clusters=None,
                    metric='euclidean'):
    X = adata.X.copy()
    if sqrt:
        sc.pp.sqrt(adata)

    if 'X_latent' not in adata.obsm:
        sc.tl.pca(adata, n_comps=n_comps, zero_center=False)
        adata.obsm['X_latent'] = adata.obsm['X_pca']

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X_latent',
                    metric=metric)

    resolution = 1

    sc.tl.leiden(adata, resolution=resolution, key_added='leiden1')

    adata.obs['leiden'] = adata.obs['leiden1']

    if target_n_clusters:
        while True:
            n_clusters = len(np.unique(adata.obs['leiden']))
            if n_clusters < target_n_clusters:
                resolution = resolution * 1.1
                sc.tl.leiden(adata, resolution=resolution)
            elif n_clusters > target_n_clusters * 1.5:
                resolution = resolution * 0.9
                sc.tl.leiden(adata, resolution=resolution)
            else:
                break

    sc.tl.umap(adata)
    adata.uns['n_clusters'] = len(np.unique(adata.obs['leiden']))

    if kmeans_clusters:
        kmeans = KMeans(n_clusters=kmeans_clusters)
        adata.obs['kmeans'] = kmeans.fit_predict(adata.obsm['X_latent'])
        adata.uns['n_clusters'] = len(np.unique(adata.obs['leiden']))

    # restore original values of X
    adata.X = X

from sklearn.metrics import adjusted_rand_score

def clust_to_label(clusters, labels):
    "Given a clustering and a labeling, map each cluster to its most frequent label."
    ct = pd.crosstab(clusters, labels)
    consensus_label = np.argmax(pd.crosstab(clusters, labels).values, axis = 1)
    remapper = dict(zip(ct.index, ct.columns[consensus_label]))
    new_labels = clusters.map(remapper)
    return new_labels

def induced_label_ari(clusters, labels):
    return adjusted_rand_score(clust_to_label(clusters, labels), labels)


def compute_ari(adata, key):

    if 'leiden' in adata.obs.columns:
        adata.uns['imputed_ari'] = induced_label_ari(
            adata.obs['leiden'], adata.obs[key])
        adata.uns['ari'] = adjusted_rand_score(
            adata.obs['leiden'], adata.obs[key])

    if 'leiden1' in adata.obs.columns:
        adata.uns['imputed_ari1'] = induced_label_ari(
            adata.obs['leiden1'], adata.obs[key])
        adata.uns['ari1'] = adjusted_rand_score(
            adata.obs['leiden1'], adata.obs[key])
    if 'kmeans' in adata.obs.columns:
        adata.uns['ari_kmeans'] = adjusted_rand_score(
            adata.obs['kmeans'], adata.obs[key])


def extract_best(denoised):
    """Extract adata with best mcv from a list"""
    best_mcv = np.inf
    best_adata = None

    for adata in denoised:
        if best_mcv > adata.uns['mcv']:
            best_mcv = adata.uns['mcv']
            best_adata = adata
    return best_adata


def plot_scalar(denoised, field, ax=None):
    """Plot a scalar against the parameters used to sweep."""
    params = []
    values = []

    for adata in denoised:
        params.append(adata.uns['denoiser_param'])
        values.append(adata.uns[field])
    if ax is None:
        plt.plot(params, values)
        plt.title(field)
    else:
        ax.plot(params, values)
        ax.set_title(field)

def plot_scalars(denoised, fields):
    nrow = (len(fields)+2)//3
    ncol = 3
    fig, ax = plt.subplots(nrow, 3, figsize=(4*ncol, 4*nrow))
    for i, field in enumerate(fields):
        plot_scalar(denoised, field, ax[i//3, i % 3])

def eval_denoised(denoised, gt, label, target_n_clusters=None, loss='mse'):
    """Run all postprocessing metrics on a denoised matrix."""
    if 'ggcorr' not in gt.varm:
        gene_gene_corr(gt, mat_key='X_true')

    for adata in denoised:
        if loss == 'mse':
            compute_gt_mse(adata, gt)
        elif loss == 'poisson':
            compute_gt_poisson(adata, gt)
        elif loss == 'scvi':
            compute_gt_scvi(adata, gt)
        else:
            raise NotImplementedError()

        gene_gene_corr(adata)
        adata.uns['ggcorr_error'] = corr_mat_dist(adata.varm['ggcorr'],
                                                  gt.varm['ggcorr'])

        compute_cluster(adata, target_n_clusters=target_n_clusters)
        compute_ari(adata, label)


def cluster_denoised(denoised, true_cluster_label, adaptive=False, kmeans=False, **kwargs):
    n_clusters = len(denoised[0].obs[true_cluster_label].unique())

    print(f"There are {n_clusters} true clusters.")

    for adata in tqdm(denoised):
        target_n_clusters = n_clusters if adaptive else None
        kmeans_clusters = n_clusters if kmeans else None

        compute_cluster(adata, target_n_clusters=target_n_clusters,
                            kmeans_clusters=kmeans_clusters, **kwargs)
        compute_ari(adata, true_cluster_label)