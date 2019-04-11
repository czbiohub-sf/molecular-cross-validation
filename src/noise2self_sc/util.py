import numpy as np
import scanpy as sc


def normalize_rows(X, n_counts=None):
    if n_counts is None:
        n_counts = np.median(X.sum(axis=1))

    return X / X.sum(axis=1, keepdims=True) * n_counts


def split_counts(X):
    X = X.astype(np.int)
    X1 = np.random.binomial(X, 0.5)
    X2 = X - X1
    return X1, X2


def mse(x, y):
    return ((x - y) ** 2).mean()


def standard_scanpy(adata):

    if adata.raw is None:
        adata.raw = adata
    sc.pp.normalize_per_cell(adata)
    sc.pp.sqrt(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=25)
    sc.tl.umap(adata)

    sc.tl.louvain(adata)

    return adata

def poisson_expected_sqrt(X, n_samples):
    Y = np.zeros(X.shape)
    for i in range(n_samples):
        Y += np.sqrt(np.random.poisson(X))
    return Y/n_samples