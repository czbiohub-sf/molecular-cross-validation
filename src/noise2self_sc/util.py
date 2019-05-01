import numpy as np
import scanpy as sc
from scipy.special import factorial


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


def expected_sqrt(mean, samples=None):
    """Return expected square root of a poisson distribution. Expects ndarray input.
    
    If samples = None, uses Taylor series centered at 0 or mean, as appropriate.
    
    If samples = int, uses that many samples for an empirical distribution."""
    
    if samples is None:
    
        truncated_taylor_around_0 = np.zeros(mean.shape)
        nonzeros = (mean != 0)
        mean = mean + 1e-8
        small_values = mean*(mean < 4)
        for k in range(15):
            truncated_taylor_around_0 += small_values**k/factorial(k) * np.sqrt(k)
        truncated_taylor_around_0 *= np.exp(-small_values)

        truncated_taylor_around_mean = np.sqrt(mean) - np.sqrt(mean)**(-0.5)/8 + np.sqrt(mean)**(-1.5)/16

        expectation = nonzeros*(truncated_taylor_around_0 * (mean < 4) + truncated_taylor_around_mean * (mean >= 4))
        
    else:
        tot = np.zeros(mean.shape)
        for i in range(samples):
            tot += np.sqrt(np.random.poisson(mean))
        expectation = tot/samples
    return expectation


def expected_log1p(mean, samples=None):
    """Return expected square root of a poisson distribution. Expects ndarray input.
    
    If samples = None, uses Taylor series centered at 0 or mean, as appropriate.
    
    If samples = int, uses that many samples for an empirical distribution."""
    
    if samples is None:
        raise NotImplementedError
    else:
        tot = np.zeros(mean.shape)
        for i in range(samples):
            tot += np.log1p(np.random.poisson(mean))
        expectation = tot/samples
    return expectation