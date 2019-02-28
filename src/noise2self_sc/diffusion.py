import magic
import numpy as np
from scipy.sparse import issparse
from util import split_counts, normalize_rows, mse
from matplotlib import pyplot as plt


def sweep_time(diff_op, X1, X2, max_t):
    t_range = np.arange(max_t)

    losses = np.zeros(max_t)
    best_t = 0
    best_loss = mse(X2, 0)
    denoised = X1
    best_denoised = denoised

    for t in t_range:
        loss = mse(denoised, X2)
        losses[t] = loss
        if loss < best_loss:
            best_loss = loss
            best_t = t
            best_denoised = denoised
        denoised = diff_op.dot(denoised)

    return best_denoised, best_t, t_range, losses


def sweep_time_per_gene(diff_op, X1, X2, max_t):

    t_range = np.arange(max_t)

    gene_losses = np.zeros((max_t, X1.shape[1]))

    denoised = X1
    best_denoised = denoised

    for t in t_range:
        gene_losses[t] = ((denoised - X2) ** 2).sum(axis=0)

        new_best_idx = np.min(gene_losses, axis=0) == gene_losses[t]
        best_denoised[:, new_best_idx] = denoised[:, new_best_idx]

        denoised = diff_op.dot(denoised)

    best_t = np.argmin(gene_losses, axis=0)

    return best_denoised, best_t, t_range, gene_losses


def n2s_magic(
    adata, max_t=20, lazy_p=0.5, plot=True, per_gene=False, verbose=True, **kwargs
):

    if adata.raw is None:
        X = adata.X
    else:
        X = adata.raw.X

    if issparse(X):
        X = np.array(X.todense())

    X1, X2 = split_counts(X)

    median_counts = np.median(X.sum(axis=1)) / 2

    X1 = np.sqrt(normalize_rows(X1, median_counts))
    X2 = np.sqrt(normalize_rows(X2, median_counts))

    magic_op = magic.MAGIC(**kwargs)
    magic_op.fit(X1)

    diff_op = np.array(magic_op.diff_op.todense())

    diff_op = lazy_p * diff_op + (1 - lazy_p) * np.eye(diff_op.shape[0])

    if per_gene:
        best_denoised, best_t, t_range, losses = sweep_time_per_gene(
            diff_op, X1, X2, max_t
        )

        if plot:
            plt.hist(best_t, bins=20)
            plt.xlabel("diffusion time")
            plt.title("Optimal per-gene diffusion times")

    else:
        best_denoised, best_t, t_range, losses = sweep_time(diff_op, X1, X2, max_t)

        if plot:
            plt.plot(t_range, losses)
            plt.xlabel("diffusion time")
            plt.ylabel("Self-Supervised Loss")
            plt.title("Sweep t")
            plt.axvline(best_t, color="k", linestyle="--")

    denoised_adata = adata.copy()
    denoised_adata.X = best_denoised

    if verbose:
        return denoised_adata, best_t, t_range, losses

    else:
        return denoised_adata
