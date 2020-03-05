import scanpy as sc
from sklearn.metrics import mean_squared_error
from scipy.sparse import issparse, dok_matrix

import numpy as np
from tqdm import tqdm


from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd
from scipy.spatial.distance import pdist, squareform


def split_adata(adata, p=0.5):
    X = adata.raw.X
    if issparse(X):
        X = np.array(X.todense())
    if np.allclose(X, X.astype(np.int)):
        X = X.astype(np.int)
    else:
        raise TypeError(
            "Molecular cross-validation requires integer count data.")

    X1 = np.random.binomial(X, p).astype(np.float)
    X2 = X - X1

    adata1 = sc.AnnData(X=X1)
    adata2 = sc.AnnData(X=X2)

    return adata1, adata2

def sweep_pca_mcv2(base_adata, max_pcs=30, n_neighbors=100):
    adata = base_adata.copy()
    adata1, adata2 = split_adata(adata, p=0.5)

    sc.pp.sqrt(adata1)
    sc.pp.sqrt(adata2)

    sc.tl.pca(adata1, n_comps=max_pcs, zero_center=False, random_state=1)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(adata1.obsm['X_pca'])
    adjacency = nbrs.kneighbors_graph(adata1.obsm['X_pca'], mode="connectivity")
    adjacency = np.array(adjacency.todense())
    adjacency = adjacency + adjacency.T
    adjacency -= np.diag(np.diag(adjacency))
    adjacency = squareform(adjacency > 0)

    k_range = pca_range(max_pcs)

    denoised = []

    for i, k in enumerate(tqdm(k_range)):

        reconstruction1 = adata1.obsm['X_pca'][:, :k].dot(
            adata1.varm['PCs'].T[:k])

        mcv = mean_squared_error(reconstruction1, adata2.X)

        deviations = reconstruction1 - adata2.X
        pairwise_distances = pdist(deviations, metric='euclidean')**2

        mcv2 = pairwise_distances.mean()
        mcv2_local = pairwise_distances[adjacency].mean()

        adata_denoised = sc.AnnData(X=reconstruction1,
                                    obsm={'X_deviations': deviations},
                                    uns={'denoiser': 'pca',
                                         'denoiser_param': k,
                                         'denoiser_model': None,
                                         'mcv': mcv,
                                         'mcv2': mcv2,
                                         'mcv2_local': mcv2_local
                                         },
                                    obs=base_adata.obs)
        denoised.append(adata_denoised)
    return denoised

def sweep_pca(base_adata, max_pcs=30, p=0.9):
    adata = base_adata.copy()
    adata1, adata2 = split_adata(adata, p)

    sc.tl.pca(adata, n_comps=max_pcs, zero_center=False, random_state=1)
    sc.tl.pca(adata1, n_comps=max_pcs, zero_center=False, random_state=1)

    k_range = pca_range(max_pcs)

    denoised = []
    for i, k in enumerate(tqdm(k_range)):
        reconstruction = adata.obsm['X_pca'][:, :k].dot(adata.varm['PCs'].T[:k])
        reconstruction = np.maximum(reconstruction, 0)

        reconstruction1 = adata1.obsm['X_pca'][:, :k].dot(
            adata1.varm['PCs'].T[:k])
        mcv = mean_squared_error(reconstruction1 * (1 - p) / p, adata2.X)

        adata_denoised = sc.AnnData(X=reconstruction,
                                    uns={'denoiser': 'pca',
                                         'denoiser_param': k,
                                         'denoiser_model': None,
                                         'mcv': mcv
                                         },
                                    obsm={'X_pca': adata.obsm['X_pca'][:, :k],
                                          'X_latent': adata.obsm['X_pca'][:,
                                                      :k]},
                                    obs=base_adata.obs)
        denoised.append(adata_denoised)
    return denoised


from molecular_cross_validation.util import convert_expectations

def sweep_pca_sqrt(base_adata, max_pcs=30, p=0.9, save_reconstruction=True):
    adata = base_adata.copy()
    adata1, adata2 = split_adata(adata, p)

    sc.pp.sqrt(adata1)
    sc.pp.sqrt(adata2)
    sc.pp.sqrt(adata)

    sc.tl.pca(adata, n_comps=max_pcs, zero_center=False, random_state=1)
    sc.tl.pca(adata1, n_comps=max_pcs, zero_center=False, random_state=1)

    k_range = pca_range(max_pcs)

    denoised = []
    for i, k in enumerate(tqdm(k_range)):
        reconstruction = adata.obsm['X_pca'][:, :k].dot(adata.varm['PCs'].T[:k])
        reconstruction = np.maximum(reconstruction, 0)

        reconstruction1 = adata1.obsm['X_pca'][:, :k].dot(
            adata1.varm['PCs'].T[:k])
        mcv = mean_squared_error(convert_expectations(reconstruction1, p, 1 - p), adata2.X)

        if not save_reconstruction:
            reconstruction = dok_matrix(reconstruction.shape)

        adata_denoised = sc.AnnData(X=reconstruction,
                                    uns={'denoiser': 'pca',
                                         'denoiser_param': k,
                                         'denoiser_model': None,
                                         'mcv': mcv
                                         },
                                    obsm={'X_pca': adata.obsm['X_pca'][:, :k],
                                          'X_latent': adata.obsm['X_pca'][:,
                                                      :k]},
                                    obs=base_adata.obs)
        denoised.append(adata_denoised)
    return denoised


# Helper function for the diffusion denoiser, computes the diffusion operator
# as a lazy walk over a k-nearest-neighbors graph of the data in PC space.
def compute_diff_op(
        umis: np.ndarray,
        n_components: int = 30,
        n_neighbors: int = 10,
        tr_prob: float = 0.5,
):
    n_counts = np.median(umis.sum(axis=1))

    x1_norm = np.sqrt(umis / umis.sum(axis=1, keepdims=True) * n_counts)

    U, S, V = randomized_svd(x1_norm, n_components)

    p = U.dot(np.diag(S))

    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(p)

    diff_op = np.array(nbrs.kneighbors_graph(p, mode="connectivity").todense())
    diff_op += diff_op.T
    diff_op = diff_op / diff_op.sum(axis=1, keepdims=True)
    diff_op = tr_prob * diff_op + (1 - tr_prob) * np.eye(diff_op.shape[0])

    return diff_op


# Diffuse gene values using a lazy walk over a kNN graph. First,
# calculates a kNN graph in PC space and uses that to define a diffusion
# operator. Next, iteratively smoothes the input data according to the
# operator, and returns the result.
def gene_diffusion(umis: np.ndarray, time_steps: int, **diff_op_kwargs):
    # calculate the diffusion operator, e.g. a weighting over a kNN graph
    diff_op = compute_diff_op(umis, **diff_op_kwargs)
    diffused_umis = umis.copy().astype(np.float)

    # perform diffusion over the knn graph
    for t in range(time_steps):
        diffused_umis = diff_op.dot(diffused_umis)

    return diffused_umis


from scipy.special import loggamma


def poisson_nll_loss(y_pred: np.ndarray, y_true: np.ndarray,
                     per_gene=False) -> float:
    '''likelihood is e^-y y^k/k! with log   -y + k log y - log(k!)'''
    assert (np.all(y_true >= 0))
    if per_gene:
        lik = (y_pred - y_true * np.log(y_pred + 1e-6) +
               loggamma(y_true + 1)).mean(axis=0)
    else:
        lik = (y_pred - y_true * np.log(y_pred + 1e-6) +
               loggamma(y_true + 1)).mean()
    return lik


def sweep_diffusion(base_adata, max_t=10, p=0.9):
    adata = base_adata.copy()
    adata1, adata2 = split_adata(adata, p)

    diff_op = compute_diff_op(adata.X)
    diff_op1 = compute_diff_op(adata1.X)

    diffused_umis = adata.X.copy().astype(np.float)
    diffused_umis1 = adata1.X.copy().astype(np.float)

    t_range = np.arange(max_t)

    denoised = []
    for i, t in enumerate(tqdm(t_range)):
        diffused_umis = diff_op.dot(diffused_umis)
        diffused_umis1 = diff_op1.dot(diffused_umis1)

        mcv = poisson_nll_loss(diffused_umis1 * (1 - p) / p, adata2.X)

        adata_denoised = sc.AnnData(X=diffused_umis,
                                    uns={'denoiser': 'diffusion',
                                         'denoiser_param': t,
                                         'denoiser_model': None,
                                         'mcv': mcv
                                         },
                                    obs=base_adata.obs)
        denoised.append(adata_denoised)
    return denoised


def sweep_diffusion_per_gene(base_adata, max_t=10, p=0.9):
    adata = base_adata.copy()
    adata1, adata2 = split_adata(adata, p)

    diff_op = compute_diff_op(adata.X)
    diff_op1 = compute_diff_op(adata1.X)

    diffused_umis = adata.X.copy().astype(np.float)
    diffused_umis1 = adata1.X.copy().astype(np.float)

    best_diffused_umis = diffused_umis.copy()
    best_diffused_umis1 = diffused_umis1.copy()

    t_range = np.arange(max_t)

    mcv_losses = np.zeros((max_t, diffused_umis.shape[1]))

    for i, t in enumerate(tqdm(t_range)):
        # Compute mcv for each gene for current diffusion
        mcv_losses[i] = poisson_nll_loss(diffused_umis1 * (1 - p) / p, adata2.X,
                                         per_gene=True)

        # Where is this best yet? Update those values
        new_best_idx = (np.min(mcv_losses[:(i + 1)], axis=0) == mcv_losses[i])
        best_diffused_umis[:, new_best_idx] = diffused_umis[:, new_best_idx]
        best_diffused_umis1[:, new_best_idx] = diffused_umis1[:, new_best_idx]

        # Take a step
        diffused_umis = diff_op.dot(diffused_umis)
        diffused_umis1 = diff_op1.dot(diffused_umis1)

    mcv = poisson_nll_loss(best_diffused_umis1 * (1 - p) / p, adata2.X)

    adata_denoised = sc.AnnData(X=best_diffused_umis,
                                uns={'denoiser': 'diffusion per gene',
                                     'denoiser_model': None,
                                     'mcv': mcv,
                                     'best_t': np.argmin(mcv_losses, axis=0)
                                     },
                                obs=base_adata.obs)
    denoised = [adata_denoised]
    return denoised

def recipe_seurat(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10, zero_center=False)
    return adata

def recipe_sqrt(adata):
    sc.pp.sqrt(adata)
    return adata

def recipe_sqrt_norm(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.sqrt(adata)
    return adata


def pca_range(max_pcs):
    return np.concatenate([np.arange(2, np.minimum(10, max_pcs+1), 1),
                           np.arange(10, np.minimum(30, max_pcs+1), 2),
                           np.arange(30, np.minimum(100, max_pcs+1), 5),
                           np.arange(100, np.minimum(500, max_pcs+1), 20)])

def mcv_sweep_recipe_pca(base_adata, recipe, max_pcs=30, save_reconstruction=True):
    adata = base_adata.copy()
    adata1, adata2 = split_adata(adata, 0.5)

    adata1 = recipe(adata1)
    adata2 = recipe(adata2)

    adata = recipe(adata)

    sc.tl.pca(adata, n_comps=max_pcs, zero_center=False, random_state=1)
    sc.tl.pca(adata1, n_comps=max_pcs, zero_center=False, random_state=1)

    k_range = pca_range(max_pcs)

    denoised = []
    for i, k in enumerate(tqdm(k_range)):
        reconstruction = adata.obsm['X_pca'][:, :k].dot(adata.varm['PCs'].T[:k])
        reconstruction = np.maximum(reconstruction, 0)

        reconstruction1 = adata1.obsm['X_pca'][:, :k].dot(
            adata1.varm['PCs'].T[:k])
        mcv = mean_squared_error(reconstruction1, adata2.X)

        if not save_reconstruction:
            reconstruction = dok_matrix(reconstruction.shape)

        adata_denoised = sc.AnnData(X=reconstruction,
                                    uns={'denoiser': 'pca',
                                         'denoiser_param': k,
                                         'denoiser_model': None,
                                         'mcv': mcv
                                         },
                                    obsm={'X_pca': adata.obsm['X_pca'][:, :k],
                                          'X_latent': adata.obsm['X_pca'][:, :k]},
                                    obs=base_adata.obs)
        denoised.append(adata_denoised)
    return denoised


def highly_variable(base_adata):
    adata = base_adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=None,
                                min_disp=0.5)
    print(f"{adata.var['highly_variable'].sum()} highly variable genes.")

    base_adata.var['highly_variable'] = adata.var['highly_variable']