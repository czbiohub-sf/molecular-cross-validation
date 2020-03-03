import scanpy as sc

def load_kidney():
    adata = sc.read('/Users/josh/src/molecular-cross-validation/data/tabula-muris-senis/tabula-muris-senis-droplet-processed-official-annotations-Kidney.h5ad')
    adata = sc.AnnData(X = adata.raw.X, obs = adata.obs)
    adata.raw = adata
    return adata


def load_citeseq():
    adata = sc.read('/Users/josh/src/molecular-cross-validation/data/citeseq/bm.cite.h5ad')
    adata = sc.AnnData(X = adata.raw.X, obs = adata.obs)
    adata.X = np.array(adata.X.todense())
    return adata

def load_symsim(suffix=''):
    adata = sc.read('/Users/josh/src/molecular-cross-validation/data/symsim/symsim' + suffix + '.h5ad')
    adata.raw = adata
    return adata

def load_blood():
    adata = sc.read('/Users/josh/src/molecular-cross-validation/data/blood/10x_blood_data_subset_1k_cells.h5ad')
    adata.raw = adata
    return adata