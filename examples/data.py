import scanpy as sc

DATASETS = ['kidney', 'blood', 'symsim2k', 'bipolar']

def load_data(dataset, data_dir='/Users/josh/src/molecular-cross-validation/data'):
    assert dataset in DATASETS

    if dataset == 'kidney':
        adata = sc.read(
            data_dir + '/tabula-muris-senis/tabula-muris-senis-droplet-processed-official-annotations-Kidney.h5ad')
        adata.obs['cell_type'] = adata.obs['cell_ontology_class']

    if dataset == 'blood':
        adata = sc.read(
            data_dir + '/blood/10x_blood_data_subset_1k_cells.h5ad')

    if dataset == 'bipolar':
        adata = sc.read(
            data_dir + '/bipolar/bipolar.h5ad')
        adata.obs['cell_type'] = adata.obs['CLUSTER']

    if 'symsim' in dataset:
        adata = sc.read(
            data_dir + '/symsim/' + dataset + '.h5ad')
        adata.obs['cell_type'] = adata.obs['pop']

    X = adata.X if adata.raw is None else adata.raw.X
    adata = sc.AnnData(X=X, obs=adata.obs)
    adata.raw = adata

    return adata

def load_kidney():
    adata = sc.read(
        '/Users/josh/src/molecular-cross-validation/data/tabula-muris-senis/tabula-muris-senis-droplet-processed-official-annotations-Kidney.h5ad')
    adata.obs['cell_type'] = adata.obs['cell_ontology_class']
    adata = sc.AnnData(X = adata.raw.X, obs = adata.obs)
    adata.raw = adata

    return adata

def load_blood():
    adata = sc.read('/Users/josh/src/molecular-cross-validation/data/blood/10x_blood_data_subset_1k_cells.h5ad')
    adata.raw = adata
    return adata

def load_bipolar():
    adata = sc.read('/Users/josh/src/molecular-cross-validation/data/bipolar/bipolar.h5ad')
    adata.raw = adata
    adata.obs['cell_type'] = adata.obs['CLUSTER']
    return adata

def load_symsim(suffix=''):
    adata = sc.read('/Users/josh/src/molecular-cross-validation/data/symsim/symsim' + suffix + '.h5ad')
    adata.raw = adata
    adata.obs['cell_type'] = adata.obs['pop']
    return adata

def load_citeseq():
    adata = sc.read('/Users/josh/src/molecular-cross-validation/data/citeseq/bm.cite.h5ad')
    adata = sc.AnnData(X = adata.raw.X, obs = adata.obs)
    adata.X = np.array(adata.X.todense())
    return adata
