#!/usr/bin/env bash
set -xe

pip install --upgrade pip

# 1. Install PyTorch
conda install -y pytorch torchvision cudatoolkit=9.2 -c pytorch

# install jupyterlab, umap, altair, scanpy
conda install -y jupyterlab numpy numba pytables seaborn scikit-learn scipy statsmodels
conda install -y altair louvain python-igraph umap-learn -c conda-forge
pip install scanpy
pip install magic-impute

git clone https://github.com/czbiohub/simscity.git
(cd simscity && python setup.py install)
cd ${HOME}

git clone https://github.com/czbiohub/molecular-cross-validation.git
(cd molecular-cross-validation && python setup.py install)
cd ${HOME}
