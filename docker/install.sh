#!/usr/bin/env bash
set -xe

# 1. Install PyTorch
conda install -y pytorch cudatoolkit=10.2 -c pytorch

# install jupyterlab, umap, altair, scanpy
conda install -y pip jupyterlab nodejs scanpy -c bioconda -c conda-forge
pip install magic-impute

git clone https://github.com/czbiohub/simscity.git
(cd simscity && python setup.py install)
cd ${HOME}

git clone https://github.com/czbiohub/molecular-cross-validation.git
(cd molecular-cross-validation && python setup.py install)
cd ${HOME}
