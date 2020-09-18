#!/usr/bin/env bash
set -xe

pip install --upgrade pip

# 1. Install PyTorch
conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch

# install jupyterlab, umap, altair, scanpy
conda install -y matplotlib=3.0 jupyterlab numpy scipy umap-learn nodejs scanpy -c bioconda -c conda-forge
pip install magic-impute

git clone https://github.com/czbiohub/simscity.git
(cd simscity && pip install .)
cd ${HOME}

git clone https://github.com/czbiohub/molecular-cross-validation.git
(cd molecular-cross-validation && pip install -e .)
cd ${HOME}
