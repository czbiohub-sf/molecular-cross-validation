#!/usr/bin/env bash
set -xe

pip install --upgrade pip

# 1. Install PyTorch
conda install -y pytorch torchvision -c pytorch
if [ ${cuda} = 1 ]; then conda install -y cudatoolkit=9.0 -c pytorch; fi

# install jupyterlab, umap, altair, scanpy
conda install -y jupyterlab numpy numba pytables seaborn scikit-learn scipy statsmodels
conda install -y altair louvain python-igraph umap-learn -c conda-forge
pip install scanpy
pip install magic-impute

git clone https://github.com/czbiohub/simscity.git
(cd simscity && python setup.py install)
cd ${HOME}

#git clone https://github.com/czbiohub/noise2self-single-cell.git
#(cd noise2self-single-cell && python setup.py install)
#cd ${HOME}
