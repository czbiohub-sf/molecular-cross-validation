#!/usr/bin/env python

import io
import glob
import os

import setuptools


def read(*names, **kwargs):
    return io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ).read()


setuptools.setup(
    name="noise2self-single-cell",
    version="0.1",
    license="MIT License",
    description="Noise2Self for single-cell gene expression",
    long_description=read("README.md"),
    author="Josh Batson, James Webber",
    author_email="josh.batson@czbiohub.org",
    url="https://github.com/czbiohub/noise2self-single-cell",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    py_modules=[
        os.path.splitext(os.path.basename(path))[0] for path in glob.glob("src/*.py")
    ],
    zip_safe=False,
    install_requires=[
        "numpy",
        "torch",
        "magic-impute",
        "matplotlib",
        "pandas<0.24",
        "scanpy",
        "scikit-learn",
        "simscity",
    ],
    entry_points={
        "console_scripts": [
            "autoencoder_sweep = noise2self_sc.scripts.autoencoder_sweep:main",
            "diffusion_sweep = noise2self_sc.scripts.diffusion_sweep:main",
            "pca_sweep = noise2self_sc.scripts.pca_sweep:main",
            "process_h5ad = noise2self_sc.scripts.process_h5ad:main",
            "simulate_dataset = noise2self_sc.scripts.simulate_dataset:main",
        ]
    },
)
