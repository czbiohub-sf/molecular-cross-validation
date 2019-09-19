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
    name="molecular-cross-validation",
    version="0.1",
    license="MIT License",
    description="Self-supervised calibration for denoising single-cell gene expression",
    long_description=read("README.md"),
    author="Josh Batson, James Webber",
    author_email="josh.batson@czbiohub.org",
    url="https://github.com/czbiohub/molecular-cross-validation",
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
            "autoencoder_sweep = molecular_cross_validation.scripts.autoencoder_sweep:main",
            "diffusion_sweep = molecular_cross_validation.scripts.diffusion_sweep:main",
            "pca_sweep = molecular_cross_validation.scripts.pca_sweep:main",
            "magic_sweep = molecular_cross_validation.scripts.magic_sweep:main",
            "process_h5ad = molecular_cross_validation.scripts.process_h5ad:main",
            "simulate_dataset = molecular_cross_validation.scripts.simulate_dataset:main",
        ]
    },
)
