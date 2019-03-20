#!/usr/bin/env python

import numpy as np

from simscity import latent, drug, sequencing


def simulate_classes(
    n_classes: int,
    n_latent: int,
    n_cells: int,
    n_features: int,
    prog_kw: dict = None,
    class_kw: dict = None,
    library_kw: dict = None,
):
    prog_kw = (
        dict(scale=1.0 / np.sqrt(n_features), sparsity=1.0)
        if prog_kw is None
        else prog_kw.copy()
    )
    class_kw = (
        dict(scale=1.0 / np.sqrt(n_latent), sparsity=1.0)
        if class_kw is None
        else class_kw.copy()
    )
    library_kw = (
        dict(loc=np.log(0.1 * n_features), scale=0.5)
        if library_kw is None
        else library_kw.copy()
    )

    assert n_cells // n_classes == n_cells / n_classes

    n_cells_per_class = n_cells // n_classes

    programs = latent.gen_programs(n_latent, n_features, **prog_kw)

    classes = latent.gen_classes(n_latent, n_classes, **class_kw)

    class_labels = np.tile(np.arange(n_classes), n_cells_per_class)

    latent_exp = np.empty((n_cells_per_class, n_classes, n_latent))
    for i in np.arange(n_classes):
        latent_exp[:, i, :] = latent.gen_class_samples(n_cells_per_class, classes[i, :])

    exp = np.dot(latent_exp, programs)

    lib_size = sequencing.library_size((n_cells_per_class, n_classes), **library_kw)

    umis = sequencing.umi_counts(np.exp(exp), lib_size=lib_size)

    return (
        latent_exp.reshape(n_cells, n_latent),
        class_labels,
        programs,
        lib_size.reshape(n_cells),
        umis.reshape(n_cells, n_features),
    )
