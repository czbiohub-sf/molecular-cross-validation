#!/usr/bin/env python

import itertools

import altair as alt
import numpy as np
import pandas as pd

import sklearn
import sklearn.decomposition

from umap import UMAP

from typing import Sequence


def make_grid(*charts: Sequence[alt.Chart], n_cols: int) -> alt.Chart:
    it = iter(charts)
    return alt.vconcat(
        *(
            alt.hconcat(*filter(None, c))
            for c in itertools.zip_longest(*((it,) * n_cols))
        )
    )


def log_likelihood(
    train_elbo: Sequence[float], test_elbo: Sequence[float], test_int: int
) -> alt.Chart:
    x = np.arange(len(train_elbo))

    d = pd.concat(
        [
            pd.DataFrame({"x": x, "y": train_elbo, "run": "train"}),
            pd.DataFrame({"x": x[::test_int], "y": test_elbo, "run": "test"}),
        ]
    )

    return (
        alt.Chart(d)
        .encode(x="x:Q", y="y:Q", color="run:N")
        .mark_line(point=True)
        .properties(height=240, width=240)
    )


def scatter(
    gene_values: np.ndarray,
    gene_names: Sequence[str],
    g_x: str,
    g_y: str,
    g_z: str,
    title: str = None,
) -> alt.Chart:
    """Scatter two genes against each other and color the point by a third gene

    :param gene_values: an array of shape (n_cells, n_genes)
    :param gene_names: the gene names, in the order of ``gene_values``
    :param g_x: gene to plot on x axis
    :param g_y: gene to plot on y axis
    :param g_z: gene to plot on color axis
    :param title: optionial title for the plot
    """
    gene_names = np.asarray(gene_names)
    get_i = lambda g: (gene_names == g).argmax()

    x = gene_values[:, get_i(g_x)]
    y = gene_values[:, get_i(g_y)]
    z = gene_values[:, get_i(g_z)]

    bot_d, top_d = np.percentile(z, (2.5, 97.5))

    return (
        alt.Chart(pd.DataFrame({g_x: x, g_y: y, g_z: z}))
        .mark_point(opacity=0.3)
        .encode(
            x=f"{g_x}:Q",
            y=f"{g_y}:Q",
            color=alt.Color(
                f"{g_z}:Q",
                scale=alt.Scale(scheme="spectral", clamp=True, domain=(bot_d, top_d)),
            ),
        )
        .properties(title=title or f"{g_x} vs {g_y}", background="white")
    )


def pca(
    z: np.ndarray,
    depth: np.ndarray,
    labels: np.ndarray = None,
    k: int = 3,
    **chart_properties,
) -> alt.Chart:
    """Plot scatters of pairs of principal components up to a specified number.

    :param z: input data for PCA.
    :param depth: summed across columns and used to color a final scatter of PC0 v PC1.
    :param labels: categorial labels for the rows of z, if any.
    :param k: number of PCs to plot. All pairs are plotted.
    :param chart_properties: keyword arguments are passed to altair.Chart.properties
    """
    k = min(k, z.shape[1])

    pc = sklearn.decomposition.PCA(n_components=k).fit_transform(z)

    if labels is None:
        labels = np.zeros(z.shape[0])

    log_d = np.log1p(depth.sum(1))
    bot_d, top_d = np.percentile(log_d, (2.5, 97.5))

    df = {f"pc{i}": pc[:, i] for i in range(k)}
    df.update({"c": labels, "log_d": log_d})

    c = alt.Chart(pd.DataFrame(df)).properties(**chart_properties)

    return make_grid(
        *(
            c.mark_point(opacity=0.3).encode(
                x=f"pc{i}:Q", y=f"pc{j}:Q", color=alt.Color("c:N", legend=None)
            )
            for i, j in itertools.combinations(range(k), 2)
        ),
        c.mark_point(opacity=0.8).encode(
            x="pc0:Q",
            y="pc1:Q",
            color=alt.Color(
                "log_d:Q",
                scale=alt.Scale(
                    scheme="viridis", clamp=True, nice=True, domain=(bot_d, top_d)
                ),
            ),
        ),
        n_cols=4,
    )


def umap(
    z: np.ndarray,
    depth: np.ndarray,
    labels: np.ndarray = None,
    n_neighbors: int = 8,
    **chart_properties,
) -> alt.Chart:
    """Plot scatter plot of z embedded into 2D using the UMAP algorithm.

    :param z: input data for UMAP
    :param depth: summed across columns, used to color a second plot of the embedding.
    :param labels: categorial labels for the rows of z, if any.
    :param n_neighbors: number of neighbors used in UMAP algorithm
    :param chart_properties: keyword arguments are passed to altair.Chart.properties
    """
    u = UMAP(n_neighbors=n_neighbors, metric="cosine").fit_transform(z)

    if labels is None:
        labels = np.zeros(z.shape[0])

    log_d = np.log1p(depth.sum(1))
    bot_d, top_d = np.percentile(log_d, (2.5, 97.5))

    c = alt.Chart(
        pd.DataFrame({"u0": u[:, 0], "u1": u[:, 1], "c": labels, "log_d": log_d})
    ).properties(**chart_properties)

    return alt.hconcat(
        c.mark_point(opacity=0.3).encode(
            x="u0:Q", y="u1:Q", color=alt.Color("c:N", legend=None)
        ),
        c.mark_point(opacity=0.8).encode(
            x="u0:Q",
            y="u1:Q",
            color=alt.Color(
                "log_d:Q",
                scale=alt.Scale(
                    scheme="viridis", clamp=True, nice=True, domain=(bot_d, top_d)
                ),
            ),
        ),
    )
