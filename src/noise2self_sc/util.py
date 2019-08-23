from typing import Tuple, Union

import numpy as np
import scipy.stats
import scipy.special

import numba as nb


# caching some values for efficiency
taylor_range = np.arange(1, 30)
taylor_factors = scipy.special.factorial(taylor_range) / np.sqrt(taylor_range)


@nb.vectorize([nb.float64(nb.float64)], identity=0.0, target="parallel")
def taylor_expand(x):
    return (x ** taylor_range / taylor_factors).sum()


@nb.vectorize([nb.float64(nb.float64)], identity=0.0, target="parallel")
def taylor_around_mean(x):
    return np.sqrt(x) - x ** (-0.5) / 8 + x ** (-1.5) / 16 - 5 * x ** (-2.5) / 128


def expected_sqrt(mean_expression: np.ndarray, cutoff: float = 34.94) -> np.ndarray:
    """Return expected square root of a poisson distribution. Uses Taylor series
     centered at 0 or mean, as appropriate.

    :param mean_expression: Array of expected mean expression values
    :param cutoff: point for switching between approximations (default is ~optimal)
    :return: Array of expected sqrt mean expression values
    """

    nonzeros = mean_expression != 0
    mean_expression = mean_expression + 1e-8
    clipped_mean_expression = np.minimum(mean_expression, cutoff)

    truncated_taylor_around_0 = taylor_expand(clipped_mean_expression)
    truncated_taylor_around_0 *= np.exp(-mean_expression)

    truncated_taylor_around_mean = taylor_around_mean(mean_expression)

    return nonzeros * (
        truncated_taylor_around_0 * (mean_expression < cutoff)
        + truncated_taylor_around_mean * (mean_expression >= cutoff)
    )


def convert_expectations(
    exp_sqrt: np.ndarray, a: float, b: Union[float, np.ndarray] = None
) -> np.ndarray:
    """Takes expected sqrt expression calculated for one scaling factor and converts
    to the corresponding levels at a second scaling factor

    :param exp_sqrt: Expected sqrt values calculated at ``scale``
    :param a: Input scaling factor of the data
    :param b: Scale for the output. Set to ``1 - a`` by default
    :return: A scaled array of expected sqrt expression
    """
    if b is None:
        b = 1.0 - a

    exp_sqrt = np.maximum(exp_sqrt, 0)
    max_val = np.max(exp_sqrt ** 2) / a

    vs = 2 ** np.arange(0, np.ceil(np.log2(max_val + 1)) + 1) - 1
    p_range = np.hstack(
        [np.arange(v, vs[i + 1], 2 ** (i + 1) * 0.01) for i, v in enumerate(vs[:-1])]
    )

    xp = expected_sqrt(p_range * a)
    fp = expected_sqrt(p_range * b)

    if isinstance(b, np.ndarray):
        interps = np.empty_like(exp_sqrt)
        for i in range(exp_sqrt.shape[0]):
            interps[i, :] = np.interp(exp_sqrt[i, :], xp, fp[i, :])

        return interps
    else:
        return np.interp(exp_sqrt, xp, fp)


def poisson_fit(umis: np.ndarray) -> np.ndarray:
    """Takes an array of UMI counts and calculates per-gene deviation from a poisson
    distribution representing even expression across all cells.

    :param umis: Unscaled UMI counts for ``n_cells * n_genes``
    :return: An array of p-values of size ``n_genes``
    """
    n_cells = umis.shape[0]
    pct = (umis > 0).sum(0) / n_cells
    exp = umis.sum(0) / umis.sum()
    numis = umis.sum(1)

    prob_zero = np.exp(-np.dot(exp[:, None], numis[None, :]))
    exp_pct_nz = (1 - prob_zero).mean(1)

    var_pct_nz = (prob_zero * (1 - prob_zero)).mean(1) / n_cells
    std_pct_nz = np.sqrt(var_pct_nz)

    exp_p = np.ones_like(pct)
    ix = std_pct_nz != 0
    exp_p[ix] = scipy.stats.norm.cdf(pct[ix], loc=exp_pct_nz[ix], scale=std_pct_nz[ix])

    return exp_p


def overlap_correction(
    split: float,
    sample_counts: Union[float, np.ndarray],
    true_count: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Computes expected overlap between two independent draws from a cell generated by
    partitioning a sample into two groups, taking into account the finite size of the
    original cell.

    Calculating this factor is recommended if the sample counts are believed to be a
    significant fraction of the total molecules in the original cell.

    :param split: Proportion of the sample going into the first group
    :param sample_counts: The number of molecules in each cell
    :param true_count: Estimated number of molecules in the original cells
    :return: The correction factor by which to increase the second sample
    """
    split_complement = (1 - split) / (1 - split * sample_counts / true_count)
    return split + split_complement - 1


def split_molecules(
    umis: np.ndarray,
    split: float,
    overlap: float = 0.0,
    random_state: np.random.RandomState = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Splits molecules into two (overlapping) groups.

    :param umis: Array of molecules to split
    :param split: Proportion of molecules to assign to the first group
    :param overlap: Overlap correction factor, if desired
    :param random_state: For reproducible sampling
    :return: umis_X and umis_Y, representing ``split`` and ``~(1 - split)`` counts
             sampled from the input array
    """
    if random_state is None:
        random_state = np.random.RandomState()

    umis_X_disjoint = random_state.binomial(umis, split - overlap)
    umis_Y_disjoint = random_state.binomial(
        umis - umis_X_disjoint, (1 - split) / (1 - split + overlap)
    )
    overlap = umis - umis_X_disjoint - umis_Y_disjoint
    umis_X = umis_X_disjoint + overlap
    umis_Y = umis_Y_disjoint + overlap

    return umis_X, umis_Y
