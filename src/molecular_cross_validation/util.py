#!/usr/bin/env python

from typing import Callable, Tuple, Union

import numpy as np
import scipy.stats
import scipy.special

import numba as nb


# caching some values for efficiency
coefficient_range = np.arange(1, 128)
sqrt_expansion_coefficients = scipy.special.factorial(coefficient_range) / np.sqrt(
    coefficient_range
)
log1p_expansion_coefficients = scipy.special.factorial(coefficient_range) / np.log1p(
    coefficient_range
)


@nb.vectorize([nb.float64(nb.float64)], target="parallel")
def sqrt_poisson_around_zero(x):
    return np.exp(-x) * (x ** coefficient_range / sqrt_expansion_coefficients).sum()


@nb.vectorize([nb.float64(nb.float64)], target="parallel")
def sqrt_poisson_around_mean(x):
    return np.sqrt(x) - x ** (-0.5) / 8 + x ** (-1.5) / 16 - 5 * x ** (-2.5) / 128


@nb.vectorize([nb.float64(nb.float64)], target="parallel")
def log1p_poisson_around_zero(x):
    return np.exp(-x) * (x ** coefficient_range / log1p_expansion_coefficients).sum()


@nb.vectorize([nb.float64(nb.float64)], target="parallel")
def log1p_poisson_around_mean(x):
    """
    Use Taylor expansion of log(1 + y) around y = x to evaluate
    the expected value if y ~ Poisson(x). Note that the central 2nd and 3rd
    moments of Poisson(x) are both equal to x, and that the second and third
    derivatives of log(1 + y) are -(1 + y)**(-2) and 2*(1 + y)**(-3).

    :param x: mean of poisson
    :return: expected value of log(pseudocount + x)
    """
    return np.log1p(x) - x * (1.0 + x) ** (-2) / 2 + x * (1.0 + x) ** (-3) / 3


def expected_sqrt(mean_expression: np.ndarray, cutoff: float = 85.61) -> np.ndarray:
    """Return expected square root of a poisson distribution. Uses Taylor series
     centered at 0 or mean, as appropriate.

    :param mean_expression: Array of expected mean expression values
    :param cutoff: point for switching between approximations (default is ~optimal)
    :return: Array of expected sqrt mean expression values
    """
    above_cutoff = mean_expression >= cutoff

    truncated_taylor = sqrt_poisson_around_zero(np.minimum(mean_expression, cutoff))
    truncated_taylor[above_cutoff] = sqrt_poisson_around_mean(
        mean_expression[above_cutoff]
    )

    return truncated_taylor


def expected_log1p(mean_expression: np.ndarray, cutoff: float = 86.53) -> np.ndarray:
    """Return expected log1p of a poisson distribution. Uses Taylor series
     centered at 0 or mean, as appropriate.

    :param mean_expression: Array of expected mean expression values
    :param cutoff: point for switching between approximations (default is ~optimal)
    :return: Array of expected sqrt mean expression values
    """
    above_cutoff = mean_expression >= cutoff

    truncated_taylor = log1p_poisson_around_zero(np.minimum(mean_expression, cutoff))
    truncated_taylor[above_cutoff] = log1p_poisson_around_mean(
        mean_expression[above_cutoff]
    )

    return truncated_taylor


def convert_expectations(
    exp_values: np.ndarray,
    expected_func: Callable,
    max_val: float,
    a: Union[float, np.ndarray],
    b: Union[float, np.ndarray] = None,
) -> np.ndarray:
    """Given an estimate of the mean of f(X) where X is a Poisson random variable, this
    function will scale those estimates from scale ``a`` to ``b`` by using the function
    ``expected_func`` to calculate a grid of values over the relevant range defined by
    ``[0, max_val]``. Used by the methods below for scaling sqrt count and log1p counts.

    :param exp_values: The estimated mean of f(X) calculated at scale ``a``
    :param expected_func: Function to esimate E[f(X)] from a Poisson mean.
    :param max_val: Largest count relevant for computing interpolation. Using a very
        high value will use more memory but is otherwise harmless.
    :param a: Scaling factor(s) of the input data
    :param b: Scale for the output. Set to ``1 - a`` by default
    :return: A scaled array of mean expression values
    """
    if b is None:
        b = 1.0 - a

    # this code creates a grid of values for computing the interpolation arrays. We use
    # exponentially growing space between points to save memory
    vs = 2 ** np.arange(0, np.ceil(np.log2(max_val + 1)) + 1) - 1
    p_range = np.hstack(
        [np.arange(v, vs[i + 1], 2 ** (i + 1) * 0.01) for i, v in enumerate(vs[:-1])]
    )

    xp = expected_func(p_range * a)
    fp = expected_func(p_range * b)

    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        xp = np.broadcast_to(xp, (exp_values.shape[0], p_range.shape[0]))
        fp = np.broadcast_to(fp, (exp_values.shape[0], p_range.shape[0]))

        interps = np.empty_like(exp_values)
        for i in range(exp_values.shape[0]):
            interps[i, :] = np.interp(exp_values[i, :], xp[i, :], fp[i, :])

        return interps
    else:
        return np.interp(exp_values, xp, fp)


def convert_exp_sqrt(
    exp_sqrt: np.ndarray,
    a: Union[float, np.ndarray],
    b: Union[float, np.ndarray] = None,
) -> np.ndarray:
    """Takes estimated sqrt expression calculated for one scaling factor and converts
    to the corresponding levels at a second scaling factor.

    :param exp_sqrt: Estimated sqrt values calculated at scale ``a``. Negative values
        are set to zero.
    :param a: Scaling factor(s) of the input data
    :param b: Scale for the output. Set to ``1 - a`` by default
    :return: A scaled array of estimated sqrt expression
    """
    if b is None:
        b = 1.0 - a

    exp_sqrt = np.maximum(exp_sqrt, 0)
    max_val = np.max(exp_sqrt) ** 2 / np.min(a)

    return convert_expectations(exp_sqrt, expected_sqrt, max_val, a, b)


def convert_exp_log1p(
    exp_log1p: np.ndarray,
    a: Union[float, np.ndarray],
    b: Union[float, np.ndarray] = None,
) -> np.ndarray:
    """Takes estimated log1p expression calculated for one scaling factor and converts
    to the corresponding levels at a second scaling factor

    :param exp_log1p: Estimated log1p values calculated at scale ``a``. Negative values
        are set to zero.
    :param a: Scaling factor(s) of the input data
    :param b: Scale for the output. Set to ``1 - a`` by default
    :return: A scaled array of estimated log1p expression
    """
    if b is None:
        b = 1.0 - a

    exp_log1p = np.maximum(exp_log1p, 0)
    max_val = np.exp(np.max(exp_log1p)) / np.min(a)

    return convert_expectations(exp_log1p, expected_log1p, max_val, a, b)


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
    data_split: float, sample_ratio: Union[float, np.ndarray] = None
) -> Union[float, np.ndarray]:
    """Computes expected overlap between two independent draws from a cell generated by
    partitioning a sample into two groups, taking into account the finite size of the
    original cell.

    Calculating this factor is recommended if the sample counts are believed to be a
    significant fraction of the total molecules in the original cell.

    :param data_split: Proportion of the sample going into the first group
    :param sample_ratio: Ratio of counts in the sample compared to the original cells
    :return: Adjusted values for data_split, and the overlap correction factor
    """
    if sample_ratio is None or np.all(sample_ratio == 0.0):
        return data_split, 1 - data_split, 0.0

    a = (1 - data_split) / data_split

    p = ((1 + a) - np.sqrt((1 + a) ** 2 - 4 * a * sample_ratio)) / (2 * a)
    q = a * p

    assert np.allclose(p + q - p * q, sample_ratio)

    new_split = p / sample_ratio
    new_split_complement = q / sample_ratio
    overlap_factor = new_split + new_split_complement - 1

    return new_split, new_split_complement, overlap_factor


def split_molecules(
    umis: np.ndarray,
    data_split: float,
    overlap_factor: float = 0.0,
    random_state: np.random.RandomState = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Splits molecules into two (potentially overlapping) groups.

    :param umis: Array of molecules to split
    :param data_split: Proportion of molecules to assign to the first group
    :param overlap_factor: Overlap correction factor, if desired
    :param random_state: For reproducible sampling
    :return: umis_X and umis_Y, representing ``split`` and ``~(1 - split)`` counts
             sampled from the input array
    """
    if random_state is None:
        random_state = np.random.RandomState()

    umis_X_disjoint = random_state.binomial(umis, data_split - overlap_factor)
    umis_Y_disjoint = random_state.binomial(
        umis - umis_X_disjoint, (1 - data_split) / (1 - data_split + overlap_factor)
    )
    overlap_factor = umis - umis_X_disjoint - umis_Y_disjoint
    umis_X = umis_X_disjoint + overlap_factor
    umis_Y = umis_Y_disjoint + overlap_factor

    return umis_X, umis_Y
