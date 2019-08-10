import math

import numpy as np
import scipy.stats


def expected_sqrt(mean_expression: np.ndarray) -> np.ndarray:
    """Return expected square root of a poisson distribution. Uses Taylor series
     centered at 0 or mean, as appropriate.

    :param mean_expression: Array of expected mean expression values
    :return: Array of expected sqrt mean expression values
    """

    truncated_taylor_around_0 = np.zeros(mean_expression.shape)
    nonzeros = mean_expression != 0
    mean_expression = mean_expression + 1e-8
    for k in range(15):
        truncated_taylor_around_0 += (
            mean_expression ** k / math.factorial(k) * np.sqrt(k)
        )

    truncated_taylor_around_0 *= np.exp(-mean_expression)
    truncated_taylor_around_mean = (
        np.sqrt(mean_expression)
        - np.sqrt(mean_expression) ** (-0.5) / 8
        + np.sqrt(mean_expression) ** (-1.5) / 16
    )

    return nonzeros * (
        truncated_taylor_around_0 * (mean_expression < 4)
        + truncated_taylor_around_mean * (mean_expression >= 4)
    )


def convert_expectations(exp_sqrt: np.ndarray, a: float, b: float = None) -> np.ndarray:
    """Takes expected sqrt expression calculated for one scaling factor and converts
    to the corresponding levels at a second scaling factor

    :param exp_sqrt: Expected sqrt values calculated at ``scale``
    :param a: Input scaling factor of the data
    :param b: Scale for the output. Set to ``1 - a`` by default
    :return: A scaled array of expected sqrt expression
    """
    if b is None:
        b = 1.0 - a
    if a == b:
        return exp_sqrt

    exp_sqrt = np.maximum(exp_sqrt, 0)
    max_val = np.max(exp_sqrt ** 2) / a

    xp = expected_sqrt(np.arange(0, max_val, 0.01) * a)
    fp = expected_sqrt(np.arange(0, max_val, 0.01) * b)

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
